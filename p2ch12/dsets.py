import copy
import csv
import functools
import glob
import math
import os
import random

from collections import namedtuple

import SimpleITK as sitk
import numpy as np

import torch
import torch.cuda
import torch.nn.functional as F
from torch.utils.data import Dataset

from util.disk import getCache
from util.util import XyzTuple, xyz2irc
from util.logconf import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

raw_cache = getCache('part2ch12_raw')

CandidateInfoTuple = namedtuple('CandidateInfoTuple', 'isNodule_bool, diameter_mm, series_uid, center_xyz')

@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool=True):
    mhd_list = glob.glob('data-unversioned/part2/luna/subset*/*.mhd')
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    diameter_dict = {}
    with open('data/part2/lung/annotations.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])

            diameter_dict.setdefault(series_uid, []).append(
                (annotationCenter_xyz, annotationDiameter_mm),
            )

    candidateInfo_list = []
    with open('data/part2/luna/candidates.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue

            isNodule_bool = bool(int(row[4]))
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])

            candidateDiameter_mm = 0.0
            for annotation_tup in diameter_dict.get(series_uid, []):
                annotationCenter_xyz, annotationDiameter_mm = annotation_tup
                for i in range(3):
                    delta_mm = abs(candidateCenter_xyz[i] - annotationCenter_xyz[i])
                    if delta_mm > annotationDiameter_mm / 4:
                        break
                else:
                    candidateDiameter_mm = annotationDiameter_mm
                    break

            candidateInfo_list.append(CandidateInfoTuple(
                isNodule_bool,
                candidateDiameter_mm,
                series_uid,
                candidateCenter_xyz,
            ))

    candidateInfo_list.sort(reverse=True)
    return candidateInfo_list


class Ct:
    def __init__(self, series_uid):
        # 해당 series_uid에 해당하는 .mhd 파일 경로를 glob을 이용해 찾음.
        # 'subset*'을 포함한 모든 폴더에서 해당 UID 이름을 가진 .mhd 파일을 찾는다.
        mhd_path = glob.glob(
            'data-unversioned/part2/luna/subset*/{}.mhd'.format(series_uid)
        )[0] # 리스트에서 첫 번째 파일 경로를 사용 (보통 하나만 매칭됨)

        # SimpleITK를 이용해 .mhd 파일을 읽어들임(3D CT 이미지 로딩)
        ct_mhd = sitk.ReadImage(mhd_path)

        # sitk 이미지 객체를 numpy 배열로 변환하고, float32 타입으로 저장
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
        
        # HU 값을 -1000 ~ 1000 범위로 클리핑하여 잡음 제거
        ct_a.clip(-1000, 1000, ct_a)

        # series_uid 저장
        self.series_uid = series_uid
        self.hu_a = ct_a # 전처리된 HU 값 배열을 클래스 멤버로 저장

        # CT 이미지의 원점 좌표를 가져와 XyzTuple 형식으로 저장 (좌표계 기준)
        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        
        # CT 이미지의 voxel (복셀) 크기를 가져와 XyzTuple 형식으로 저장 (3D 픽셀 간 거리)
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())

        # 방향 정보를 3x3 배열로 가져옴. CT의 축 방향 회전 정보가 담겨 있음.
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    def getRawCandidate(self, center_xyz, width_irc): # xyz 좌표계에서 CT 이미지 배열 좌표계 irc로 변환.
        center_irc = xyz2irc(
            center_xyz, # 중심 좌표 (mm 단위, 월드 좌표)
            self.origin_xyz, # CT 이미지의 시작 위치 (원점)
            self.vxSize_xyz, # 복셀 크기
            self.direction_a, # 방향 행렬 (CT 축 회전 정보)
        )

        slice_list = [] # 각 축 (z, y, x)에 대한 슬라이싱 범위를 저장할 리스트

        for axis, center_val in enumerate(center_irc): # z, y, x 축을 하나씩 처리
            # 중심 좌표로부터 절반씩 나누어 시작/끝 인덱스를 계산
            start_ndx = int(round(center_val - width_irc[axis]/2))
            end_ndx = int(start_ndx + width_irc[axis])

            # 중심 좌표가 CT 이미지 내부에 있는지 확인 (디버깅을 위해 assert 활용)
            assert center_val >= 0 and center_val < self.hu_a.shape[axis], 
            repr([self.series_uid, 
                center_xyz, 
                self.origin_xyz, 
                self.vxSize_xyz, 
                center_irc, axis])

            # 시작 인덱스가 음수면 0으로 보정하고, 끝 인덱스는 고정된 패치 크기로 설정.
            if start_ndx < 0:
                start_ndx = 0
                end_ndx = int(width_irc[axis])


            # 끝 인덱스가 이미지 범위를 넘으면, 끝을 이미지 최대 범위로 하고 시작 보정
            if end_ndx > self.hu_a.shape[axis]:
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            # 현재 축에 대한 슬라이싱 객체(slice(start, end)) 저장
            slice_list.append(slice(start_ndx, end_ndx))

        # 슬라이싱 리스트를 이용해 HU 배열에서 해당 패치 추출
        ct_chunk = self.hu_a[tuple(slice_list)] # 3D 패치: [z 축 범위, y축 범위, x축 범위]

        # 패치와 함께 중심 좌표 (IRC 기준)를 반환
        return ct_chunk, center_irc


@functools.lru_cache(1, typed=True) # 데코레이터는 series_uid가 동일하면 다시 만들지 않고 저장된 Ct 객체 재사용.
# CT 데이터 하나를 메모리에 한번만 읽고, 다음부터는 재사용하는 함수.
def getCt(series_uid):
    return Ct(series_uid)

# 이것도 캐시를 위한 데코레이터. 같은 계산이면 다시 계산하지 않고 저장된 결과를 재사용.
@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    # ct_chunk: 잘라낸 HU 3D 배열.
    # center_irc: 중심 좌표 (IRC 좌표계 기준)
    ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    return ct_chunk, center_irc # 결과를 반환함.


def getCtAugmentedCandidate( 
        augmentation_dict, # 데이터 증강 방식을 지정한 딕셔너리
        series_uid, center_xyz, width_irc,  # CT 데이터의 고유 ID, 중심 좌표, 자를 크기
        use_cache=True): # 캐시를 사용할지 여부 (True면 메모리/디스크 캐시 활용)
    
    if use_cache:
        # 캐시된 결과가 있으면 getCtRawCandidate를 통해 패치와 중심좌표 가져옴.
        ct_chunk, center_irc = \
            getCtRawCandidate(series_uid, center_xyz, width_irc)
    else:
        # 캐시를 사용하지 않으면 매번 새로 읽고 잘라냄
        ct = getCt(series_uid)
        ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)

    # 잘라낸 패치(numpy 배열)를 torch 텐서로 변환하고 shape을 (1, 1, D, H, W)로 맞춤.
    ct_t = torch.tensor(ct_chunk).unsqueeze(0).unsqueeze(0).to(torch.float32)

    # 3D 좌표 변환을 위한 단위 행렬(4x4)을 초기화 (Affine transform용)
    transform_t = torch.eye(4)

    # z,y,x 축 각각에 대해 augmentation 적용
    for i in range(3):

        if 'flip' in augmentation_dict:
            # 'flip' augmentation이 설정돼 있다면,
            # 50% 확률로 해당 축 방향을 반전시킴(좌우반전, 상하반전 등)
            if random.random() > 0.5:
                transform_t[i,i] *= - 1 # 해당 축 방향 반전 (스케일 -1 곱하기)

        if 'offset' in augmentation_dict:
            # 'offset' augmentation이 설정돼 있다면,
            # -offset ~ +offset 범위에서 무작위 이동 적용
            offset_float = augmentation_dict['offset'] # 최대 이동 크기
            random_float = (random.random() * 2 - 1) # -1 ~ +1 사이 난수 생성

    
    if 'rotate' in augmentation_dict:
        # 만약 회전 증강을 설정했으면
        angle_rad = random.random() * math.pi * 2
        # 0 ~ 2pi 사이의 랜덤한 회전 각도를 라디안으로 생성함.
        s = math.sin(angle_rad) # 사인값 계산
        c = math.cos(angle_rad) # 코사인값 계산

        # 회전 행렬을 구성 (z축 기준 회전)
        rotation_t = torch.tensor([
            [c, -s, 0, 0], # 회전 행렬의 x축, y축 부분
            [s, c, 0, 0], # 회전 행렬의 x축, y축 부분
            [0, 0, 1, 0], # z축은 그대로
            [0, 0, 0, 1], # 동차좌표를 유지
        ])

# _t는 tensor를 보통 의미한다. ct_t 의 경우 ct 데이터를 numpy이 배열이 아닌 torch tensor로 바꾼 것
# affine는 수학에서 선형변환 + 이동을 의미한다.

        # 현재 변환 행렬에 회전 행렬을 곱하여 누적 적용 (회전 포함)
        # @= 연산은 행렬 곱셈 (기존 transform_t에 누적 곱)
        transform_t @= rotation_t

    # 최종 변환 행렬(transform_t)을 기반으로 3D affine grid 생성
    affine_t = F.affien_grid(
        transform_t[:3].unsqueeze(0).to(torch.float32), # 상위 3행만 사용 (3x4 형태로)
        ct_t.size(), # 입력 텐서 크기 (batch, channel, D, H, W)
        align_corners=False, # corner align 여부 설정
        )
    
    # affine grid를 기반으로 입력 CT 텐서를 보간해서 변환 적용
    augmented_chunk = F.grid_sample(
        ct_t, # 원본 텐서 (1, 1, D, H, W)
        affine_t, # 위치를 변환시킬 그리드
        padding_mode='border', # 바깥 범위는 가장자리 값으로 채움
        align_corners=False,
        ).to('cpu') # 연산 후 CPU로 이동 (필요 시)

    if 'noise' in augmentation_dict:
        # 노이즈 증강이 설정되어 있으면
        noise_t = torch.randn_like(augmented_chunk)
        # 원본과 동일한 shape으로 표준정규분포 난수 생성
        noise_t *= augmentation_dict['noise']
        # 지정된 노이즈 세기로 스케일링
        augmented_chunk += noise_t
        # 노이즈를 텐서에 추가 (정규분포 기반 가우시안 노이즈)

# 텐서에서 batch=1, channel=1을 제거하고 실제 패치 데이터만 반환
    return augmented_chunk[0], center_irc


class LunaDataset(Dataset):
    def __init__(self,
                 val_stried=0, # 검증 세트 분할 간격
                 isValSet_bool=None, # 현재 이 데이터셋이 train인지 val인지 판단
                 series_uid=None, # 특정 ct스캔 하나만 사용할 경우 그 때 ID
                 sortby_str='random', # 데이터 정렬 방식 ex. random이나 series_uid 등 방식으로 지정.
                 ratio_int=0, # 전체 후보 중 일부만 사용할 경우. 10이면 1/10만 사용. 불균형한 데이터 비율 맞추기 용도.
                 augmentation_dict=None, # 데이터 증강 설정 딕셔너리
                 candidateInfo_list=None, # 데이터셋으로 사용할 결절 후보 정보 리스트
                 ):
        self.ratio_int = ratio_int # 일부 샘플만 선택적으로 불러올 때 사용
        self.augmentation_dict = augmentation_dict # flip, noise, rotate 같은 증강을 적용할지 결정

        if candidateInfo_list:
            self.candidateInfo_list = copy.copy(candidateInfo_list):
            self.use_cache = False
        else:
            self.candidateInfo_list = copy.copy(getCandidateInfoList())
            self.use_cache = True

        if series_uid:
            self.candidateInfo_list = [
                x for x in self.candidateInfo_lsit if x.series_uid == series_uid
            ]