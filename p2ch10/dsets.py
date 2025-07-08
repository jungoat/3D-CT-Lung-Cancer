import copy # 객체 복사용
import csv # csv 파일 다룰때 씀
import functools # 데코레이터 및 고차 함수 지원
import glob # 파일 경우 패턴 매칭
import os # 파일 경로

from collections import namedtuple # 튜플 기반의 자료 구조

import SimpleITK as sitk # 의료 영상 처리 라이브러리
import numpy as np # 넘파이

import torch # 토치
import torch.cuda # 쿠다 관련 기능
from torch.utils.data import Dataset # 파이토치의 커스텀 데이터셋

from util.disk import getCache # 
from util.util import XyzTuple, xyz2irc
from util.logconf import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

raw_cache = getCache('part2ch10_raw') # 디스크 기반 캐시 객체 생성, 무거운 데이터를 메모리에 안들고 디스크에 압축 저장해서 빠르게 불러옴

# 통합 인터페이스 쉽게 보기좋게 만든 튜플 자료구조
CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple', # 이름
    'isNodule_bool, diameter_mm, series_uid, center_xyz' # 4개의 필드
)


@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool=True):
    mhd_list = glob.glob('data-unversioned/part2/lung/subset*/*.mhd')
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}
# 나중에 csv에서 읽을 후보들이랑 UID 포함된 것만 필터링하게 하는거임.


# annotation.csv에서 결절과 좌표에 대한 정보를 저장
    diameter_dict = {}
    with open('data/part2/luna/annotations.csv', 'r') as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0] # series_uid가 존재함
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]]) # 결절의 x,y,z좌표
            annotationDiameter_mm = float(row[4]) # 결절의 지름이 존재함.
# series_uid가 딕셔너리에 없으면 리스트를 만들어서 값을 반환한다. (setdefault)
            diameter_dict.setdefault(series_uid, []).append(
                (annotationCenter_xyz, annotationDiameter_mm)
            )

    candidateInfo_list = []
    with open('data/part2/lung/candidates.csv',"r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

# disk에 없는 ct니까 건너뛰자
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

# 튜플안에 있는 내용을 순차적으로 리스트에 넣어서 저장함.
            candidateInfo_list.append(CandidateInfoTuple(
                isNodule_bool,
                candidateDiameter_mm,
                series_uid,
                candidateCenter_xyz,
            ))

# 모든 결절을 내림차순으로 정렬하고, 그 뒤에는 결절이 아닌 샘플이 이어진 데이터가 된다.
    candidateInfo_list.sort(reverse=True)
    return candidateInfo_list

# 이제 디스크에서 CT 데이터를 가져와 파이산 객체로 변환해서 3차원 결절 밀도 데이터로 사용하게 하는 작업!
class Ct:
    def __init__(self, series_uid):
# 파일 이름을 format을 활용해서 series_uid.mhd로 만든다. [0]은 찾은 파일 리스트 중 첫 파일만 사용한다는 의미!
        mhd_path = glob.glob(
            'data-unversioned/part2/lung/subset*/{}.mhd'.format(series_uid)
        )[0]
        
        
        ct_mhd = sitk.ReadImage(mhd_path) #CT 이미지 전체를 메모리에 불러오는 작업.
# sitk.GetArrayFromImage로 ct_mhd를 numpy 배열로 변환함. 이 배열은 3d ct의 실제 픽셀 (hu)를 가짐.
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32) 
# 공기는 -1000HU, 물은 0HU, 뼈는 +1000HU임.        
        ct_a.clip(-1000, 1000, ct_a)
        
        self.series_uid = series_uid
        self.hu_a = ct_a

# 밀리미터로 나타낸 위치정보는 배열 인덱스로 넣어서는 제대로 작업이 안됨. x,y,z -> i,r,c 복셀 주소 기반 좌표계로 변환함.
        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin()) # ct원점의 실제 위치
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing()) # 복셀 하나의 크기
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3) # 각 축이 어떤 방향을 바라보는지

# xyz -> irc 표
    def getRawCandidate(self, center_xyz, width_irc):
        center_irc = xyz2irc(
            center_xyz,
            self.origin_xyz, # 시작점
            self.vxSize_xyz, # 복셀크기
            self.direction_a # 방향 정보
        )
# ct는 일반적으로 512 x 512이고, 인덱스 차원은 100~250개의 단면으로 이루어짐.
        slice_list = []
# center_irc 는 (i,r,c) = (z,y,x) 형태. axis = 0, 1, 2(z,y,x 축), center_val은 각 축의 중심 인덱스 값.
        for axis, center_val in enumerate(center_irc):
# 자르려는 영역의 시작점/끝점 인덱스 계산, width_irc[axis]: 잘라낼 범위 크기,  center_val - width/2 : 중심 기준으로 좌우 자르기.             
            start_ndx = int(round(center_val - width_irc[axis]/2))
            end_ndx = int(start_ndx + width_irc[axis])
# 중심 좌표가 ct내부에 있는지 확인. self.hu_a.shape는 (z,y,x) 축의 전체 크기.
            assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr([self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])

            if start_ndx < 0: # 시작 인덱스가 0일 경우에는 0부터 시작해요
                start_ndx = 0 
                end_ndx = int(width_irc[axis]) 

            if end_ndx > self.hu_a.shape[axis]: # 끝 인덱스가 ct의 크기를 넘을 경우
                end_ndx = self.hu_a.shape[axis] # ct크기 한계로 자른다.
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx)) # 슬라이싱 범위 저장. 총 end_ndx - start_ndx 개수 만큼.
# self.hu_a는 ct전체 3D배열 (shape=(z,y,x)). slice
        ct_chunk = self.hu_a[tuple(slice_list)] # 패치 자르기
# chunk는 슬라이스 z축만큼 있는 3D 부피덩어리
        return ct_chunk, center_irc
    

# lru_cache는 최근 사용한 결과를 캐시에 저장해서 사용하는 데코레이터
@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)


# 캐싱 데코레이터
@raw_cache.memorize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    return ct_chunk, center_irc


class LunaDataset(Dataset):
    def __init__(self,
                 val_stride=0, # val_set 분리 간격
                 isValSet_bool=None, # 이 데이터셋이 검증용인지 아닌지 나타냄
                 series_uid=None, # 특정 CT 시리즈 id하나만 필터링해서 사용할때 씀.
            ):
# 모든 CT의 결절 후보 정보 리스트를 반환하는 함수이다.
        self.candidateInfo_list = copy.copy(getCandidateInfoList())

# series_uid가 주어졌다면, 리스트에서 해당 시리즈의 결절 후보들만 남긴다.
        if series_uid:
            self.candidateInfo_list = [
                x for x in self.candidateInfo_list if x.series_uid == series_uid
            ]

# 검증용 데이터셋 만들기.
        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.candidateInfo_list = self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list # 결과 리스트가 비어있으면 안됨.
        elif val_stride > 0:
            del self.candidateInfo_list[::val_stride] # 전체 후보리스트에서 val_stride 간격으로 삭제한다
            assert self.candidateInfo_list

        log.info("{!r}: {} {} samples".format(
            self, # 현재 데이터셋 객체. !r은 self객체의 문자열 표현.
            len(self.candidateInfo_list), # 후보 수
            "validation" if isValSet_bool else "training", # 어떤 용도인지
        ))

    def __len__(self):
        return len(self.candidateInfo_list) # 결절 후보리스트의 길이를 반환.

    def __getitem__(self, ndx): # ndx는 정수 인덱스

        '''CandidateInfoTuple(
        isNodule_bool=True/False,
        diameter_mm=3.5,
        series_uid="1.2.3...",
        center_xyz=(x, y, z)'''

        candidateInfo_tup = self.candidateInfo_list[ndx] 
        width_irc = (32, 48, 48) # 슬라이스 32개, 높이 49, 너비 48로 자름.

# getCtRawCandidate는 Ct를 불러오고 center_xyz는 위치를 기준으로, width_irc크기만큼 자른 3d패치(candidate_a)를 리턴한다.
        candidate_a, center_irc = getCtRawCandidate(
            candidateInfo_tup.series_uid,
            candidateInfo_tup.center_xyz,
            width_irc,
        )

# candidate_a 는 넘파이 배열이므로 텐서로 바꿔줌.
        candidate_t = torch.from_numpy(candidate_a)
        candidate_t = candidate_t.to(torch.float32) # 타입을 float32로 바꿔줌
        candidate_t = candidate_t.unsqueeze(0) # 텐서 앞에 채널 차원을 추가 [32, 48, 48] -> [1, 32, 48, 48]

        pos_t = torch.tensor([ # 정답 레이블 생성
            not candidateInfo_tup.isNodule_bool, # 결절이면 [0,1] 아니먄 [1,0]
            candidateInfo_tup.isNodule_bool
        ],
        dtype=torch.long, # 정수향 클래스 레이블 사용.
        )

# 튜플로 반환.
        return (
            candidate_t, # 입력 CT 패치 (Tensor)
            pos_t, # 라벨 (이진)
            candidateInfo_tup.series_uid, # 시리즈 ID (문자열)
            torch.tensor(center_irc), # 중심 위치 (IRC 좌표)
        )