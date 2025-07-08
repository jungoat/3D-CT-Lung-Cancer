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
            