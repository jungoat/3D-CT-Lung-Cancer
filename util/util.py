import collections
import copy
import datetime
import gc
import time

import torch
import numpy as np

IrcTuple = collections.namedtuple('IrcTuple', ['index','row','col']) # z, y, x 좌표
XyzTuple = collections.namedtuple('XyzTuple'. ['x','y','z']) # x, y, z 좌표


# coord_irc : 인덱스 (i,r,c) = (z,y,x)
# origin_xyz : CT 시작 위치 (x, y, z) mm
# vxSize_xyz : 복셀 크기 (x, y, z 방향 크기)
# direction_a : CT의 회전 행렬 (3x3)


def irc2xyz(coord_irc, origin_xyz, vxSize_xyz, direction_a):
    cri_a = np.array(coord_irc)[::-1] # 넘파이 배열로 변환하며 순서를 바꿈,
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
# (cri_a * vxSize_a) 실제 크기로 스케일 조정함, direction_a @ : 방향에 따라 회전 보정, origin_a : CT 전체의 시작점을 더해서 위치 계산
    coords_xyz = (direction_a @ (cri_a * vxSize_a)) + origin_a
    return XyzTuple(*coords_xyz)

def xyz2irc(coord_xyz, origin_xyz, vxSize_xyz, direction_a):
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coord_a = np.array(coord_xyz)
    cri_a = ((coord_a - origin_a) @ np.linalg.inv(direction_a)) / vxSize_a
    cri_a = np.round(cri_a) # 정수로 변환 전 반올림해주기
    return IrcTuple(int(cri_a[2], int(cri_a[1],), int(cri_a[0]))) 
