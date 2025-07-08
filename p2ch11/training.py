import argparse
import datetime
import os
import sys

import numpy as np

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

from util.util import enumerateWithEstimate
from .dsets import LunaDataset
from util.logconf import logging
from .model import LunaModel

class LunaTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None: # 인자 없이 호출하면 명령행으로부터 얻는다.
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--train_num-workers', default=8, type=int,)
        parser.add_argument('--val-num-workers', default=8, type=int,)
        parser.add_argument('--train--batch-size', default=64, type=int,)
        parser.add_argument('--val-batch-size', default=64, type=int,)
        parser.add_argument('--epochs', default=1, type=int,)
        parser.add_argument('--tb-prefix',default='p2ch11',)
        parser.add_argument('comment',nargs='?',default='dwlpt',)
        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S') # 훈련회차 식별용 타임 스탬프

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0 # 전체 훈련 샘플 수 누적

        if not torch.cuda.is_availalbe():
            raise RuntimeError("cuda is not available")
        
        self.use_cuda = True
        self.device = torch.device("cuda")

        self.model = self.initModel()
        self.optimizer = self.ininOptimizer()

# 모델을 초기화하고 gpu로 올리는 메서드 호출    
    def initModel(self):
        if not self.use_cuda:
            raise RuntimeError("CUDA is not available")
        model = LunaModel().to(self.device)
        return model
    
    def initOptimizer(self):
        return SGD(self.model.parameters(), lr=0.001, momentum=0.99)
    
    def initTrainDl(self):
        train_ds = LunaDataset( # 커스텀 데이터셋
            val_stride=10, # 데이터 셋 10개중 1개를 검증용으로 나눔.
            isValSet_bool=False, # 이 데이터셋은 검증용이 아닌 훈련용
        )

        batch_size = self.cli_args.train_batch_size # 명령줄 인자를 저장한 객체. argparse

# 데이터 로더 만들기
        train_dl = DataLoader( # 바로 사용하면 되는 클래스
            train_ds,
            batch_size=batch_size, # 알아서 배치로 나눈다.
            num_workers=self.cli_args.train_num_workers,
            pin_memory=self.use_cuda, # 고정된 메모리 영역이 gpu쪽으로 빠르게 전송된다. 
        )
        return train_dl
    

    def initValDl(self):
        val_ds = LunaDataset(
            val_stride=10,
            isValSet_bool=True,
        )

        batch_size = self.cli_args.val_batch_size
        
        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.cli_arg.val_batch_size,
            pin_memory=self.use_cuda,
        )

        return val_dl