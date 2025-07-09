import argparse
import datetime
import os
import sys

import numpy as np

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter


from util.util import enumerateWithEstimate
from .dsets import LunaDataset
from util.logconf import logging
from .model import LunaModel

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_LOSS_NDX = 2
METRICS_SIZE = 3


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
    
    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join('runs', self.cli_args.tb_prefix, self.time_str)

            self.trn_writer = SummaryWriter(
                log_dir=log_dir + '-trn_cls-' + self.cli_args.comment)
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '-val_cls-' + self.cli_args.comment)
            

    def main(self): # 학습 전체 과정 메인 함수 정의
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args)) # 현재 실행 중인 클래스 이름과 CLI 인자를 로그에 출력

        train_dl = self.initTrainDl() # 학습 데이터 로더 초기화
        val_dl = self.initValDl() # 검증 데이터 로더 초기화

        for epoch_ndx in range(1, self.cli_args.epochs + 1): # 1부터 설정한 에폭 단위로 학습 시작

            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx, # 현재 에폭
                self.cli_args.epochs, # 전체 에폭 수
                len(train_dl), # 학습 배치 개수
                len(val_dl), # 검증 배치 개수
                self.cli_args.train_batch_size, # 배치 사이즈
            ))

            trnMetrics_t = self.doTraining(epoch_ndx, train_dl) # 해당 에폭에서 학습할 메트릭결과를 받아옴
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t) # 학습 결과를 로그에 저장함

            valMetrics_t = self.doValidation(epoch_ndx, val_dl) # 해당 에폭의 검증을 수행하고, 메트릭 결과를 받아옴
            self.logMetrics(epoch_ndx, 'val', valMetrics_t) # 검증 결과를 로그에 저장함.

        if hasattr(self, 'trn_writer'): # tensorboard writer가 존재하면
            self.trn_writer.close()
            self.val_writer.close() # 학습 및 검증 writer를 닫아줌 (리소스 정리)


    def doTraining(self, epoch_ndx, train_dl): # 에폭 단위로 학습을 수행하는 함수  train_dl로 1에폭 학습
        self.model.train() # 모델을 학습 모드로 설정 (Dropout, BatchNorm 등이 학습 모드로 작동함)
        trnMetrics_g = torch.zeros( 
            METRICS_SIZE, # 저장할 메트릭 종류 수
            len(train_dl.dataset), # 전체 학습 데이터 개수 만큼
            device=self.device, # gpu 설정
        )

        batch_iter = enumerateWithEstimate( 
            train_dl, # 학습 데이터 로더
            "E {} Training".format(epoch_ndx), # 진행 상황 표시
            start_ndx=train_dl.num_workers, # 진행률 추청 시 시작 인덱스
        )
        for batch_ndx, batch_tup in batch_iter: # 배치 단위로 학습 반복
            self.optimizer.zero_grad() # 이전 배치의 gradient를 초기화 (누적 방지)

            loss_var = self.computeBatchLoss(
                batch_ndx, # 현재 배치 인덱스
                batch_tup, # 현재 배치 데이터 (입력, 라벨 등)
                train_dl.batch_size, # 배치 사이즈
                trnMetrics_g # 메트릭 저장할 글로벌 텐서
            )
# 현재 배치에 대한 loss 계산 및 매트릭 기록
            loss_var.backward() # 역전파 계산 
            self.optimizer.step() # 계산된 gradient로 모델 가중치 업데이트

        self.totalTrainingSamples_count += len(train_dl.dataset) # 전체 학습 데이터 수를 누적(학습된 샘플 수 기록)

        return trnMetrics_g.tp('cpu') # gpu에서 계산된 메트릭 텐서를 cpu로 옮겨 반환 (후처리나 로깅용)
    

    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()
            valMetrics_g = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
            )

            batch_iter = enumerateWithEstimate(
                val_dl,
                "E{} Validation ".format(epoch_ndx),
                start_ndx=val_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                self.computeBatchLoss(
                    batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g)

        return valMetrics_g.to('cpu')
    

    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        # 하나의 배치(batch_tup)에 대해 손실 및 예측값 계산, 메트릭 저장까지 수행

        input_t, label_t, _series_list, _center_list = batch_tup
        # 배치 데이터 unpack
        # input_t: CT 이미지 텐서
        # label_t: 정답 라벨 텐서
        # _series_list, _center_list: 시리즈 UID 및 결절 중심 좌표 (여기선 사용 안 함)

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)
        # 입력과 정답 라벨을 gpu로 이동 (non_blocking=True로 속도 최적화)

        logits_g, probability_g = self.model(input_g)
        # 모델에 입력을 넣어 예측 수행
        # logits_g: softmax 전의 raw score (B, 2)
        # probability_g: softmax 결과 확률값 (B, 2)

        loss_func = nn.CrossEntropyLoss(reduction='none')
        # 손실 함수 정의 (reduction='none'으로 개별 샘플 손실값 반환)

        loss_g = loss_func(
            logits_g,
            label_g[:,1],  # 두 번째 열만 사용 → 실제 클래스 값 (0 or 1)
        )
        # 로짓과 정답을 비교하여 샘플별 cross entropy 손실 계산 (1D 벡터)

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_g.size(0)
        # 전체 데이터셋 기준으로 현재 배치의 시작 인덱스와 끝 인덱스 계산
        # ex: 배치 인덱스가 2고 배치 사이즈 64이면 → 128~192

        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = \
            label_g[:,1].detach()
        # 정답 라벨 저장. detach()로 계산 그래프 분리

        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = \
            probability_g[:,1].detach()
        # 예측 확률 중 클래스 1(양성)일 확률만 저장

        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = \
            loss_g.detach()
        # 개별 샘플의 손실 값 저장

        return loss_g.mean()
        # 평균 손실 반환 (역전파용)


    def logMetrics(
            self,
            epoch_ndx,                  # 현재 에폭 번호
            mode_str,                   # 'trn' 또는 'val' (학습 or 검증 모드)
            metrics_t,                  # (3, 전체 샘플 수) 형태의 메트릭 텐서
            classificationThreshold=0.5 # 0.5 기준으로 예측을 양성/음성으로 분류
    ):
        self.initTensorboardWriters()
        # TensorBoard writer 초기화 (없으면 생성)

        log.info("E{} {}".format(
            epoch_ndx,
            type(self).__name__,
        ))
        # 현재 에폭과 클래스 이름을 로그로 출력

        # 음성/양성 라벨, 예측 마스크 생성 (0.5 기준)
        negLabel_mask = metrics_t[METRICS_LABEL_NDX] <= classificationThreshold
        negPred_mask = metrics_t[METRICS_PRED_NDX] <= classificationThreshold

        posLabel_mask = ~negLabel_mask
        posPred_mask = ~negPred_mask

        # 양성/음성 샘플 개수 계산
        neg_count = int(negLabel_mask.sum())
        pos_count = int(posLabel_mask.sum())

        # 양성/음성 정확하게 예측한 개수 계산
        neg_correct = int((negLabel_mask & negPred_mask).sum())
        pos_correct = int((posLabel_mask & posPred_mask).sum())

        metrics_dict = {}
        # 손실(Loss) 관련 통계
        metrics_dict['loss/all'] = metrics_t[METRICS_LOSS_NDX].mean()
        metrics_dict['loss/neg'] = metrics_t[METRICS_LOSS_NDX, negLabel_mask].mean()
        metrics_dict['loss/pos'] = metrics_t[METRICS_LOSS_NDX, posLabel_mask].mean()

        # 정확도(Accuracy) 관련 통계냄
        metrics_dict['correct/all'] = (pos_correct + neg_correct) / np.float32(metrics_t.shape[1]) * 100
        metrics_dict['correct/neg'] = neg_correct / np.float32(neg_count) * 100
        metrics_dict['correct/pos'] = pos_correct / np.float32(pos_count) * 100

        # 전체 요약 출력
        log.info(
            ("E{} {:8} {loss/all:.4f} loss, "
            + "{correct/all:-5.1f}% correct, "
            ).format(
                epoch_ndx,
                mode_str,
                **metrics_dict,
            )
        )
        # 음성 샘플에 대한 손실/정확도 로그 출력
        log.info(
            ("E{} {:8} {loss/neg:.4f} loss, "
            + "{correct/neg:-5.1f}% correct ({neg_correct:} of {neg_count:})"
            ).format(
                epoch_ndx,
                mode_str + '_neg',
                neg_correct=neg_correct,
                neg_count=neg_count,
                **metrics_dict,
            )
        )
        # 양성 샘플에 대한 손실/정확도 로그 출력
        log.info(
            ("E{} {:8} {loss/pos:.4f} loss, "
            + "{correct/pos:-5.1f}% correct ({pos_correct:} of {pos_count:})"
            ).format(
                epoch_ndx,
                mode_str + '_pos',
                pos_correct=pos_correct,
                pos_count=pos_count,
                **metrics_dict,
            )
        )

        writer = getattr(self, mode_str + '_writer')
        # 'trn_writer' 또는 'val_writer' 가져오기

        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, self.totalTrainingSamples_count)
            # 각 메트릭 값을 TensorBoard에 기록

        writer.add_pr_curve(
            'pr',
            metrics_t[METRICS_LABEL_NDX],  # 실제 정답 (0/1)
            metrics_t[METRICS_PRED_NDX],   # 예측 확률
            self.totalTrainingSamples_count,
        )
        # PR 곡선(Precision-Recall Curve) 기록

        bins = [x/50.0 for x in range(51)]
        # 0~1 구간을 0.02 간격으로 나눈 히스토그램 bin 설정

        # 신뢰도 높은 음성/양성 예측에 대해 히스토그램 시각화 준비
        negHist_mask = negLabel_mask & (metrics_t[METRICS_PRED_NDX] > 0.01)
        posHist_mask = posLabel_mask & (metrics_t[METRICS_PRED_NDX] < 0.99)

        if negHist_mask.any():
            writer.add_histogram(
                'is_neg',
                metrics_t[METRICS_PRED_NDX, negHist_mask],
                self.totalTrainingSamples_count,
                bins=bins,
            )
        if posHist_mask.any():
            writer.add_histogram(
                'is_pos',
                metrics_t[METRICS_PRED_NDX, posHist_mask],
                self.totalTrainingSamples_count,
                bins=bins,
            )
        # 양성/음성 클래스별 예측 확률 히스토그램을 TensorBoard에 기록