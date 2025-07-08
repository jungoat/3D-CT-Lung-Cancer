import argparse
import datetime
import os
import sys

import numpy as np

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from util.util import enumerateWithEstimate
from .dsets import LunaDataset
from util.logconf import logging
from .model import LunaModel

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

METRICS_LABEL_NDX=0
METRICS_PRED_NDX=1
METRICS_LOSS_NDX=2
METRICS_SIZE = 3

class LunaTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size', default=128, type=int)
        parser.add_argument('--num-workers', default=8, type=int)
        parser.add_argument('--epochs', default=1, type=int)
        parser.add_argument('--balanced', action='store_true', default=False)
        parser.add_argument('--augmented', action='store_true', default=False)
        parser.add_argument('--augment-flip', action='store_true', default=False)
        parser.add_argument('--augment-offset', action='store_true', default=False)
        parser.add_argument('--augment-scale', action='store_true', default=False)
        parser.add_argument('--augment-rotate', action='store_true', default=False)
        parser.add_argument('--augment-noise', action='store_true', default=False)
        parser.add_argument('--tb-prefix', default='p2ch12')
        parser.add_argument('comment', nargs='?', default='dlwpt')

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.augmentation_dict = {}
        if self.cli_args.augmented or self.cli_args.augment_flip:
            self.augmentation_dict['flip'] = True
        if self.cli_args.augmented or self.cli_args.augment_offset:
            self.augmentation_dict['offset'] = 0.1
        if self.cli_args.augmented or self.cli_args.augment_scale:
            self.augmentation_dict['scale'] = 0.2
        if self.cli_args.augmented or self.cli_args.augment_rotate:
            self.augmentation_dict['rotate'] = True
        if self.cli_args.augmented or self.cli_args.augment_noise:
            self.augmentation_dict['noise'] = 25.0

        self.use_cuda = torch.cuda.is_available()
        if not self.use_cuda:
            raise RuntimeError("CUDA is not available or not properly configured.")
        self.device = torch.device("cuda")

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()

    def initModel(self):
        model = LunaModel().to(self.device)
        return model

    def initOptimizer(self):
        return SGD(self.model.parameters(), lr=0.001, momentum=0.99)

    def initTrainDl(self):
        train_ds = LunaDataset(val_stride=10, isValSet_bool=False, ratio_int=int(self.cli_args.balanced), augmentation_dict=self.augmentation_dict)
        return DataLoader(train_ds, batch_size=self.cli_args.batch_size, num_workers=self.cli_args.num_workers, pin_memory=True)

    def initValDl(self):
        val_ds = LunaDataset(val_stride=10, isValSet_bool=True)
        return DataLoader(val_ds, batch_size=self.cli_args.batch_size, num_workers=self.cli_args.num_workers, pin_memory=True)

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join('runs', self.cli_args.tb_prefix, self.time_str)
            self.trn_writer = SummaryWriter(log_dir=log_dir + '-trn_cls-' + self.cli_args.comment)
            self.val_writer = SummaryWriter(log_dir=log_dir + '-val_cls-' + self.cli_args.comment)

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        train_dl = self.initTrainDl()
        val_dl = self.initValDl()

        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            log.info("Epoch {} of {}, {}/{} batches of size {}".format(epoch_ndx, self.cli_args.epochs, len(train_dl), len(val_dl), self.cli_args.batch_size))
            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)

            valMetrics_t = self.doValidation(epoch_ndx, val_dl)
            self.logMetrics(epoch_ndx, 'val', valMetrics_t)

        if self.trn_writer:
            self.trn_writer.close()
            self.val_writer.close()

    def doTraining(self, epoch_ndx, train_dl):
        self.model.train()
        train_dl.dataset.shuffleSamples()
        trnMetrics_g = torch.zeros(METRICS_SIZE, len(train_dl.dataset), device=self.device)

        batch_iter = enumerateWithEstimate(train_dl, f"E{epoch_ndx} Training")
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()
            loss_var = self.computeBatchLoss(batch_ndx, batch_tup, train_dl.batch_size, trnMetrics_g)
            loss_var.backward()
            self.optimizer.step()

        self.totalTrainingSamples_count += len(train_dl.dataset)
        return trnMetrics_g.to('cpu')

    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()
            valMetrics_g = torch.zeros(METRICS_SIZE, len(val_dl.dataset), device=self.device)
            batch_iter = enumerateWithEstimate(val_dl, f"E{epoch_ndx} Validation")
            for batch_ndx, batch_tup in batch_iter:
                self.computeBatchLoss(batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g)
        return valMetrics_g.to('cpu')

    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        input_t, label_t, _series_list, _center_list = batch_tup
        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        logits_g, probability_g = self.model(input_g)

        loss_func = nn.CrossEntropyLoss(reduction='none')
        loss_g = loss_func(logits_g, label_g[:,1])

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_g.size(0)

        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = label_g[:,1]
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = probability_g[:,1]
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_g

        return loss_g.mean()

    def logMetrics(self, epoch_ndx, mode_str, metrics_t, classificationThreshold=0.5):
        self.initTensorboardWriters()
        log.info("E{} {}".format(epoch_ndx, type(self).__name__))

        negLabel_mask = metrics_t[METRICS_LABEL_NDX] <= classificationThreshold
        negPred_mask = metrics_t[METRICS_PRED_NDX] <= classificationThreshold

        posLabel_mask = ~negLabel_mask
        posPred_mask = ~negPred_mask

        neg_count = int(negLabel_mask.sum())
        pos_count = int(posLabel_mask.sum())

        trueNeg_count = neg_correct = int((negLabel_mask & negPred_mask).sum())
        truePos_count = pos_correct = int((posLabel_mask & posPred_mask).sum())

        falsePos_count = neg_count - neg_correct
        falseNeg_count = pos_count - pos_correct

        metrics_dict = {
            'loss/all': metrics_t[METRICS_LOSS_NDX].mean(),
            'loss/neg': metrics_t[METRICS_LOSS_NDX, negLabel_mask].mean(),
            'loss/pos': metrics_t[METRICS_LOSS_NDX, posLabel_mask].mean(),
            'correct/all': (pos_correct + neg_correct) / metrics_t.shape[1] * 100,
            'correct/neg': (neg_correct) / neg_count * 100,
            'correct/pos': (pos_correct) / pos_count * 100,
        }

        precision = metrics_dict['pr/precision'] = truePos_count / np.float32(truePos_count + falsePos_count)
        recall = metrics_dict['pr/recall'] = truePos_count / np.float32(truePos_count + falseNeg_count)
        metrics_dict['pr/f1_score'] = 2 * (precision * recall) / (precision + recall)

        log.info(("E{} {:8} {loss/all:.4f} loss, {correct/all:-5.1f}% correct, {pr/precision:.4f} precision, {pr/recall:.4f} recall, {pr/f1_score:.4f} f1 score").format(epoch_ndx, mode_str, **metrics_dict))
        log.info(("E{} {:8} {loss/neg:.4f} loss, {correct/neg:-5.1f}% correct ({neg_correct:} of {neg_count:})").format(epoch_ndx, mode_str + '_neg', neg_correct=neg_correct, neg_count=neg_count, **metrics_dict))
        log.info(("E{} {:8} {loss/pos:.4f} loss, {correct/pos:-5.1f}% correct ({pos_correct:} of {pos_count:})").format(epoch_ndx, mode_str + '_pos', pos_correct=pos_correct, pos_count=pos_count, **metrics_dict))

        writer = getattr(self, mode_str + '_writer')
        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, self.totalTrainingSamples_count)

        writer.add_pr_curve('pr', metrics_t[METRICS_LABEL_NDX], metrics_t[METRICS_PRED_NDX], self.totalTrainingSamples_count)

        bins = [x/50.0 for x in range(51)]
        negHist_mask = negLabel_mask & (metrics_t[METRICS_PRED_NDX] > 0.01)
        posHist_mask = posLabel_mask & (metrics_t[METRICS_PRED_NDX] < 0.99)

        if negHist_mask.any():
            writer.add_histogram('is_neg', metrics_t[METRICS_PRED_NDX, negHist_mask], self.totalTrainingSamples_count, bins=bins)
        if posHist_mask.any():
            writer.add_histogram('is_pos', metrics_t[METRICS_PRED_NDX, posHist_mask], self.totalTrainingSamples_count, bins=bins)

if __name__ == '__main__':
    LunaTrainingApp().main()
