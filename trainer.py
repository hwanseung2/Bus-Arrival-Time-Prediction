import os
import time
import random

import numpy as np
import pandas as pd
from collections import defaultdict

from preprocessor import Preprocessor
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import nsml


class MyDatasetTraining(Dataset):
    def __init__(
        self, concat, seq_len, label_len, pred_len, mode, args, dur_normalizer
    ):
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.mode = mode
        self.args = args
        self.train_mean = dur_normalizer.train_mean
        self.train_std = dur_normalizer.train_std

        len_ary = np.array([])
        concat_grouped = concat.groupby(["data_index"])
        for name, group in concat_grouped:
            len_ary = np.append(len_ary, group.shape[0])
        range_ary = len_ary - self.seq_len - self.pred_len + 1

        self.access_length = sum(range_ary)
        self.cumsum_ary = np.cumsum(range_ary).tolist()
        self.cumsum_ary = list(map(int, self.cumsum_ary))
        self.hash = defaultdict(int)
        prev = 0
        for idx, item in enumerate(self.cumsum_ary):
            for i in range(prev, item):
                self.hash[i] = idx
            prev = item

        concat = concat[
            [
                "route_id",
                "station_id",
                "direction",
                "hour",
                "dow",
                "next_station_distance",
                "prev_duration",
                "next_duration",
            ]
        ]
        data = concat.values
        self.data_x = data[:, 6:7]
        self.data_y = data[:, -1:]
        self.data_mark = data[:, :6]
        self.data_mark[:, 4] = self.data_mark[:, 4] / 7 - 0.5

    def __getitem__(self, index):
        s_begin = index + self.hash[index] * (self.seq_len + self.pred_len - 1)
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        if (
            self.mode == "train"
            and self.args.using_aug == True
            and random.randint(0, 9) < 3
        ):
            rand_s_begin = random.randint(0, self.seq_len - 7)
            rand_s_section = random.randint(3, 7)
            rand_r_begin = random.randint(0, self.pred_len - 7)
            rand_r_section = random.randint(3, 7)
            rand_plus_delta = 1 + random.randint(20, 50) / 100
            rand_minus_delta = random.randint(66, 90) / 100

            temp_x = self.data_x[s_begin:s_end, :].copy()
            temp_y = self.data_y[r_begin + self.label_len : r_end].copy()
            # 앞이 plus, 뒤가 minus일 경우
            if random.randint(0, 1) == 0:
                # temp_x의 길이는 seq_len만큼
                temp_x[rand_s_begin : rand_s_begin + rand_s_section, -1] = (
                    temp_x[rand_s_begin : rand_s_begin + rand_s_section, -1]
                    * rand_plus_delta
                )
                temp_y[rand_r_begin : rand_r_begin + rand_r_section, -1] = (
                    temp_y[rand_r_begin : rand_r_begin + rand_r_section, -1]
                    * rand_minus_delta
                )
            else:
                temp_x[rand_s_begin : rand_s_begin + rand_s_section, -1] = (
                    temp_x[rand_s_begin : rand_s_begin + rand_s_section, -1]
                    * rand_minus_delta
                )
                temp_y[rand_r_begin : rand_r_begin + rand_r_section, -1] = (
                    temp_y[rand_r_begin : rand_r_begin + rand_r_section, -1]
                    * rand_plus_delta
                )

            temp_x[:, -1] = (temp_x[:, -1] - self.train_mean) / self.train_std
            temp_y[:, -1] = (temp_y[:, -1] - self.train_mean) / self.train_std

            seq_x = temp_x
            tmp_y1 = temp_x[-self.label_len :, -1:]
            tmp_y2 = temp_y
            seq_y = np.concatenate([tmp_y1, tmp_y2], axis=0)

        else:
            temp_x = self.data_x[s_begin:s_end, :].copy()
            temp_y = self.data_y[r_begin + self.label_len : r_end].copy()
            temp_x[:, -1] = (temp_x[:, -1] - self.train_mean) / self.train_std
            temp_y[:, -1] = (temp_y[:, -1] - self.train_mean) / self.train_std

            seq_x = temp_x
            tmp_y1 = seq_x[-self.label_len :, -1:]
            seq_y = np.concatenate([tmp_y1, temp_y], axis=0)

        seq_x_mark = self.data_mark[s_begin:s_end, :]
        seq_y_mark = self.data_mark[r_begin:r_end, :]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return int(self.access_length.item())


class Trainer:
    def __init__(self, args, model, optimizer, criterion):
        self.preprocessor = Preprocessor(args)
        self.model = model.cuda()
        self.optimizer = optimizer
        self.criterion = criterion
        self.args = args
        self.train_mean = None
        self.train_std = None

    def _process_one_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().cuda()
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float().cuda()
        batch_y_mark = batch_y_mark.float().cuda()

        if self.args.padding == 0:
            dec_inp = torch.zeros(
                [batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]
            ).float()
        elif self.args.padding == 1:
            dec_inp = torch.ones(
                [batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]
            ).float()
        dec_inp = (
            torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
            .float()
            .cuda()
        )

        if self.args.output_attention:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        f_dim = -1
        batch_y = batch_y[:, -self.args.pred_len :, :].cuda()

        return outputs, batch_y

    def validation(self, valid_loader):
        self.model.eval()
        total_loss = []

        preds = []
        trues = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                valid_loader
            ):
                pred, true = self._process_one_batch(
                    batch_x, batch_y, batch_x_mark, batch_y_mark
                )
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        print("validation shape:", preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print("validation shape:", preds.shape, trues.shape)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print("mse:{}, mae:{}".format(mse, mae))

        self.model.train()
        return mae.item(), mse.item(), rmse.item()

    def training(self):
        (
            trainset,
            validset,
            dur_normalizer,
        ) = self.preprocessor.preprocess_train_dataset()
        print("[INFO - TRAINER]: Finished preprocessing training data...")

        trainset = MyDatasetTraining(
            trainset,
            self.args.seq_len,
            self.args.label_len,
            self.args.pred_len,
            "train",
            self.args,
            dur_normalizer,
        )
        validset = MyDatasetTraining(
            validset,
            self.args.seq_len,
            self.args.label_len,
            self.args.pred_len,
            "valid",
            self.args,
            dur_normalizer,
        )
        self.train_mean = dur_normalizer.train_mean
        self.train_std = dur_normalizer.train_std
        print(
            self.train_mean,
            self.train_std,
            dur_normalizer.train_mean,
            dur_normalizer.train_std,
        )

        train_loader = DataLoader(
            trainset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            drop_last=True,
        )

        valid_loader = DataLoader(
            validset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            drop_last=True,
        )

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        """
        initial validation
        """
        print()
        print("[INFO - TRAINING] start")
        t = 0
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                train_loader
            ):
                iter_count += 1
                self.optimizer.zero_grad()
                pred, true = self._process_one_batch(
                    batch_x, batch_y, batch_x_mark, batch_y_mark
                )
                loss = self.criterion(pred, true)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0} / {1}, epoch: {2} | loss: {3:.7f}".format(
                            i + 1, train_steps + 1, epoch + 1, loss.item()
                        )
                    )
                    print("\n\n")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    print()
                    print(
                        "[BATCH 0 SAMPLING] [PREDICTION]\n",
                        pred[0, :, 0],
                        "\n",
                        "[LABEL]\n",
                        true[0, :, 0],
                        "\n\n",
                    )
                    print(
                        "[BATCH 5 SAMPLING] [PREDICTION]\n",
                        pred[5, :, 0],
                        "\n",
                        "[LABEL]\n",
                        true[5, :, 0],
                        "\n\n",
                    )
                    print(
                        "[BATCH 9 SAMPLING] [PREDICTION]\n",
                        pred[9, :, 0],
                        "\n",
                        "[LABEL]\n",
                        true[9, :, 0],
                        "\n\n",
                    )
                    iter_count = 0
                    time_now = time.time()

                t += 1
                loss.backward()
                if (
                    self.args.using_lradj
                    and (self.args.lradj == "type3" or self.args.lradj == "type4")
                    and t <= 10002
                ):
                    adjust_learning_rate(self.optimizer, epoch, t, self.args)
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss).item()
            mae, mse, rmse = self.validation(valid_loader=valid_loader)
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali RMSE: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, rmse
                )
            )
            nsml.report(
                summary=True,
                scope=locals(),
                train_loss=train_loss,
                valid_rmse=rmse,
                valid_mae=mae,
                valid_mse=mse,
                step=epoch,
            )

            early_stopping(mse, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            nsml.save(str(epoch + 1))
            if (
                self.args.using_lradj
                and (self.args.lradj == "type3" or self.args.lradj == "type4")
                and t > 10002
            ):
                adjust_learning_rate(self.optimizer, epoch, t, self.args)

    def testing(self, test_data, k, n):
        self.model.eval()
        seq_len = k - 1
        label_len = k - 1
        pred_len = n - k + 1
        self.model.pred_len = pred_len

        test_data = test_data[
            [
                "route_id",
                "station_id",
                "direction",
                "hour",
                "dow",
                "next_station_distance",
                "prev_duration",
            ]
        ]

        data = test_data.values

        data[:, 4] = data[:, 4] / 7 - 0.5
        seq_x = torch.tensor(data[np.newaxis, :seq_len, 6:])
        seq_y = torch.tensor(data[np.newaxis, :label_len, 6:])
        seq_x_mark = torch.tensor(data[np.newaxis, :seq_len, :6])
        seq_y_mark = torch.tensor(data[np.newaxis, :, :6])
        seq_x_mark = seq_x_mark.float().cuda()
        seq_y_mark = seq_y_mark.float().cuda()

        seq_x = seq_x.float().cuda()
        seq_y = seq_y.float()

        dec_inp = torch.zeros([1, pred_len, seq_y.shape[-1]]).float()
        dec_inp = torch.cat([seq_y[:, :label_len, -1:], dec_inp], dim=1).float().cuda()

        output = self.model(seq_x, seq_x_mark, dec_inp, seq_y_mark)
        output = output[0, :, 0].detach().cpu().numpy()
        output = (output * self.train_std) + self.train_mean
        output = output.tolist()

        return output
