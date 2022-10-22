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
    def __init__(self, concat, seq_len, label_len, pred_len, mode, args, dur_normalizer):
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.mode = mode
        self.args = args
        self.train_mean = dur_normalizer.train_mean
        self.train_std = dur_normalizer.train_std

        len_ary = np.array([])
        concat_grouped = concat.groupby(['data_index'])
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


        concat = concat[['route_id', 'station_id', 'direction', 'hour', 'dow', 'next_station_distance', 'prev_duration', 'next_duration']]
        data = concat.values 
        self.data_x = data[:, 6:7] 
        self.data_y = data[:, -1:]
        self.data_mark = data[:, :6]
        self.data_mark[:,4] = self.data_mark[:,4] / 7 - 0.5 

        

        
    def __getitem__(self, index):
        s_begin = index + self.hash[index] * (self.seq_len + self.pred_len - 1)
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        if self.mode == 'train' and self.args.using_aug == True and random.randint(0,9) < 3:
            rand_s_begin = random.randint(0, self.seq_len-7)
            rand_s_section = random.randint(3, 7)
            rand_r_begin = random.randint(0, self.pred_len - 7)
            rand_r_section = random.randint(3, 7)
            rand_plus_delta = 1 + random.randint(20, 50) / 100
            rand_minus_delta = random.randint(66, 90) / 100

            temp_x = self.data_x[s_begin:s_end, :].copy() 
            temp_y = self.data_y[r_begin+self.label_len:r_end].copy()
            #앞이 plus, 뒤가 minus일 경우
            if random.randint(0,1) == 0:
                # temp_x의 길이는 seq_len만큼
                temp_x[rand_s_begin:rand_s_begin + rand_s_section, -1] = temp_x[rand_s_begin:rand_s_begin + rand_s_section, -1] * rand_plus_delta
                temp_y[rand_r_begin: rand_r_begin + rand_r_section, -1] = temp_y[rand_r_begin: rand_r_begin + rand_r_section, -1] * rand_minus_delta
            else:
                temp_x[rand_s_begin:rand_s_begin + rand_s_section, -1] = temp_x[rand_s_begin:rand_s_begin + rand_s_section, -1] * rand_minus_delta
                temp_y[rand_r_begin: rand_r_begin + rand_r_section, -1] = temp_y[rand_r_begin: rand_r_begin + rand_r_section, -1] * rand_plus_delta
            
            temp_x[:, -1] = (temp_x[:, -1] - self.train_mean) / self.train_std
            temp_y[:, -1] = (temp_y[:, -1] - self.train_mean) / self.train_std

            seq_x = temp_x
            tmp_y1 = temp_x[-self.label_len:, -1:]
            tmp_y2 = temp_y
            seq_y = np.concatenate([tmp_y1, tmp_y2], axis = 0)
            

        else:
            temp_x = self.data_x[s_begin:s_end, :].copy()
            temp_y = self.data_y[r_begin+self.label_len:r_end].copy()
            temp_x[:, -1] = (temp_x[:, -1] - self.train_mean) / self.train_std
            temp_y[:, -1] = (temp_y[:, -1] - self.train_mean) / self.train_std

            seq_x = temp_x
            tmp_y1 = seq_x[-self.label_len:, -1:]
            seq_y = np.concatenate([tmp_y1, temp_y], axis = 0)



        seq_x_mark = self.data_mark[s_begin:s_end, :] 
        seq_y_mark = self.data_mark[r_begin:r_end, :]


        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return int(self.access_length.item())


class Trainer():
    def __init__(self, args, model, optimizer, criterion):
        self.preprocessor = Preprocessor(args)
        self.model = model.cuda()
        self.optimizer = optimizer
        self.criterion = criterion
        self.args = args
        self.train_mean = None
        self.train_std = None