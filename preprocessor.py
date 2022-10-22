import os
import random
import time
from tqdm import tqdm
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import torch

from nsml import DATASET_PATH

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)  

LABEL_COLUMNS = ["data_index", "route_id", "plate_no", "operation_id", "station_seq", "next_duration"]

def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value) 
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def hash_mapping(check_seq, mode, duration, route_id, station_seq, hash):
    if mode == 'mean':
        if check_seq == False:
            if hash.get((route_id, station_seq)) == None:
                # 여기에 1067 - 116 / 1358 - 124가 들어온다.
                if (route_id == 1358 and station_seq == 124) or (route_id == 1067 and station_seq == 116):
                    return 38
                elif route_id == 1067 and station_seq > 62:
                    tmp = station_seq - 1 - 62
                    target = 62 - tmp - 1
                    return hash[(route_id, target)]
                elif route_id == 1358 and station_seq > 66:
                    tmp = station_seq - 1 - 66
                    target = 66 - tmp - 1
                    return hash[(route_id, target)]
                else:

                    raise Exception(f"duration: {duration}, route_id: {route_id}, station_seq: {station_seq}")
            else:
                return hash[(route_id, station_seq)]
        else:
            return duration
    elif mode == 'median':
        if hash.get((route_id, station_seq)) == None:
            raise Exception(f"\n\n median hash key None!!! - route_id: {route_id}, station_seq: {station_seq}\n")
        else:
            return hash[(route_id, station_seq)]

class StandardScaler():
    def __init__(self):
        self.train_mean = None
        self.train_std = None
    
    def build(self, train_data):
        self.train_mean = train_data.mean()
        self.train_std = train_data.std()

    def normalize(self, df):
        return (df - self.train_mean) / self.train_std

class Preprocessor():
    def __init__(self, args):
        """
        Data & Data Info
        """
        self.args = args
        print(f"random_sampling: {self.args.random_sampling}")
        self.train_data_path = os.path.join(DATASET_PATH, "train", "train_data", "data")
        self.train_label_path = os.path.join(DATASET_PATH, "train", "train_label")
        self.direction_df = None
        self.route_df = None
        """
        Label Encoder & Standard Scaler & dictionary for imputation
        """
        self.route_encoder = LabelEncoder()
        self.station_encoder = LabelEncoder()

        self.dist_normalizer = StandardScaler()
        self.dur_normalizer = StandardScaler()
        
        # self.direction_hash = defaultdict(list)
        self.station_seq_hash = defaultdict()
        self.distance_hash = defaultdict()
        self.prev_mean_hash = defaultdict(float)
        self.prev_median_hash = defaultdict(float)
        self.next_mean_hash = defaultdict(float)
        self.next_median_hash = defaultdict(float)
        

    def _load_train_dataset(self, train_data_path = None):
        print("starting to load train data: ")
        # random_seed(self.config.seed, False)
        print(self.train_data_path)
        
        train_data = pd.read_parquet(self.train_data_path) \
            .sort_values(by = ["data_index", "station_seq"], ignore_index = True)
            
        print("train data shape: ")
        print(train_data.shape)
        
        train_label = pd.read_csv(self.train_label_path, header = 0, low_memory = False)\
            .sort_values(by = ["data_index", "station_seq"], ignore_index = True)
            
        print("train label shape: ")
        print(train_label.shape)

        print("starting to load info data: ")
        train_station_path = os.path.join(DATASET_PATH, "train", "train_data/info", "shapes.csv")
        train_route_path = os.path.join(DATASET_PATH, "train", "train_data/info", "routes.csv")
        shape_df = pd.read_csv(train_station_path, low_memory = False).astype('int')
        route_df = pd.read_csv(train_route_path, low_memory = False)
        self.route_df = route_df

        print("starting to save n_stations for imputation")
        route_temp_df = route_df[['route_id', 'n_stations']]
        for index, row in route_temp_df.iterrows():
            route_id = row['route_id']
            n_stations = row['n_stations']
            if route_id == 18525:
                n_stations = 99
            self.station_seq_hash[route_id] = n_stations
        del route_temp_df

        print("starting to save distance for imputation")
        dist_temp_df = shape_df[['route_id', 'station_seq', 'station_id', 'distance']]
        for index, row in dist_temp_df.iterrows():
            route_id = row['route_id']
            station_seq = row['station_seq']
            distance = row['distance']
            station_id = row['station_id']
            self.distance_hash[(route_id, station_seq)] = [station_id, distance]
        del dist_temp_df

        print("starting to make route encoder")
        # self.direction_df = route_df[['route_id', 'turning_point_sequence']]
        list_route = list(set(shape_df['route_id'].unique())) + [0]
        self.route_encoder.fit(list_route)
        list_station = list(set(shape_df['station_id'].unique())) + [0]
        self.station_encoder.fit(list_station)

        
        return train_data, train_label


    def preprocess_train_dataset(self):
        print("load train data to preprocess...")
        train_data, train_label = self._load_train_dataset()
        concat = pd.concat([train_data, train_label['next_duration']], axis = 1)
        
        print("preprocess train set")
        print("make prev_duration column & check_seq(True, False) column", '\n\n\n')
        concat["prev_ts"] = concat.groupby("data_index")['ts'].shift(1)
        concat["prev_seq"] = concat.groupby("data_index")['station_seq'].shift(1)
        concat["prev_ts"] = concat["prev_ts"].fillna(0)
        concat["prev_duration"] = np.where(concat["prev_ts"] == 0, 0, concat["ts"] - concat["prev_ts"])
        concat['check_seq'] = np.where(concat['station_seq'] - concat['prev_seq'] == 1, True, False)

        print("process missing value(imputation)")
        concat = self.process_missing_value(concat = concat, mode = 'train', k = None)
        print(f"[INFO - IMPUTATION]: isnull \n {concat.isnull().sum()}")
        print(f"[INFO - IMPUTATION]: concat shape : {concat.shape}", '\n\n')
        print(concat.describe())
        """
        Sampling
        """
        print("[SAMPLING] SAMPLING START")
        if 0 < self.args.random_sampling < 1:
            concat = self.sampling(concat)
        print("sampled data shape:")
        print(concat.shape)
        print("[SAMPLING] SAMPLING END")

        print(f"[INFO - DUPLICATION]: delete duplication(next_duration : 0)")
        concat = self.delete_duplication(concat)
        print(f"[INFO - DUPLICATION]: after delete duplication concat shape: {concat.shape}\n\n")
        
        print(f"[INFO - REPLACE]: replace out-lier-  x < 5sec")
        concat = self.replace_outlier_using_mean(concat)
        print(f"[INFO - REPLACE]: End")

        print("[INFO - DIRECTION FEATURE] start")
        self.direction_df = pd.merge(concat[['route_id']], self.route_df[['route_id', 'turning_point_sequence']], on = 'route_id')
        concat['direction'] = np.where(concat['station_seq'] <= self.direction_df['turning_point_sequence'], 0, 1)
        print("[INFO - DIRECTION FEATURE] end", '\n\n')

        print(concat.describe())
        print(concat.info())

        print("[INFO - LABEL ENCODING] route_id label encoding start")
        concat['route_id'] = self.route_encoder.transform(concat['route_id'])
        concat['station_id'] = self.station_encoder.transform(concat['station_id'])
        print("[INFO - LABEL ENCODING] route_id label encoding end", '\n\n')

        print("[INFO - STANDARD SCALER] distance & prev duration scale & prev distance" )
        self.dist_normalizer.build(concat['next_station_distance'])
        self.dur_normalizer.build(concat['next_duration'])
        concat['prev_station_distance'] = concat.groupby('data_index')['next_station_distance'].shift(1)
        concat['prev_station_distance'] = concat['prev_station_distance'].fillna(0)


        concat['next_station_distance'] = self.dist_normalizer.normalize(concat['next_station_distance'])
        concat['prev_station_distance'] = self.dist_normalizer.normalize(concat['prev_station_distance'])
        # concat['prev_duration'] = self.dur_normalizer.normalize(concat['prev_duration'])
        print("[INFO - STANDARD SCALER] PREV NEXT DISTANCE와 PREV DURATION만 normalize")
        print("[INFO - STANDARD SCALER] NEXT DURATION 적용 X", '\n\n')

        print(concat.head(200), '\n\n')

        print("[INFO - DATA SPLIT]")
        stime = time.time()
        trainset, validset = self.split_data(concat)
        print(f"trainset shape : {trainset.shape}, validset shape : {validset.shape}")
        print(f"[INFO - DATA SPLIT]: split time {time.time() - stime}", '\n\n')
        print(trainset.head(150), '\n\n')
        print(validset.head(150))

        return trainset, validset, self.dur_normalizer