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

pd.set_option("display.max_columns", 500)
pd.set_option("display.max_rows", 500)

LABEL_COLUMNS = [
    "data_index",
    "route_id",
    "plate_no",
    "operation_id",
    "station_seq",
    "next_duration",
]


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
    if mode == "mean":
        if check_seq == False:
            if hash.get((route_id, station_seq)) == None:
                # 여기에 1067 - 116 / 1358 - 124가 들어온다.
                if (route_id == 1358 and station_seq == 124) or (
                    route_id == 1067 and station_seq == 116
                ):
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

                    raise Exception(
                        f"duration: {duration}, route_id: {route_id}, station_seq: {station_seq}"
                    )
            else:
                return hash[(route_id, station_seq)]
        else:
            return duration
    elif mode == "median":
        if hash.get((route_id, station_seq)) == None:
            raise Exception(
                f"\n\n median hash key None!!! - route_id: {route_id}, station_seq: {station_seq}\n"
            )
        else:
            return hash[(route_id, station_seq)]


class StandardScaler:
    def __init__(self):
        self.train_mean = None
        self.train_std = None

    def build(self, train_data):
        self.train_mean = train_data.mean()
        self.train_std = train_data.std()

    def normalize(self, df):
        return (df - self.train_mean) / self.train_std


class Preprocessor:
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

    def _load_train_dataset(self, train_data_path=None):
        print("starting to load train data: ")
        # random_seed(self.config.seed, False)
        print(self.train_data_path)

        train_data = pd.read_parquet(self.train_data_path).sort_values(
            by=["data_index", "station_seq"], ignore_index=True
        )

        print("train data shape: ")
        print(train_data.shape)

        train_label = pd.read_csv(
            self.train_label_path, header=0, low_memory=False
        ).sort_values(by=["data_index", "station_seq"], ignore_index=True)

        print("train label shape: ")
        print(train_label.shape)

        print("starting to load info data: ")
        train_station_path = os.path.join(
            DATASET_PATH, "train", "train_data/info", "shapes.csv"
        )
        train_route_path = os.path.join(
            DATASET_PATH, "train", "train_data/info", "routes.csv"
        )
        shape_df = pd.read_csv(train_station_path, low_memory=False).astype("int")
        route_df = pd.read_csv(train_route_path, low_memory=False)
        self.route_df = route_df

        print("starting to save n_stations for imputation")
        route_temp_df = route_df[["route_id", "n_stations"]]
        for index, row in route_temp_df.iterrows():
            route_id = row["route_id"]
            n_stations = row["n_stations"]
            if route_id == 18525:
                n_stations = 99
            self.station_seq_hash[route_id] = n_stations
        del route_temp_df

        print("starting to save distance for imputation")
        dist_temp_df = shape_df[["route_id", "station_seq", "station_id", "distance"]]
        for index, row in dist_temp_df.iterrows():
            route_id = row["route_id"]
            station_seq = row["station_seq"]
            distance = row["distance"]
            station_id = row["station_id"]
            self.distance_hash[(route_id, station_seq)] = [station_id, distance]
        del dist_temp_df

        print("starting to make route encoder")
        # self.direction_df = route_df[['route_id', 'turning_point_sequence']]
        list_route = list(set(shape_df["route_id"].unique())) + [0]
        self.route_encoder.fit(list_route)
        list_station = list(set(shape_df["station_id"].unique())) + [0]
        self.station_encoder.fit(list_station)

        return train_data, train_label

    def process_missing_value(self, concat, mode, k):
        print("[INFO - IMPUTATION]: imputation setting")
        outlier = concat[concat["next_duration"] > 1500][["route_id", "station_seq"]]
        outlier_list = []
        for index, row in outlier.iterrows():
            route_id = row["route_id"]
            station_seq = row["station_seq"]
            outlier_list.append((route_id, station_seq))
        outlier_set = set(outlier_list)

        """
        check seq이 True인 애들로만 할 필요가 없음. next_duration은 next station을 명시하고 있으니까.
        """
        prev_median_grouped = concat.groupby(["route_id", "station_seq"])
        for name, group in prev_median_grouped:
            prev_median = np.median(group["prev_duration"])
            self.prev_median_hash[name] = prev_median
            next_median = np.median(group["next_duration"])
            self.next_median_hash[name] = next_median

        for key_ in self.prev_median_hash.keys():
            if self.prev_median_hash[key_] > 1500:
                print(key_, self.prev_median_hash[key_])

        print("#" * 50)
        print("#" * 50)
        print("#" * 50)
        for key_ in self.next_median_hash.keys():
            if self.next_median_hash[key_] > 1500:
                print(key_, self.next_median_hash[key_])

        idx = concat[concat["prev_duration"] > 1500].index
        concat.loc[idx, ["prev_duration"]] = concat.loc[
            idx, ["prev_duration", "route_id", "station_seq"]
        ].apply(
            lambda x: hash_mapping(
                False, "median", x[0], x[1], x[2], self.prev_median_hash
            ),
            axis=1,
        )

        idx = concat[concat["next_duration"] > 1500].index
        concat.loc[idx, ["next_duration"]] = concat.loc[
            idx, ["next_duration", "route_id", "station_seq"]
        ].apply(
            lambda x: hash_mapping(
                False, "median", x[0], x[1], x[2], self.next_median_hash
            ),
            axis=1,
        )

        print("median 처리 후, next_duration에 1500 이상 값 있는지 체크")
        print(concat[concat["next_duration"] > 1500].head(200))

        """
        start
        """
        concat_complete = concat[concat["check_seq"] == True]

        prev_mean_grouped = concat_complete.groupby(["route_id", "station_seq"])

        for name, group in prev_mean_grouped:
            mean_ = group["prev_duration"].mean()
            self.prev_mean_hash[name] = mean_

        iteration = list(self.prev_mean_hash.keys()).copy()
        for key_ in iteration:
            self.prev_mean_hash[(key_[0], 1)] = 0

        del concat_complete
        del prev_mean_grouped
        del prev_median_grouped

        print("[INFO - IMPUTATION]: calculating prev mean dataframe")
        print(
            "[INFO - IMPUTATION]: replace previous duration, check_seq == False case only"
        )
        idx = concat[concat["check_seq"] == False].index
        concat.loc[idx, ["prev_duration"]] = concat.loc[
            idx, ["check_seq", "prev_duration", "route_id", "station_seq"]
        ].apply(
            lambda x: hash_mapping(x[0], "mean", x[1], x[2], x[3], self.prev_mean_hash),
            axis=1,
        )
        concat["prev_seq"] = concat["prev_seq"].fillna(0)

        print("[INFO - IMPUTATION]: calculating next mean dataframe")
        next_mean_grouped = concat.groupby(["route_id", "station_seq"])
        for name, group in next_mean_grouped:
            mean_ = group["next_duration"].mean()
            self.next_mean_hash[name] = mean_

        del next_mean_grouped

        print("[INFO - IMPUTATION]: select column")
        concat = concat[
            [
                "route_id",
                "dow",
                "station_id",
                "station_seq",
                "next_station_distance",
                "hour",
                "data_index",
                "prev_duration",
                "next_duration",
            ]
        ]
        missing_grouped = concat.groupby(["data_index"])

        imputation_cnt = 0
        missing_cnt = {"1": 0, "2": 0, "3": 0}
        data_entry_list = []

        for name, group in tqdm(missing_grouped):
            route_id = group["route_id"].iloc[0]
            dow = int(group["dow"].iloc[0])
            series_station_seq = set(list(group["station_seq"]))
            if mode == "train":
                missing_seq_list = [
                    x
                    for x in range(1, self.station_seq_hash[route_id] - 1)
                    if x not in series_station_seq
                ]
            for seq_case in missing_seq_list:
                if (route_id, seq_case) not in outlier_set:
                    if (
                        self.prev_mean_hash.get((route_id, seq_case)) != None
                        and self.next_mean_hash.get((route_id, seq_case)) != None
                    ):
                        # 둘 다 hash에 있는 경우
                        case = {
                            "route_id": route_id,
                            "dow": dow,
                            "station_id": self.distance_hash[(route_id, seq_case)][0],
                            "station_seq": seq_case,
                            "hour": np.nan,
                            "next_station_distance": self.distance_hash[
                                (route_id, seq_case)
                            ][1],
                            "data_index": name,
                            "prev_duration": self.prev_mean_hash[(route_id, seq_case)],
                            "next_duration": self.next_mean_hash[(route_id, seq_case)],
                        }
                        data_entry_list.append(case)
                        imputation_cnt += 1

                    elif (
                        self.prev_mean_hash.get((route_id, seq_case)) != None
                        and self.next_mean_hash.get((route_id, seq_case)) == None
                    ):
                        # prev mean은 계산 돼 있고, next_mean이 없는 경우.
                        missing_cnt["1"] += 1
                        print("is test set imputation error case 1?")
                        raise ValueError

                    elif (
                        self.prev_mean_hash.get((route_id, seq_case)) == None
                        and self.next_mean_hash.get((route_id, seq_case)) != None
                    ):
                        if (route_id == 1358 and seq_case == 124) or (
                            route_id == 1067 and seq_case == 116
                        ):
                            case = {
                                "route_id": route_id,
                                "dow": dow,
                                "station_id": self.distance_hash[(route_id, seq_case)][
                                    0
                                ],
                                "station_seq": seq_case,
                                "hour": np.nan,
                                "next_station_distance": self.distance_hash[
                                    (route_id, seq_case)
                                ][1],
                                "data_index": name,
                                "prev_duration": 38,
                                "next_duration": self.next_mean_hash[
                                    (route_id, seq_case)
                                ],
                            }
                            data_entry_list.append(case)
                        else:
                            print("except 1358, 1067 route id imputation case 2")
                            raise ValueError
                        missing_cnt["2"] += 1
                    else:
                        # print("hash에 하나라도 없는 경우")
                        if (route_id == 1358 and seq_case == 123) or (
                            route_id == 1067 and seq_case == 115
                        ):
                            case = {
                                "route_id": route_id,
                                "dow": dow,
                                "station_id": self.distance_hash[(route_id, seq_case)][
                                    0
                                ],
                                "station_seq": seq_case,
                                "hour": np.nan,
                                "next_station_distance": self.distance_hash[
                                    (route_id, seq_case)
                                ][1],
                                "data_index": name,
                                "prev_duration": 120,
                                "next_duration": 38,
                            }
                            data_entry_list.append(case)
                        # else는 그냥 패스하면됨
                        missing_cnt["3"] += 1
                elif (route_id, seq_case) in outlier_set:
                    if (
                        self.prev_median_hash.get((route_id, seq_case)) != None
                        and self.next_median_hash.get((route_id, seq_case)) != None
                    ):
                        # 둘 다 hash에 있는 경우
                        case = {
                            "route_id": route_id,
                            "dow": dow,
                            "station_id": self.distance_hash[(route_id, seq_case)][0],
                            "station_seq": seq_case,
                            "hour": np.nan,
                            "next_station_distance": self.distance_hash[
                                (route_id, seq_case)
                            ][1],
                            "data_index": name,
                            "prev_duration": self.prev_median_hash[
                                (route_id, seq_case)
                            ],
                            "next_duration": self.next_median_hash[
                                (route_id, seq_case)
                            ],
                        }
                        data_entry_list.append(case)
                        imputation_cnt += 1

                    elif (
                        self.prev_median_hash.get((route_id, seq_case)) != None
                        and self.next_median_hash.get((route_id, seq_case)) == None
                    ):
                        # prev median은 계산 돼 있고, next_median이 없는 경우.
                        missing_cnt["1"] += 1
                        print("is test set imputation error case 1?")
                        raise ValueError

                    elif (
                        self.prev_median_hash.get((route_id, seq_case)) == None
                        and self.next_median_hash.get((route_id, seq_case)) != None
                    ):
                        if (route_id == 1358 and seq_case == 124) or (
                            route_id == 1067 and seq_case == 116
                        ):
                            case = {
                                "route_id": route_id,
                                "dow": dow,
                                "station_id": self.distance_hash[(route_id, seq_case)][
                                    0
                                ],
                                "station_seq": seq_case,
                                "hour": np.nan,
                                "next_station_distance": self.distance_hash[
                                    (route_id, seq_case)
                                ][1],
                                "data_index": name,
                                "prev_duration": 38,
                                "next_duration": self.next_median_hash[
                                    (route_id, seq_case)
                                ],
                            }
                            data_entry_list.append(case)
                        else:
                            print("except 1358, 1067 route id imputation case 2")
                            raise ValueError
                        missing_cnt["2"] += 1
                    else:
                        # print("hash에 하나라도 없는 경우")
                        if (route_id == 1358 and seq_case == 123) or (
                            route_id == 1067 and seq_case == 115
                        ):
                            case = {
                                "route_id": route_id,
                                "dow": dow,
                                "station_id": self.distance_hash[(route_id, seq_case)][
                                    0
                                ],
                                "station_seq": seq_case,
                                "hour": np.nan,
                                "next_station_distance": self.distance_hash[
                                    (route_id, seq_case)
                                ][1],
                                "data_index": name,
                                "prev_duration": 120,
                                "next_duration": 38,
                            }
                            data_entry_list.append(case)
                        # else는 그냥 패스하면됨
                        missing_cnt["3"] += 1
        print("[INFO - IMPUTATION]: create data entry list end & make dataframe")
        new = pd.DataFrame(data_entry_list)
        concat = pd.concat([concat, new]).sort_values(
            by=["data_index", "station_seq"], ignore_index=True
        )

        print("[INFO- IMPUTATION]: imputation end")
        print(
            f">>>>>>> imputation count : {imputation_cnt}, prev mean만 존재 : {missing_cnt['1']}, next mean만 존재 : {missing_cnt['2']}, 둘 다 mean이 없는 경우 : {missing_cnt['3']}, total : {missing_cnt['1'] + missing_cnt['2'] + missing_cnt['3']}"
        )

        """
        여기부터 짜야함!!!!
        여기짜고 테스트셋에도 똑같이 반영!!!!
        """

        concat_hour = (
            concat.groupby(["data_index"]).fillna(method="ffill").fillna(method="bfill")
        )
        concat["hour"] = concat_hour["hour"]
        del concat_hour

        return concat

    def delete_duplication(self, concat):
        missing_grouped = concat.groupby(["data_index"])
        idxes = np.array([])
        for name, group in tqdm(missing_grouped):
            s_s = np.array(group["station_seq"].values)
            unique = np.unique(s_s, return_counts=True)
            if np.any(unique[1] > 1):
                idx = group[group["next_duration"] == 0].index
                idxes = np.append(idxes, np.array(idx))
        concat = concat.drop(idxes, axis=0)
        concat = concat.reset_index(drop=True)
        return concat

    def replace_outlier_using_mean(self, concat):
        print("[INFO - REPLACE]", "\n")

        idx = concat[concat["prev_duration"] < 5].index
        concat.loc[idx, ["prev_duration"]] = concat.loc[
            idx, ["prev_duration", "route_id", "station_seq"]
        ].apply(
            lambda x: hash_mapping(
                False, "mean", x[0], x[1], x[2], self.prev_mean_hash
            ),
            axis=1,
        )
        idx = concat[concat["next_duration"] < 5].index
        concat.loc[idx, ["next_duration"]] = concat.loc[
            idx, ["next_duration", "route_id", "station_seq"]
        ].apply(
            lambda x: hash_mapping(
                False, "mean", x[0], x[1], x[2], self.next_mean_hash
            ),
            axis=1,
        )
        print("[INFO - REPLACE] End", "\n")
        return concat

    def split_data(self, concat):
        # train_idx, valid_idx = train_test_split(np.arange(1,383327 + 1), test_size = 0.2)
        train_idx, valid_idx = train_test_split(np.arange(1, 383327 + 1), test_size=0.2)
        train_idx, valid_idx = set(train_idx.tolist()), set(valid_idx.tolist())
        data_grouped = concat.groupby(["data_index"])
        train_list = []
        valid_list = []
        for idx, group in data_grouped:
            if idx in train_idx:
                train_list.append(group)
            else:
                valid_list.append(group)
        trainset = pd.concat(train_list)
        validset = pd.concat(valid_list)

        trainset = trainset.reset_index(drop=True)
        validset = validset.reset_index(drop=True)
        trainset = trainset.sort_values(
            by=["data_index", "station_seq"], ignore_index=True
        )
        validset = validset.sort_values(
            by=["data_index", "station_seq"], ignore_index=True
        )

        return trainset, validset

    def sampling(self, concat):
        full_len = 383328
        sample_len = int(full_len * self.args.random_sampling)
        sampling_idx = np.random.choice(full_len, sample_len)
        sampling_idx = set(sampling_idx.tolist())
        data_grouped = concat.groupby(["data_index"])
        sampling_list = []
        for idx, group in data_grouped:
            if idx in sampling_idx:
                sampling_list.append(group)

        sampled = pd.concat(sampling_list)

        return sampled

    def preprocess_train_dataset(self):
        print("load train data to preprocess...")
        train_data, train_label = self._load_train_dataset()
        concat = pd.concat([train_data, train_label["next_duration"]], axis=1)

        print("preprocess train set")
        print("make prev_duration column & check_seq(True, False) column", "\n\n\n")
        concat["prev_ts"] = concat.groupby("data_index")["ts"].shift(1)
        concat["prev_seq"] = concat.groupby("data_index")["station_seq"].shift(1)
        concat["prev_ts"] = concat["prev_ts"].fillna(0)
        concat["prev_duration"] = np.where(
            concat["prev_ts"] == 0, 0, concat["ts"] - concat["prev_ts"]
        )
        concat["check_seq"] = np.where(
            concat["station_seq"] - concat["prev_seq"] == 1, True, False
        )

        print("process missing value(imputation)")
        concat = self.process_missing_value(concat=concat, mode="train", k=None)
        print(f"[INFO - IMPUTATION]: isnull \n {concat.isnull().sum()}")
        print(f"[INFO - IMPUTATION]: concat shape : {concat.shape}", "\n\n")
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
        print(
            f"[INFO - DUPLICATION]: after delete duplication concat shape: {concat.shape}\n\n"
        )

        print(f"[INFO - REPLACE]: replace out-lier-  x < 5sec")
        concat = self.replace_outlier_using_mean(concat)
        print(f"[INFO - REPLACE]: End")

        print("[INFO - DIRECTION FEATURE] start")
        self.direction_df = pd.merge(
            concat[["route_id"]],
            self.route_df[["route_id", "turning_point_sequence"]],
            on="route_id",
        )
        concat["direction"] = np.where(
            concat["station_seq"] <= self.direction_df["turning_point_sequence"], 0, 1
        )
        print("[INFO - DIRECTION FEATURE] end", "\n\n")

        print(concat.describe())
        print(concat.info())

        print("[INFO - LABEL ENCODING] route_id label encoding start")
        concat["route_id"] = self.route_encoder.transform(concat["route_id"])
        concat["station_id"] = self.station_encoder.transform(concat["station_id"])
        print("[INFO - LABEL ENCODING] route_id label encoding end", "\n\n")

        print("[INFO - STANDARD SCALER] distance & prev duration scale & prev distance")
        self.dist_normalizer.build(concat["next_station_distance"])
        self.dur_normalizer.build(concat["next_duration"])
        concat["prev_station_distance"] = concat.groupby("data_index")[
            "next_station_distance"
        ].shift(1)
        concat["prev_station_distance"] = concat["prev_station_distance"].fillna(0)

        concat["next_station_distance"] = self.dist_normalizer.normalize(
            concat["next_station_distance"]
        )
        concat["prev_station_distance"] = self.dist_normalizer.normalize(
            concat["prev_station_distance"]
        )
        print("[INFO - STANDARD SCALER] PREV NEXT DISTANCE와 PREV DURATION만 normalize")
        print("[INFO - STANDARD SCALER] NEXT DURATION 적용 X", "\n\n")

        print(concat.head(200), "\n\n")

        print("[INFO - DATA SPLIT]")
        stime = time.time()
        trainset, validset = self.split_data(concat)
        print(f"trainset shape : {trainset.shape}, validset shape : {validset.shape}")
        print(f"[INFO - DATA SPLIT]: split time {time.time() - stime}", "\n\n")
        print(trainset.head(150), "\n\n")
        print(validset.head(150))

        return trainset, validset, self.dur_normalizer

    def process_missing_value_for_test(self, test_data, k, n):
        idx = test_data[test_data["check_seq"] == False].index
        test_data.loc[idx, ["prev_duration"]] = test_data.loc[
            idx, ["check_seq", "prev_duration", "route_id", "station_seq"]
        ].apply(
            lambda x: hash_mapping(x[0], "mean", x[1], x[2], x[3], self.prev_mean_hash),
            axis=1,
        )

        test_data = test_data[
            [
                "route_id",
                "dow",
                "hour",
                "plate_no",
                "operation_id",
                "station_id",
                "station_seq",
                "next_station_distance",
                "data_index",
                "prev_duration",
            ]
        ]
        data_entry_list = []

        route_id = test_data["route_id"].iloc[0]
        dow = int(test_data["dow"].iloc[0])
        plate_no = test_data["plate_no"].iloc[0]
        operation_id = test_data["operation_id"].iloc[0]
        data_index = test_data["data_index"].iloc[0]

        series_station_seq = set(list(test_data["station_seq"]))
        missing_seq_list = [x for x in range(1, n + 1) if x not in series_station_seq]
        for seq_case in missing_seq_list:
            if seq_case < k:
                if self.prev_mean_hash.get((route_id, seq_case)) != None:
                    case = {
                        "route_id": route_id,
                        "dow": dow,
                        "plate_no": plate_no,
                        "operation_id": operation_id,
                        "station_id": self.distance_hash[(route_id, seq_case)][0],
                        "station_seq": seq_case,
                        "hour": np.nan,
                        "next_station_distance": self.distance_hash[
                            (route_id, seq_case)
                        ][1],
                        "data_index": data_index,
                        "prev_duration": self.prev_mean_hash[(route_id, seq_case)],
                    }
                    data_entry_list.append(case)
                else:
                    if (route_id == 1358 and seq_case == 124) or (
                        route_id == 1067 and seq_case == 116
                    ):
                        case = {
                            "route_id": route_id,
                            "dow": dow,
                            "station_id": self.distance_hash[(route_id, seq_case)][0],
                            "station_seq": seq_case,
                            "hour": np.nan,
                            "next_station_distance": self.distance_hash[
                                (route_id, seq_case)
                            ][1],
                            "data_index": data_index,
                            "prev_duration": 38,
                        }
                        data_entry_list.append(case)
                    else:
                        raise ValueError
            elif seq_case >= k:
                case = {
                    "route_id": route_id,
                    "dow": dow,
                    "plate_no": plate_no,
                    "operation_id": operation_id,
                    "station_id": self.distance_hash[(route_id, seq_case)][0],
                    "station_seq": seq_case,
                    "hour": np.nan,
                    "next_station_distance": self.distance_hash[(route_id, seq_case)][
                        1
                    ],
                    "data_index": data_index,
                    "prev_duration": np.nan,
                }
                data_entry_list.append(case)
        new = pd.DataFrame(data_entry_list)
        test_data = pd.concat([test_data, new]).sort_values(
            by=["data_index", "station_seq"], ignore_index=True
        )
        """
        outlier 제거!!!!
        """
        idx = test_data[test_data["prev_duration"] > 1500].index
        test_data.loc[idx, ["prev_duration"]] = test_data.loc[
            idx, ["prev_duration", "route_id", "station_seq"]
        ].apply(
            lambda x: hash_mapping(
                False, "median", x[0], x[1], x[2], self.prev_median_hash
            ),
            axis=1,
        )

        # concat = concat.groupby(['data_index']).fillna(method = 'ffill').fillna(method = 'bfill')
        test_data_hour = (
            test_data.groupby(["data_index"])
            .fillna(method="ffill")
            .fillna(method="bfill")
        )
        test_data["hour"] = test_data_hour["hour"]

        return test_data

    def delete_duplication_for_test(self, test_data):
        s_s = np.array(test_data["station_seq"].values)
        unique = np.unique(s_s, return_counts=True)
        if np.any(unique[1] > 1):
            test_data = test_data.iloc[1:]
            test_data = test_data.reset_index(drop=True)
        return test_data

    def replace_outlier_using_mean_for_test(self, test_data):
        idx = test_data[test_data["prev_duration"] < 5].index
        test_data.loc[idx, ["prev_duration"]] = test_data.loc[
            idx, ["prev_duration", "route_id", "station_seq"]
        ].apply(
            lambda x: hash_mapping(
                False, "mean", x[0], x[1], x[2], self.prev_mean_hash
            ),
            axis=1,
        )

        return test_data

    def preprocess_test_data(self, test_data):

        data_index = test_data["data_index"].iloc[0]
        route_id = test_data["route_id"].iloc[0]
        plate_no = test_data["plate_no"].iloc[0]
        operation_id = test_data["operation_id"].iloc[0]

        info = {
            "data_index": data_index,
            "route_id": route_id,
            "plate_no": plate_no,
            "operation_id": operation_id,
        }

        # windowing to get prev duration
        test_data["prev_ts"] = test_data["ts"].shift(1)
        test_data["prev_seq"] = test_data["station_seq"].shift(1)
        test_data["prev_ts"] = test_data["prev_ts"].fillna(0)
        test_data["prev_duration"] = np.where(
            test_data["prev_ts"] == 0, 0, test_data["ts"] - test_data["prev_ts"]
        )
        test_data["check_seq"] = np.where(
            test_data["station_seq"] - test_data["prev_seq"] == 1, True, False
        )

        # raise Exception(f"{test_data}")

        # find N and K
        test_data_selected = test_data.sort_values(
            ["data_index", "station_seq"]
        ).reset_index(drop=True)
        null_data = test_data_selected[test_data_selected["ts"].isnull()]

        k = null_data["station_seq"].min()
        n = null_data["station_seq"].max()

        if route_id == 5113 and k <= 61:
            flag_300 = True
        else:
            flag_300 = False

        if route_id == 1346 and k <= 33:
            flag_1346 = True
        else:
            flag_1346 = False

        flag = True
        if isinstance(k, float) == False:
            flag = False
            test_data_selected = self.process_missing_value_for_test(
                test_data=test_data_selected, k=k, n=n
            )
            test_data_selected = self.delete_duplication_for_test(test_data_selected)
            test_data_selected = self.replace_outlier_using_mean_for_test(
                test_data_selected
            )

            direc_df = pd.merge(
                test_data_selected,
                self.route_df[["route_id", "turning_point_sequence"]],
                on="route_id",
            )
            test_data_selected["direction"] = np.where(
                test_data_selected["station_seq"] <= direc_df["turning_point_sequence"],
                0,
                1,
            )

            test_data_selected["prev_station_distance"] = test_data_selected[
                "next_station_distance"
            ].shift(1)
            test_data_selected["prev_station_distance"] = test_data_selected[
                "prev_station_distance"
            ].fillna(0)

            test_data_selected[
                "next_station_distance"
            ] = self.dist_normalizer.normalize(
                test_data_selected["next_station_distance"]
            )
            test_data_selected[
                "prev_station_distance"
            ] = self.dist_normalizer.normalize(
                test_data_selected["prev_station_distance"]
            )
            test_data_selected["prev_duration"] = self.dur_normalizer.normalize(
                test_data_selected["prev_duration"]
            )

            test_data_selected["route_id"] = self.route_encoder.transform(
                test_data_selected["route_id"]
            )
            test_data_selected["station_id"] = self.station_encoder.transform(
                test_data_selected["station_id"]
            )
            # raise Exception(f"{test_data_selected}")
            # if k < 32:
            #     raise Exception(f"\n\nk: {k}\n\n test data: \n{test_data}\n\n\n test_data selected :\n{test_data_selected}\n\n ")
            unique = np.unique(
                np.array(test_data_selected["station_seq"]), return_counts=True
            )
            if np.any(unique[1] > 1):
                raise Exception(
                    f"\n unique : {unique}, \n\n test_data: \n {test_data}\n\ntest_data_selected:\n {test_data_selected}"
                )

        # 총 n-k + 1개를 예측해야함.
        return test_data_selected, k, n, info, flag, flag_300, flag_1346


if __name__ == "__main__":
    preprocessor = Preprocessor(None)
    preprocessor.preprocess_train_dataset()
