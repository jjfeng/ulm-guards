import os
import sys
import csv
import json

import scipy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from common import pickle_to_file
import data_generator

def _get_last_datapoint(df):
    return df.Value.values[-1]

def _get_mean(df):
    if df.Value.size == 1:
        return df.Value.values[0]
    elif np.unique(df.Time).size == 1:
        return df.Value.mean()

    mean_time = df.Time.mean()
    lin_fit = scipy.stats.linregress(df.Time - mean_time, df.Value)
    return lin_fit[1]

def _get_max(df):
    return df.Value.max()

def _get_min(df):
    return df.Value.min()

def _get_sum(df):
    return df.Value.sum()

def _get_identity(x):
    return x.Value.values[0]

def _get_slope(df):
    if df.Value.size == 1 or np.unique(df.Time).size == 1:
        return 0
    return scipy.stats.linregress(df.Time/50., df.Value)[0]

LAST = _get_last_datapoint
MIN = _get_min
MAX = _get_max
WEIGHTED_MEAN = _get_mean
SUM = _get_sum
IDENTITY = _get_identity
SLOPE = _get_slope

FEATURES = {
    # Based on the paper
    "GCS": [SLOPE, LAST, WEIGHTED_MEAN, MAX, MIN],
    "HCO3": [MIN, MAX, LAST, WEIGHTED_MEAN],
    "BUN": [MIN, MAX, LAST, WEIGHTED_MEAN],
    "Urine": [SUM],
    "Age": [IDENTITY],
    "SysABP": [WEIGHTED_MEAN, LAST, MIN, MAX],
    "WBC": [LAST, WEIGHTED_MEAN, MIN, MAX],
    "Temp": [WEIGHTED_MEAN, LAST, MIN, MAX],
    "Glucose": [MAX, MIN, WEIGHTED_MEAN],
    "Na": [WEIGHTED_MEAN, MAX, MIN],
    "Lactate": [LAST, WEIGHTED_MEAN, MIN, MAX],
    # Based on SAPS II or SAPS I (https://physionet.org/challenge/2012/saps_score.m)
    "HR": [MIN, MAX, WEIGHTED_MEAN],
    "K": [MIN, MAX, WEIGHTED_MEAN],
    "ICUType": [IDENTITY],
    "HCT": [WEIGHTED_MEAN, MIN, MAX],
    "RespRate": [WEIGHTED_MEAN, MIN, MAX],
    "MechVent": [MAX],
    # Based on most common measurements
    #"Creatinine": [WEIGHTED_MEAN, MIN, MAX],
    #"Platelets": [WEIGHTED_MEAN, MIN, MAX],
    #"Mg": [WEIGHTED_MEAN, MIN, MAX],
    # Baseline measurements, general descriptors
    "Gender":[IDENTITY],
    "Weight":[IDENTITY],
    "Height":[IDENTITY],
}

META_FEATURE_GROUPS = {
        "GCS": ["GCS"],
        "Metabolic": ["HCO3", "BUN", "Na", "K", "Glucose"],
        "SysABP": ["SysABP"],
        "CBC": ["WBC", "HCT"],
        "Temp": ["Temp"],
        "Lactate": ["Lactate"],
        "HR": ["HR"],
        "Respiration": ["RespRate", "MechVent", "O2"],
        "Urine": ["Urine"],
        "General Desc": ["Gender", "Height", "Weight", "Age", "ICUType"],
        }

NORMAL_RANGES = {
    "GCS": [15,15],
    "HCO3": [20, 30],
    "BUN": [8, 28],
    "Urine": [2000, 4000],
    "SysABP": [100,199],
    "WBC": [1, 19.9],
    "Temp": [36,38.9],
    "Glucose": [62, 125],
    "Na": [135, 145],
    "Lactate": [0.5, 1],
    "HR": [70,119],
    "K": [3.6,5.2],
    "HCT": [36, 45],
    "RespRate": [12,20],
    "MechVent": [0,0],
    "O2": [200,250],
}

MAX_PROCESS = 5000

def main(args=sys.argv[1:]):
    train_size = 0.5
    seed = 0

    # Read the y data
    outcomes = pd.read_csv("../data/Outcomes-a.txt")
    subject_outcomes = outcomes[["RecordID", "Length_of_stay", "Survival"]]

    # Create a dictionary of features for each subject
    # Using a dictionary because some of the features don't appear in all subjects...
    value_range = {} # this is just for printing out ranges of the values
    file_folder = "../data/set-a/"
    all_subject_features = {}
    for idx, filename in enumerate(os.listdir(file_folder)[:MAX_PROCESS]):
        df = pd.read_csv("%s%s" % (file_folder, filename))
        df["hour"] = np.array([time.split(":")[0] for time in df.Time.values], dtype=int)
        df["minute"] = np.array([time.split(":")[1] for time in df.Time.values], dtype=int)
        df.Time = df.hour * 60 + df.minute

        record_id = int(df.loc[0].Value)
        subject_features = {"RecordID": record_id}
        for feat_name, process_func_list in FEATURES.items():
            if WEIGHTED_MEAN in process_func_list:
                sub_df = df.loc[(df.Parameter == feat_name) & (df.Value > 0)]
            else:
                sub_df = df.loc[(df.Parameter == feat_name) & (df.Value >= 0)]

            if sub_df.shape[0] == 0:
                continue
            if feat_name not in value_range:
                value_range[feat_name] = [sub_df.Value.min(), sub_df.Value.max()]
            else:
                value_range[feat_name][0] = min(value_range[feat_name][0], sub_df.Value.min())
                value_range[feat_name][1] = max(value_range[feat_name][1], sub_df.Value.max())

            for func in process_func_list:
                value = func(sub_df)
                if not np.isfinite(value):
                    print(value, feat_name, func.__name__)
                    print(sub_df)
                assert np.isfinite(value)
                full_feature_name = "%s:%s" % (feat_name, func.__name__)
                subject_features[full_feature_name] = value

        fio2_df = df.loc[df.Parameter == "FiO2"]
        pao2_df = df.loc[df.Parameter == "PaO2"]
        if fio2_df.shape[0] and pao2_df.shape[0]:
            fio2_mean = _get_mean(fio2_df)
            pao2_mean = _get_mean(pao2_df)
            if fio2_mean > 0:
                subject_features["O2:_get_ratio"] = pao2_mean/fio2_mean

        all_subject_features[idx] = subject_features

    for k, v in value_range.items():
        print(k, v)

    subjects_x = pd.DataFrame.from_dict(all_subject_features, orient="index")

    # Merge the X and Y data
    icu_subjects = subjects_x.merge(subject_outcomes, on="RecordID")
    print(icu_subjects["Survival"])
    icu_subjects["resp"] = np.maximum(
            icu_subjects["Length_of_stay"],
            icu_subjects["Survival"])
    icu_subjects = icu_subjects.drop(columns=["RecordID"])
    print(np.mean(icu_subjects["Survival"]))
    print(np.median(icu_subjects["Survival"]))
    print(np.max(icu_subjects["Survival"]))
    print(np.mean(icu_subjects["Length_of_stay"]))
    print(np.median(icu_subjects["Length_of_stay"]))
    print(np.max(icu_subjects["Length_of_stay"]))

    # Grab column names
    column_names = list(icu_subjects.columns.values)
    icu_subjects = icu_subjects.as_matrix()

    # Center the x covariates
    centering_term = np.nanmean(icu_subjects, axis=0)
    centering_term[-1] = 0
    icu_subjects[:, :-3] -= centering_term[:-3]

    # randomly split the data
    print(column_names)
    mats = train_test_split(icu_subjects, train_size = train_size, test_size = 1.0 - train_size, random_state = seed)
    x_train = mats[0][:, :-3]
    y_train = mats[0][:, -1:]
    y_censored_train = mats[0][:, -2:-1] < 0
    x_test = mats[1][:, :-3]
    y_test = mats[1][:, -1:]
    y_censored_test = mats[1][:, -2:-1] < 0

    # Save the data
    icu_train_data = data_generator.Dataset(
        x= x_train,
        y= y_train,
        is_censored = y_censored_train)
    icu_test_data = data_generator.Dataset(
        x = x_test,
        y = y_test,
        is_censored = y_censored_test)

    ## save off as a pickle
    icu_processed_file = "../data/icu_data_processed.pkl"
    pickle_to_file({
        "train": icu_train_data,
        "test": icu_test_data},
        icu_processed_file)

    icu_column_file = "../data/icu_data_column_names.txt"
    with open(icu_column_file, "w") as f:
        for i, col in enumerate(column_names[:-1]):
            f.write("%d, %s\n" % (i, col))

if __name__ == "__main__":
    main(sys.argv[1:])
