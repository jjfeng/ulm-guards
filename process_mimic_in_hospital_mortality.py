import os
import argparse
import numpy as np
import sys
from sklearn.impute import SimpleImputer

sys.path.insert(0, 'mimic3_benchmarks')

from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3models import common_utils
from dataset import Dataset
from support_sim_settings import SupportSimSettingsComplex
from common import pickle_to_file


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--features',
            type=str,
            default='all',
            help='specifies what features to extract',
            choices=['all', 'len', 'all_but_len'])
    parser.add_argument(
            '--inflation-factor',
            type=float,
            default=0.5,
            help='how much to inflate for the support of X')
    parser.add_argument(
            '--data',
            type=str,
            help='Path to the data of in-hospital mortality task',
            default='../data/mimic/in-hospital-mortality/')
    parser.add_argument(
            '--out-train-data',
            type=str,
            help='file for storing output training dataset',
            default='../data/mimic/in-hospital-mortality/mimic_in_hospital_train.pkl')
    parser.add_argument(
            '--out-test-data',
            type=str,
            help='file for storing output test dataset',
            default='../data/mimic/in-hospital-mortality/mimic_in_hospital_test.pkl')
    args = parser.parse_args()
    return args

def read_and_extract_features(args, partition):
    data_folder = os.path.join(args.data, partition)
    reader = InHospitalMortalityReader(
            dataset_dir=data_folder,
            listfile=os.path.join(data_folder, 'listfile.csv'))

    ret = common_utils.read_chunk(reader, reader.get_number_of_examples())
    ret["meta"] = np.stack(ret["meta"])
    patients = np.array(ret["patient"], dtype=int)
    X = common_utils.extract_features_from_rawdata(ret['X'], ret['header'], period="all", features=args.features)

    # Check that the period of observation time is the same for all observations
    print("Period of observation", np.mean(ret["t"]), np.var(ret["t"]))
    assert np.var(ret["t"]) < 1e-3

    # Augment data with missing columns
    missing_flags = np.isnan(X)
    # Also add in the metadata (age, ethnicity, gender)
    augmented_X = np.concatenate([ret["meta"], X, missing_flags], axis=1)
    y = np.array(ret['y']).reshape((-1,1))
    return augmented_X, y, patients

def main(args=sys.argv[1:]):
    args = parse_args(args)
    print(args)

    train_aug_X, train_y, train_patients = read_and_extract_features(args, "train")
    test_aug_X, test_y, test_patients = read_and_extract_features(args, "test")

    print('Imputing missing values ...')
    # Impute things
    imputer = SimpleImputer(strategy="median")
    imputer.fit(train_aug_X)
    imputed_train_X = imputer.transform(train_aug_X)
    print("train data shape", imputed_train_X.shape)
    imputed_test_X = imputer.transform(test_aug_X)

    # Save things
    train_data = Dataset(x=imputed_train_X, y=train_y, group_id=train_patients)
    support_sim_settings = SupportSimSettingsComplex.create_from_dataset(train_data.x, args.inflation_factor)
    train_data_dict = {
            "train": train_data,
            "support_sim_settings": support_sim_settings,
            "imputer": imputer}
    pickle_to_file(train_data_dict, args.out_train_data)

    test_data = Dataset(x=imputed_test_X, y=test_y, group_id=test_patients)
    pickle_to_file(test_data, args.out_test_data)

if __name__ == '__main__':
    main(sys.argv[1:])
