import os
import argparse
import numpy as np
import sys
import tensorflow as tf
import cv2

from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

from dataset import Dataset
from support_sim_settings import SupportSimSettingsContinuousMulti
from common import pickle_to_file

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--data',
            type=str,
            help='Path to the data of mnist',
            default='../data/mnist')
    parser.add_argument(
            '--out-train-data',
            type=str,
            help='file for storing output training dataset',
            default="../data/mnist/mnist_train_pca.pkl")
    parser.add_argument(
            '--out-test-data',
            type=str,
            help='file for storing output test dataset',
            default="../data/mnist/mnist_test_pca.pkl")
    parser.add_argument(
            '--out-weird-data',
            type=str,
            help='file for storing output test dataset',
            default="../data/mnist/weird_mnist_test_pca.pkl")
    parser.add_argument(
            '--do-pca',
            action="store_true")
    args = parser.parse_args()
    return args

def main(args=sys.argv[1:]):
    args = parse_args(args)
    print(args)
    np.random.seed(0)

    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    orig_image_shape = x_train.shape[1:]
    x_train = x_train.reshape((x_train.shape[0], -1))
    x_test = x_test.reshape((x_test.shape[0], -1))

    num_classes = 10
    num_train_classes = 9

    data_mask = y_train < num_train_classes
    x_train = x_train[data_mask]
    y_train = y_train[data_mask]
    y_train_categorical = np.zeros((y_train.size, num_train_classes))
    y_train_categorical[np.arange(y_train.size),y_train] = 1

    y_test_categorical = np.zeros((y_test.size, num_classes))
    y_test_categorical[np.arange(y_test.size),y_test] = 1

    (_, _), (weird_x,_) = tf.keras.datasets.fashion_mnist.load_data()
    weird_x = weird_x / 255.0
    weird_x = weird_x.reshape((weird_x.shape[0], -1))

    if args.do_pca:
        pca = PCA(n_components=300, whiten=True)
        x_train = pca.fit_transform(x_train)
        print(pca.explained_variance_ratio_)
        x_test = pca.transform(x_test)
        weird_x = pca.transform(weird_x)

    num_p = x_train.shape[1]
    min_x = np.min(np.concatenate([x_train, x_test]), axis=0).reshape((1,-1))
    max_x = np.max(np.concatenate([x_train, x_test]), axis=0).reshape((1,-1))
    support_sim_settings = SupportSimSettingsContinuousMulti(
            num_p,
            min_x=min_x,
            max_x=max_x)


    train_data = Dataset(x=x_train, y=y_train_categorical, num_classes=num_train_classes)
    train_data_dict = {
            "train": train_data,
            "support_sim_settings": support_sim_settings}
    pickle_to_file(train_data_dict, args.out_train_data)

    random_idx = np.random.choice(x_test.shape[0], size=4000, replace=False)
    test_data = Dataset(x=x_test[random_idx, :], y=y_test_categorical[random_idx, :], num_classes=num_classes)
    pickle_to_file(test_data, args.out_test_data)

    check_supp = support_sim_settings.check_obs_x(weird_x)
    print("NUM WEIRD", weird_x.shape)
    print("NUM WEiRD IN SUPPORT", np.sum(check_supp))
    weird_x = weird_x[check_supp,:]
    pickle_to_file(weird_x, args.out_weird_data)

if __name__ == '__main__':
    main(sys.argv[1:])
