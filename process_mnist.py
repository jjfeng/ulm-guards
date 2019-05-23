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
            default="/fh/fast/matsen_e/jfeng2/mnist/mnist_train_pca.pkl")
    parser.add_argument(
            '--out-test-data',
            type=str,
            help='file for storing output test dataset',
            default="/fh/fast/matsen_e/jfeng2/mnist/mnist_test_pca.pkl")
    parser.add_argument(
            '--out-weird-data',
            type=str,
            help='file for storing output test dataset',
            default="/fh/fast/matsen_e/jfeng2/mnist/weird_mnist_test_pca.pkl")
    parser.add_argument(
            '--do-pca',
            action="store_true")
    parser.add_argument(
        '--not-mnist-folder',
        type=str,
        default='../data/notMNIST_small/A')
    parser.add_argument(
            '--out-not-mnist-data',
            type=str,
            help='file for storing output test dataset',
            default='../data/notMNIST_small/no_mnist_test_pca.pkl')
    parser.add_argument(
            '--out-random-images-plot',
            type=str,
            default='_output/images/random_mnist_kinda_images.png')
    args = parser.parse_args()
    return args

def read_notmnist(folder_name):
    images = [
            file_name for file_name in os.listdir(folder_name)
            if file_name.endswith(".png") and not file_name.startswith("._")]
    all_data = []
    for image_name in images:
        img = cv2.imread(os.path.join(folder_name, image_name))
        if img is not None:
            all_data.append([img[:,:,0]])
    X = np.vstack(all_data)
    print(X.shape)
    return X

def transform_data(x, pca):
    x = x.reshape((x.shape[0], -1))/255.0
    return x
    #return pca.transform(x)

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
    #print(max_x - min_x)
    support_sim_settings = SupportSimSettingsContinuousMulti(
            num_p,
            min_x=min_x,
            max_x=max_x)
    #diffs = max_x - min_x
    #print(np.sum(diffs > 0))
    #print(np.sum(np.log(diffs[diffs > 0])))
    print(x_train.shape)
    print(x_test.shape)


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

    #num_points = 10
    #random_x = support_sim_settings.support_unif_rvs(num_points)
    #random_imgs = pca.inverse_transform(random_x)
    #random_imgs = random_imgs.reshape(
    #        (num_points, orig_image_shape[0], orig_image_shape[1]))
    #fig, ax = plt.subplots(num_points)
    #for i in range(num_points):
    #    im = ax[i].imshow(random_imgs[i],
    #                         cmap=plt.cm.binary, interpolation='nearest')
    #plt.savefig(args.out_random_images_plot)

    #not_mnist_x = read_notmnist(args.not_mnist_folder)
    ##not_mnist_x = transform_data(not_mnist_x, None)
    #not_mnist_x = not_mnist_x.reshape((not_mnist_x.shape[0], -1))/255.0
    #random_idx = np.random.choice(not_mnist_x.shape[0], size=1000, replace=False)
    ##check_supp = support_sim_settings.check_obs_x(not_mnist_x)
    ##print("num not mnist", np.sum(check_supp))
    #pickle_to_file(not_mnist_x[random_idx,:], args.out_not_mnist_data)

if __name__ == '__main__':
    main(sys.argv[1:])
