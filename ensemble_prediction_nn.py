import time
import logging
import copy

import tensorflow as tf
import numpy as np
import scipy.stats
from numpy import ndarray

from nn_worker import NNWorker
from parallel_worker import *

class EnsemblePredictionNN:
    def set_model_params(self, model_params_list):
        for nn, model_param in zip(self.nns, model_params_list):
            print("SETTING MODEL PARAMS")
            nn.set_model_params(model_param)

    def fit(self, X, y):
        worker_nns = [copy.deepcopy(nn) for nn in self.nns]
        for nn in self.nns:
            nn._init_nn()
        rand_seed = np.random.randint(1)
        worker_list = [
                NNWorker(rand_seed + idx, nn)
                for idx, nn in enumerate(worker_nns)]
        if self.do_distributed and len(worker_list) > 1:
            print("DISTRIBUTED")
            # Submit jobs to slurm
            batch_manager = BatchSubmissionManager(
                    worker_list=worker_list,
                    shared_obj=(X,y),
                    num_approx_batches=len(worker_list),
                    worker_folder=self.scratch_dir)
            results = batch_manager.run()
            print("RESULTS", results)
        else:
            # Run jobs locally
            print("LOCAL")
            results = [worker.run_worker((X, y)) for worker in worker_list]

        self.set_model_params(results)

    def score(self, x, y, max_replicates=100):
        """
        This is a placeholder score func... for grid search to assess score
        """
        sess = tf.Session()
        all_losses = []
        with sess.as_default():
            # Re-initialize neural network params since we are in a new session
            for pred_nn in self.nns:
                pred_nn._init_network_variables(sess)
                num_replicates = max_replicates if self.dropout_rate > 0 else 1
                mean_obs_losses = 0
                for i in range(num_replicates):
                    obs_losses = sess.run(
                            pred_nn.prediction_loss_obs,
                        feed_dict={
                            pred_nn.x_concat_ph: x,
                            pred_nn.y_concat_ph: y,
                        })
                    mean_obs_losses += obs_losses
                mean_obs_losses = mean_obs_losses/num_replicates
                loss = np.mean(mean_obs_losses)
                all_losses.append(loss)
        sess.close()
        return -np.mean(all_losses)

    def get_prediction_loss_obs(self, x, y, max_replicates=100):
        sess = tf.Session()
        mean_all_losses = 0
        with sess.as_default():
            # Re-initialize neural network params since we are in a new session
            for pred_nn in self.nns:
                pred_nn._init_network_variables(sess)
                num_replicates = max_replicates if self.dropout_rate > 0 else 1
                mean_obs_losses = 0
                for i in range(num_replicates):
                    obs_losses = sess.run(
                            pred_nn.prediction_loss_obs,
                        feed_dict={
                            pred_nn.x_concat_ph: x,
                            pred_nn.y_concat_ph: y,
                        })
                    mean_obs_losses += obs_losses
                mean_obs_losses /= num_replicates
                mean_all_losses += mean_obs_losses
        sess.close()
        return mean_all_losses/len(self.nns)
