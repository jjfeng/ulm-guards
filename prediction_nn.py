import time
import logging

import tensorflow as tf
import numpy as np
from numpy import ndarray

from neural_network import NeuralNetwork
from neural_network_wrapper import NeuralNetworkParams

class PredictionNN:
    def set_model_params(self, model_params: NeuralNetworkParams):
        self.model_params = model_params

    def fit(self, X, y, max_tries=3):
        st_time = time.time()
        init_params = [np.copy(p) for p in self.model_params.param_list] if self.model_params is not None else []

        sess = tf.Session()
        best_loss = None
        with sess.as_default():
            for n_init in range(self.num_inits):
                logging.info("FIT INIT %d", n_init)
                tf.global_variables_initializer().run()
                for i in range(max_tries):
                    if init_params:
                        self._init_network_variables(sess, init_params)
                    model_params, train_loss = self._fit_one_init(sess, X, y)
                    if train_loss is not None:
                        break
                if best_loss is None or train_loss < best_loss:
                    self.model_params = model_params
                    #logging.info("model params update %s", str(self.model_params))
                    best_loss = train_loss
                for p in init_params:
                    p += np.random.randn(*p.shape) * 0.01

        logging.info("best_loss %s (train time %f)", str(best_loss), time.time() - st_time)
        sess.close()
        return model_params

    def _fit_one_init(self, sess, x_train, y_train, log_show=50, stop_thres=1, stop_ratio=1e-10, min_iters=500):
        """
        Fitting function for one initialization -- performing SGD
        @param log_show: number of iters to run before printing things
        """
        init_params = [np.array(p.eval(),dtype=float) for p in self.full_param_list]

        num_obs = y_train.shape[0]
        num_sgd_slices = int(num_obs/self.sgd_batch_size)
        if num_obs > self.sgd_batch_size * num_sgd_slices:
            num_sgd_slices += 1
        print("sgd slices", num_sgd_slices, "num obs", num_obs)

        # PRETRAINING: Train first where we accept all points
        prev_pen_loss = None
        for i in range(self.max_iters):
            shuffled_indices = np.random.permutation(np.arange(num_obs))
            x_train_shuffled = x_train[shuffled_indices, :]
            y_train_shuffled = y_train[shuffled_indices, :]
            do_exit = False
            for j in range(num_sgd_slices):
                start_idx = j * self.sgd_batch_size
                stop_idx = (j + 1) * self.sgd_batch_size
                x_train_sgd = x_train_shuffled[start_idx:stop_idx, :]
                y_train_sgd = y_train_shuffled[start_idx:stop_idx, :]

                # Do train step
                feed_dict = {
                    self.x_concat_ph: x_train_sgd,
                    self.y_concat_ph: y_train_sgd}
                for k, v in zip(self.ref_param_phs, init_params):
                    feed_dict[k] = v
                _, pen_loss, unpen_loss = sess.run(
                    [
                        self.train_op,
                        self.pen_loss,
                        self.loss,
                    ],
                    feed_dict=feed_dict)

                inner_idx = i * num_sgd_slices + j
                if inner_idx % log_show == 0:
                    #print("coef", [c.eval() for c in self.prediction_nn.coefs])
                    #print("inters", [c.eval() for c in self.prediction_nn.intercepts])
                    logging.info("train %d: (epoch %d), penalized-loss-all %f", inner_idx, i, pen_loss)
                    logging.info("train %d: (epoch %d), UNpenalized-loss-all %f", inner_idx, i, unpen_loss)

                # Stop if Adam is doing something crazy
                if prev_pen_loss is not None and pen_loss > max(2 * prev_pen_loss, prev_pen_loss + stop_thres):
                    logging.info("BREAK curr %f prev %f", pen_loss, prev_pen_loss)
                    do_exit = True
                    # Indicate something terrible happened and you should start over
                    pen_loss = None
                # Stop if Adam seems to have converged
                elif prev_pen_loss is not None and np.abs(pen_loss - prev_pen_loss)/prev_pen_loss < stop_ratio:
                    logging.info("BREAK curr %f prev %f", pen_loss, prev_pen_loss)
                    do_exit = True
                prev_pen_loss = pen_loss
            if i > self.max_iters/2 and do_exit:
                break

        # Save model parameter values. Otherwise they will disappear!
        model_params = NeuralNetworkParams(
            [c.eval() for c in self.full_param_list],
        )
        return model_params, pen_loss

    def _setup_optimization_problem(self):
        # Create prediction losses
        # The version where the prediction loss is weighted based on the decision function
        self.prediction_loss = tf.reduce_mean(self.prediction_loss_obs)

        self.full_param_list = self.prediction_nn.coefs + self.prediction_nn.intercepts
        self.ref_param_phs = [
                tf.placeholder(tf.float32, params.shape, name="huh%d" % idx)
                for idx, params in enumerate(self.full_param_list)]
        ## Create weight loss for decision network; adds loss for each coefficient in neural network to tensorflow
        if self.weight_penalty_type == "ridge":
            self.weight_prediction = tf.add_n([tf.nn.l2_loss(w) for w in self.prediction_nn.coefs])
        elif self.weight_penalty_type == "group_lasso":
            self.weight_prediction = tf.add_n(
                [tf.reduce_sum(tf.sqrt(tf.reduce_mean(tf.pow(self.prediction_nn.coefs[0], 2), axis=1)))]
                + [tf.nn.l2_loss(w) for w in self.prediction_nn.coefs[1:]])
        elif self.weight_penalty_type == "to_reference":
            self.weight_prediction = tf.reduce_sum([
                tf.nn.l2_loss(param - ph) for param, ph in zip(self.full_param_list, self.ref_param_phs)])
        else:
            raise ValueError("nope dont know what you want")
        self.weight_prediction_pen = self.prediction_weight_param * self.weight_prediction
        self.weight_pen = self.weight_prediction_pen

        self.loss = self.prediction_loss
        self.pen_loss = self.loss + self.weight_pen

        # Optimizer for simultaneous tuning of decision and density functions
        self.adam_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.adam_optimizer.minimize(
                self.pen_loss,
                var_list=self.full_param_list)

        # Creating parameter assignment placeholders and assignment operations
        self.param_phs, self.param_assign_ops = self._create_ph_assign_ops(self.full_param_list)

    def _create_ph_assign_ops(self, var_list):
        """
        Create placeholders and assign ops for model parameters
        """
        all_phs = []
        assign_ops = []
        for var in var_list:
            ph = tf.placeholder(
                tf.float32,
                shape=var.shape,
            )
            assign_op = var.assign(ph)
            all_phs.append(ph)
            assign_ops.append(assign_op)
        return all_phs, assign_ops

    def _init_network_variables(self, sess, param_force = None):
        """
        Initialize network variables
        """
        param_list = self.model_params.param_list if param_force is None else param_force
        for i, c_val in enumerate(param_list):
            sess.run(self.param_assign_ops[i], feed_dict={self.param_phs[i]: c_val})

    def get_univar_mapping(self, x, max_replicates=100):
        sess = tf.Session()
        with sess.as_default():
            # Re-initialize neural network params since we are in a new session
            self._init_network_variables(sess)
            num_replicates = max_replicates if self.dropout_rate > 0 else 1
            mean_univar_mapping = 0
            for i in range(num_replicates):
                univar_mapping = sess.run(
                        self.decision_univar_mapping,
                    feed_dict={
                        self.x_concat_ph: x,
                    })
                mean_univar_mapping += univar_mapping
            mean_univar_mapping /= num_replicates
        sess.close()
        return mean_univar_mapping
