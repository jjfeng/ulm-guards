import time
import logging

import tensorflow as tf
import numpy as np

from decision_prediction_base_model import DecisionPredictionBaseModel
from neural_network_wrapper import NeuralNetworkParams
from neural_network import NeuralNetwork
from common import process_params


class DecisionPredictionNNs(DecisionPredictionBaseModel):
    """
    Fits a `prediction_nn`
    """
    def set_model_params(self, model_params: NeuralNetworkParams):
        self.model_params = model_params

    def _get_model_params(self):
        return NeuralNetworkParams(
            [c.eval() for c in self.full_param_list],
        )

    def _setup_skeleton(self):
        self.input_shape = process_params(self.prediction_layer_sizes[0], int, split_str="~")
        self.num_p = self.input_shape[0]

        # Input layers; create placeholders in tensorflow
        self.x_concat_ph = tf.placeholder(tf.float32, [None] + self.input_shape, name="x")
        self.y_concat_ph = tf.placeholder(tf.float32, [None, self.label_size], name="y")
        self.is_data_ph = tf.placeholder(tf.float32, [None, 1], name="is_data")
        self.min_y = None

    def _create_decision_model(self, max_sigmoid_scale: float=100.0, init_scale: float = 0.5):
        """
        The decision function here is forced to take on the functional form of the population version
        """
        if self.decision_layer_sizes is not None:
            print("CREATING DECISION NN")
            decision_nn = NeuralNetwork.create_full_nnet(
                    self.decision_layer_sizes,
                    self.x_concat_ph,
                    act_func=getattr(tf.nn, self.act_func),
                    output_act_func=tf.nn.sigmoid,
                    dropout_rate=self.dropout_rate)
            decision_model_params = decision_nn.coefs + decision_nn.intercepts
            accept_prob = decision_nn.layers[-1]
        else:
            scale = tf.Variable(init_scale/self.cost_decline, dtype=tf.float32, name="dec0")
            scale1 = tf.Variable(0, dtype=tf.float32, name="dec1")
            decision_model_params = [scale, scale1]
            accept_prob = tf.nn.sigmoid(
                    (-self.decision_univar_mapping + self.cost_decline) * tf.minimum(tf.exp(scale), float(max_sigmoid_scale)))

        return decision_model_params, accept_prob

    def _setup_optimization_problem(self, inflation_factor: float = 0.5):
        self.decision_model_params, self.accept_prob = self._create_decision_model()

        # Create weight loss for decision network; adds loss for each coefficient in neural network to tensorflow
        if self.weight_penalty_type == "ridge":
            self.weight_prediction = tf.add_n([tf.nn.l2_loss(w) for w in self.prediction_nn.coefs])
        elif self.weight_penalty_type == "group_lasso":
            self.weight_prediction = tf.add_n(
                [tf.reduce_sum(tf.sqrt(tf.reduce_mean(tf.pow(self.prediction_nn.coefs[0], 2), axis=1)))]
                + [tf.nn.l2_loss(self.prediction_nn.coefs[i]) for i in range(1, len(self.prediction_nn.coefs))])

        else:
            raise ValueError("nope dont know what you want")
        self.weight_decision = 0.0 if len(self.decision_model_params) == 0 else tf.add_n([tf.nn.l2_loss(w) for w in self.decision_model_params])
        self.weight_decision_pen = self.decision_weight_param * self.weight_decision
        self.weight_prediction_pen = self.prediction_weight_param * self.weight_prediction
        self.weight_pen = self.weight_prediction_pen + self.weight_decision_pen

        # Optimizer for simultaneous tuning of decision and density functions
        self.adam_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        # This is the objective in round 1 where we accept all data (and no fake data)
        # The prediction loss is just the mean of the prediciton losses
        self.prediction_loss_all = tf.reduce_sum(self.prediction_loss_obs * self.is_data_ph)/tf.reduce_sum(self.is_data_ph)
        self.pen_loss_all = self.prediction_loss_all + self.weight_prediction_pen
        # Only train the prediction NN
        self.train_op_round1 = self.adam_optimizer.minimize(
                self.pen_loss_all,
                var_list=self.prediction_nn.coefs + self.prediction_nn.intercepts)

        # This is the ojective for round 2 where we can reject some of the data.
        # This is the ULM objective.
        self.prediction_loss = tf.reduce_sum(
            self.prediction_loss_obs * self.accept_prob * self.is_data_ph)/tf.reduce_sum(self.is_data_ph)

        # Create loss function (log lik of y given x)
        self.accept_support_proportion = tf.reduce_sum(self.accept_prob * (1.0 - self.is_data_ph))/tf.reduce_sum(1.0 - self.is_data_ph)
        self.tot_cost_reject = self.cost_decline * tf.reduce_sum((1 - self.accept_prob) * self.is_data_ph)/tf.reduce_sum(self.is_data_ph)

        # Do no harm regularization
        self.accept_pen = self.do_no_harm_param * self.accept_support_proportion

        # Do not let the decision function explode to 0 at the very beginning when training
        self.log_barrier = self.log_barrier_param * -tf.log(tf.reduce_sum(self.accept_prob * self.is_data_ph)/tf.reduce_sum(self.is_data_ph))
        self.proportion_data_accept = tf.reduce_sum(self.accept_prob * self.is_data_ph)/tf.reduce_sum(self.is_data_ph)

        self.loss = self.prediction_loss_all * inflation_factor + self.prediction_loss + self.tot_cost_reject
        self.actual_loss = self.prediction_loss + self.tot_cost_reject
        if self.do_no_harm_param > 0 and self.support_sim_num > 0:
            self.pen_loss = self.loss + self.weight_pen + self.accept_pen + self.log_barrier
        else:
            self.pen_loss = self.loss + self.weight_prediction_pen#self.weight_pen

        # Train both the decision and density NNs
        self.full_param_list = self.prediction_nn.coefs + self.prediction_nn.intercepts + self.decision_model_params
        self.train_op_round2 = self.adam_optimizer.minimize(
                self.pen_loss,
                var_list=self.full_param_list)

        # Creating parameter assignment placeholders and assignment operations
        self.param_phs, self.param_assign_ops = self._create_ph_assign_ops(self.full_param_list)

    def _init_nn(self):
        raise NotImplementedError()

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

    def fit(self, X, y, weights=None):
        """
        Fitting function with multiple initializations
        @param weights: ignored
        """
        num_obs = y.shape[0]
        num_p = X.shape[1]

        st_time = time.time()
        logging.info("prediction layer sizes %s do no harm %f", self.prediction_layer_sizes, self.do_no_harm_param)

        sess = tf.Session()
        best_loss = None
        best_idx = None
        self.model_params = None
        with sess.as_default():
            tot_inits = 0
            # Stop random initialization after num_inits of them are good
            for n_init in range(self.num_inits * 3):
                logging.info("FIT INIT %d", n_init)
                tf.global_variables_initializer().run()
                train_loss, accept_data = self._fit_one_init(sess, X, y)
                tot_inits += int(np.isfinite(train_loss) * (accept_data > 0.1))
                logging.info("TRAIN LOSS (tot inits %d): %f", tot_inits, train_loss)
                logging.info("ACCEPT PROB (tot inits %d): %f", tot_inits, accept_data)
                if np.isfinite(train_loss) and (self.model_params is None or train_loss < best_loss):
                    self.model_params = self._get_model_params()
                    #logging.info("model params update %s", str(self.model_params))
                    best_loss = train_loss
                    best_idx = n_init
                if tot_inits >= self.num_inits:
                    break
                else:
                    print("try again")

            #assert self.model_params is not None
            if self.model_params is None:
                tf.global_variables_initializer().run()
                self.model_params = self._get_model_params()

        logging.info("best_loss %s (best idx %s) (train time %f)", str(best_loss), str(best_idx), time.time() - st_time)
        sess.close()
        print("Done with fit")

    def _fit_one_init(self, sess, x_train, y_train, log_show=50, stop_thres=1, stop_ratio=1e-20, min_iters=1000):
        self._fit_one_init_plain(sess, self.pen_loss_all, self.train_op_round1, x_train, y_train, log_show=log_show, stop_thres=stop_thres, stop_ratio=stop_ratio, min_iters=min_iters)
        if self.do_no_harm_param > 0 and self.support_sim_num > 0:
            return self._fit_one_init_with_sim(sess, self.pen_loss, self.train_op_round2, x_train, y_train, log_show=log_show, stop_thres=stop_thres, stop_ratio=stop_ratio, min_iters=min_iters)
        else:
            return self._fit_one_init_plain(sess, self.pen_loss, self.train_op_round2, x_train, y_train, log_show=log_show, stop_thres=stop_thres, stop_ratio=stop_ratio, min_iters=min_iters)

    def _fit_one_init_plain(self, sess, loss_node, train_op, x_train, y_train, log_show=50, stop_thres=1, stop_ratio=1e-8, min_iters=100):
        """
        Fitting function for one initialization -- performing SGD
        @param log_show: number of iters to run before printing things
        """
        num_obs = y_train.shape[0]
        num_sgd_slices = int(num_obs/self.sgd_batch_size)
        if num_obs > self.sgd_batch_size * num_sgd_slices:
            num_sgd_slices += 1
        print("sgd slices", num_sgd_slices, "num obs", num_obs)

        # PRETRAINING: Train first where we accept all points
        prev_tot_loss = None
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
                is_data_sgd = np.ones((x_train_sgd.shape[0], 1))

                # Do train step
                _, tot_loss, pred_loss_all, indiv_pred_loss, single_var, log_barr, accept_prob, actual_loss = sess.run(
                    [
                        train_op,
                        loss_node,
                        self.prediction_loss_all,
                        self.prediction_loss_obs,
                        self.decision_univar_mapping,
                        self.log_barrier,
                        self.accept_prob,
                        self.actual_loss
                    ],
                    feed_dict={
                        self.x_concat_ph: x_train_sgd,
                        self.y_concat_ph: y_train_sgd,
                        self.is_data_ph: is_data_sgd,
                    })

                inner_idx = i * num_sgd_slices + j
                if inner_idx % log_show == 0:
                    logging.info("univar mapping %f %f %f", np.median(single_var), np.max(single_var), np.min(single_var))
                    self._print_train(sess, x_train, y_train)
                    #print("coef", [c.eval() for c in self.prediction_nn.coefs])
                    #print("inters", [c.eval() for c in self.prediction_nn.intercepts])
                    logging.info("train %d: (epoch %d), penalized-loss %f", inner_idx, i, tot_loss)
                    logging.info("train %d: (epoch %d), pred-loss-accept-all %f", inner_idx, i, pred_loss_all)
                    logging.info("train %d: (epoch %d), actual_loss %f", inner_idx, i, actual_loss)

                # Stop if Adam is doing something crazy
                if prev_tot_loss is not None and tot_loss > max(2 * prev_tot_loss, prev_tot_loss + stop_thres):
                    logging.info("BREAK curr %f prev %f", tot_loss, prev_tot_loss)
                    do_exit = True
                # Stop if Adam seems to have converged
                if prev_tot_loss is not None and np.abs(tot_loss - prev_tot_loss)/prev_tot_loss < stop_ratio:
                    logging.info("BREAK curr %f prev %f", tot_loss, prev_tot_loss)
                    do_exit = True
                prev_tot_loss = tot_loss
            if i > min_iters and do_exit:
                break

        return tot_loss, np.mean(accept_prob)

    def _fit_one_init_with_sim(self, sess, loss_node, train_op, x_train, y_train, log_show=50, stop_thres=1, stop_ratio=1e-8, min_iters=100):
        """
        Fitting function for one initialization -- performing SGD
        @param log_show: number of iters to run before printing things
        """
        num_obs = y_train.shape[0]
        num_sgd_slices = int(num_obs/self.sgd_batch_size)
        if num_obs > self.sgd_batch_size * num_sgd_slices:
            num_sgd_slices += 1
        print("sgd slices", num_sgd_slices, "num obs", num_obs)

        # TODO: do we want to run iters until convergence rather than a fixed number of iters?
        prev_tot_loss = None
        accept_unif_proportion = 0.5
        # ACTUAL TRAINING: Now train the decision and density functions simultaneously
        for i in range(self.max_iters):
            support_sim_num = int(min(
                self.support_sim_num,
                50.0/(accept_unif_proportion + 0.01)))
            x_random_support = self.support_sim_settings.support_unif_rvs(support_sim_num * num_sgd_slices)
            if num_sgd_slices > 1:
                shuffled_indices = np.random.permutation(np.arange(num_obs))
                x_train_shuffled = x_train[shuffled_indices]
                y_train_shuffled = y_train[shuffled_indices]
            else:
                x_train_shuffled = x_train
                y_train_shuffled = y_train

            do_exit = False
            for j in range(num_sgd_slices):
                # Subset data for SGD
                start_idx = j * self.sgd_batch_size
                stop_idx = (j + 1) * self.sgd_batch_size
                x_train_sgd = x_train_shuffled[start_idx:stop_idx]
                y_train_sgd = y_train_shuffled[start_idx:stop_idx]
                is_data_sgd = np.ones((x_train_sgd.shape[0], 1))

                # Subset random support points for SGD
                x_random_support_sgd = x_random_support[support_sim_num * j: support_sim_num * (j + 1)]
                y_filler = np.ones((support_sim_num, self.label_size))
                if self.min_y is not None:
                    y_filler = y_filler * (self.min_y + 1)

                # Concatenate the random x's for using tensorflow -- also make binary mask indicating which are
                # data and which are random support points
                x_concat = np.concatenate([x_train_sgd, x_random_support_sgd])
                y_concat = np.concatenate([y_train_sgd, y_filler])
                is_data = np.concatenate([is_data_sgd, np.zeros((support_sim_num, 1))])

                # Do train step
                _, acc_prob, tot_loss, unpen_loss, cost_reject, weight_pen, accept_unif_proportion, accept_pen, log_barr, accept_is_data_proportion, actual_loss = sess.run(
                    [
                        train_op,
                        self.accept_prob,
                        loss_node,
                        self.prediction_loss_all,
                        self.tot_cost_reject,
                        self.weight_pen,
                        self.accept_support_proportion,
                        self.accept_pen,
                        self.log_barrier,
                        self.proportion_data_accept,
                        self.actual_loss,
                    ],
                    feed_dict={
                        self.x_concat_ph: x_concat,
                        self.y_concat_ph: y_concat,
                        self.is_data_ph: is_data,
                    })
                is_data = np.array(is_data.flatten(), dtype=bool)

                inner_idx = i * num_sgd_slices + j
                if inner_idx % log_show == 0:
                    self._print_train(sess, x_train, y_train)
                    logging.info(
                            "iter %d (epoch %d): unpen-loss %f, cost reject %f, do no harm %f, log barr %f, weight_pen %f",
                            inner_idx, i, unpen_loss, cost_reject, accept_pen, log_barr, weight_pen)
                    logging.info("iter %d (epoch %d): acc unif prob %f", inner_idx, i, accept_unif_proportion)
                    logging.info("iter %d (epoch %d): acc is-data prob %f", inner_idx, i, accept_is_data_proportion)
                    logging.info("iter %d (epoch %d): penalized-loss %f", inner_idx, i, tot_loss)
                    logging.info("iter %d (epoch %d): actual-loss %f", inner_idx, i, actual_loss)

                # Stop if Adam is doing something crazy
                if prev_tot_loss is not None and tot_loss > max(2 * prev_tot_loss, prev_tot_loss + stop_thres):
                    logging.info("BREAK curr %f prev %f", tot_loss, prev_tot_loss)
                    tot_loss = np.inf
                    do_exit = True
                # Stop if Adam seems to have converged
                if prev_tot_loss is not None and np.abs(tot_loss - prev_tot_loss)/prev_tot_loss < stop_ratio:
                    logging.info("BREAK curr %f prev %f", tot_loss, prev_tot_loss)
                    do_exit = True
                prev_tot_loss = tot_loss

                if i > self.max_iters/4:
                    if do_exit or accept_is_data_proportion < 0.01:
                        break
            if not np.isfinite(tot_loss):
                print("NOT FINITE", tot_loss)
                break

        logging.info(
                "iter %d (epoch %d): unpen-loss %f, cost reject %f, do no harm %f, log barr %f, weight_pen %f",
                inner_idx, i, unpen_loss, cost_reject, accept_pen, log_barr, weight_pen)
        logging.info("iter %d (epoch %d): acc unif prob %f", inner_idx, i, accept_unif_proportion)
        logging.info("iter %d (epoch %d): acc is-data prob %f", inner_idx, i, accept_is_data_proportion)
        logging.info("iter %d (epoch %d): penalized-loss %f", inner_idx, i, tot_loss)
        logging.info("iter %d (epoch %d): actual-loss %f", inner_idx, i, actual_loss)
        return tot_loss, accept_is_data_proportion

    def _print_train(self, sess, x_train, y_train):
        """
        Utility function for printing things regarding the model (and training data) during training
        Default will do nothing...
        """
        return

    def _init_network_variables(self, sess, param_force = None):
        """
        Initialize network variables
        """
        param_list = self.model_params.param_list if param_force is None else param_force
        for i, c_val in enumerate(param_list):
            sess.run(self.param_assign_ops[i], feed_dict={self.param_phs[i]: c_val})

    def get_univar_mapping(self, x, max_replicates=100):
        """
        @param x: covariates
        @return accept prob
        """
        sess = tf.Session()
        with sess.as_default():
            # Re-initialize neural network params since we are in a new session
            self._init_network_variables(sess)

            # Predict values
            num_replicates = max_replicates if self.dropout_rate > 0 else 1
            univar_mappings = []
            for i in range(num_replicates):
                univar_mapping = sess.run(
                    self.decision_univar_mapping,
                    feed_dict={
                        self.x_concat_ph: x,
                    })
                univar_mappings.append(univar_mapping)

        sess.close()
        final_univar_mapping = np.mean(np.concatenate(univar_mappings, axis=1), axis=1)
        return final_univar_mapping

    def get_accept_prob(self, x, max_replicates=100):
        """
        @param x: covariates
        @return accept prob
        """
        sess = tf.Session()
        with sess.as_default():
            # Re-initialize neural network params since we are in a new session
            self._init_network_variables(sess)

            # Predict values
            num_replicates = max_replicates if self.dropout_rate > 0 else 1
            all_accept_probs = []
            for i in range(num_replicates):
                accept_prob = sess.run(
                    self.accept_prob,
                    feed_dict={
                        self.x_concat_ph: x,
                    })
                all_accept_probs.append(accept_prob)

        sess.close()
        final_accept_prob = np.mean(np.concatenate(all_accept_probs, axis=1), axis=1, keepdims=True)
        return final_accept_prob

    def get_prediction_interval(self, x, alpha=0.1):
        """
        @param x: covariates
        @param alpha: create PI with width (1 - alpha) coverage
        @return prediction interval
        """
        raise NotImplementedError("You need to implement this!")

    def score(self, x, y, max_replicates=1000):
        """
        Function used by cross validation function in scikit
        Higher the score the better
        """
        sess = tf.Session()
        with sess.as_default():
            # Re-initialize neural network params since we are in a new session
            self._init_network_variables(sess)
            is_data = np.ones((x.shape[0], 1))
            num_replicates = max_replicates if self.dropout_rate > 0 else 1
            all_losses = []
            for i in range(num_replicates):
                loss = sess.run(
                    self.actual_loss,
                    feed_dict={
                        self.x_concat_ph: x,
                        self.y_concat_ph: y,
                        self.is_data_ph: is_data,
                    })
                all_losses.append(loss)

        sess.close()
        final_loss = np.mean(all_losses)
        return -final_loss

    def get_prediction_loss_obs(self, x, y, max_replicates=1000):
        num_replicates = max_replicates if self.dropout_rate > 0 else 1

        sess = tf.Session()
        with sess.as_default():
            # Re-initialize neural network params since we are in a new session
            self._init_network_variables(sess)
            mean_pred_loss_obs = 0
            for i in range(num_replicates):
                pred_loss_obs = sess.run(
                    self.prediction_loss_obs,
                    feed_dict={
                        self.x_concat_ph: x,
                        self.y_concat_ph: y,
                    })
                mean_pred_loss_obs += pred_loss_obs

        sess.close()
        return mean_pred_loss_obs/num_replicates
