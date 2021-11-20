import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import losses
from layers import MultiHeadSelfAttLayer


class ReRanker:
    def __init__(self, seed, learning_rate, n_heads, num_features, loss_fn, list_size, det_model=False,
                 max_label_value=4, n=32, consider_raw_rj_dists=False, use_softmax=False):
        tf.set_random_seed(seed)

        self.global_step = tf.Variable(0, trainable=False)
        self.training = tf.placeholder(tf.bool, None)
        self.list_size_train = list_size
        out_size = 1
        if consider_raw_rj_dists:
            out_size = 3
            # out_size = 2
            self.rj = tf.placeholder(tf.float32, (None, list_size, out_size), name='relevance_judgments')
        else:
            self.rj = tf.placeholder(tf.float32, (None, list_size), name='relevance_judgments')
        self.rl_lengths_mask = tf.placeholder(tf.float32, (None, None), name='rl_lengths_mask')
        self.n_heads = n_heads
        self.n = n
        self.hidden_size = int(num_features / self.n_heads)
        self.input_docs = tf.placeholder(tf.float32, (None, list_size, num_features), name='raw_features')

        with tf.variable_scope('reranker'):
            self.multi_head_satt = MultiHeadSelfAttLayer(self.n_heads, self.input_docs.shape[-1], self.hidden_size, 0)
            self.bn0 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.4)
            hidd_size = 8  # 8 is the best
            print('hidd size: {}'.format(hidd_size))
            self.hidden_netw0 = tf.keras.layers.Dense(hidd_size, activation=tf.nn.leaky_relu)  # 32
            self.bn1 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.4)
            print('DET output layer')
            self.output_layer = tf.keras.layers.Dense(out_size, activation=None)


        ###########################################
        # self attention layer
        self.att_weights, self.hidd_doc_repr = self.multi_head_satt(self.input_docs, self.training)
        self.hidd_doc_repr = tf.concat((self.hidd_doc_repr, self.input_docs), axis=-1)
        self.hidd_doc_repr = self.bn0(self.hidd_doc_repr)
        self.hidd_doc_repr = self.hidden_netw0(self.hidd_doc_repr)
        self.hidd_doc_repr = self.bn1(self.hidd_doc_repr)
        self.regr_loss = tf.convert_to_tensor(0.0)
        coeff_regr_loss = 1  # 5e3  # 0.02  # 50 is better than 100
        print('coeff_regr_loss (with softmax on logit probs): {}'.format(coeff_regr_loss))
        if consider_raw_rj_dists:
            self.logits = self.output_layer(self.hidd_doc_repr)
            self.logits = tf.nn.softmax(self.logits, axis=-1)

            self.relevance_judgments = tf.einsum('bst, bs->bst', self.rj, self.rl_lengths_mask)
            self.logits = tf.einsum('bst, bs->bst', self.logits, self.rl_lengths_mask)

            self.aggr_logits = tf.einsum('t,bst->bs', tf.convert_to_tensor(np.array([-1, 0.5, 1]), dtype=tf.float32),
                                         self.logits)
            self.aggr_logits = (tf.ones_like(self.aggr_logits) + self.aggr_logits) / 2
            self.aggr_logits = tf.multiply(self.aggr_logits, self.rl_lengths_mask)

            self.aggr_relevance_judgments = (tf.einsum('t,bst->bs', tf.convert_to_tensor(np.array([-1, 0.5, 1]),
                                                                                         dtype=tf.float32),
                                                       self.rj) + 1) / 2
            self.aggr_relevance_judgments = tf.multiply(self.aggr_relevance_judgments, self.rl_lengths_mask)

            if use_softmax:
                print('using softmax')
                self.softmax_mask = tf.cast(tf.less_equal(self.rl_lengths_mask, 1e-6), tf.float32) * (-1e9)
                self.aggr_relevance_judgments += self.softmax_mask
                self.aggr_relevance_judgments = tf.nn.softmax(self.aggr_relevance_judgments, axis=-1)
                self.aggr_logits += self.softmax_mask
                self.aggr_logits = tf.nn.softmax(self.aggr_logits, axis=-1)
                self.aggr_logits = tf.einsum('bs, bs->bs', self.rl_lengths_mask, self.aggr_logits)
        else:
            print('standard no distrib fitting')
            self.logits = tf.squeeze(self.output_layer(self.hidd_doc_repr), axis=-1)
            self.logits = tf.multiply(self.logits, self.rl_lengths_mask)

            if use_softmax:
                print('using softmax')
                self.softmax_mask = tf.cast(tf.less_equal(self.rl_lengths_mask, 1e-6), tf.float32) * (-1e9)
                self.logits += self.softmax_mask
                self.logits = tf.nn.softmax(self.logits, axis=-1)
                self.relevance_judgments = self.rj + self.softmax_mask
                self.relevance_judgments = tf.nn.softmax(self.relevance_judgments, axis=-1)
                self.logits = tf.einsum('bs, bs->bs', self.rl_lengths_mask, self.logits)

        ###########################################
        reg_coeff = 0.0001
        reg_kl_coeff = 0.00005
        if loss_fn == 'KL_G' and not consider_raw_rj_dists:
            reg_kl_coeff = 0.00005  # 0.000002
            self.logits = tf.sigmoid(self.logits)
            self.logits = tf.multiply(self.logits, self.rl_lengths_mask)
            self.relevance_judgments = tf.multiply(self.rj, self.rl_lengths_mask)
            self.ranking_loss = self.compute_kl_div_loss(self.logits, self.relevance_judgments)
        elif loss_fn == 'KL_G' and consider_raw_rj_dists:
            reg_kl_coeff = 0.00005  # 0.000002
            # self.ranking_loss = self.compute_kl_div_loss(self.aggr_logits, self.aggr_relevance_judgments)
            self.ranking_loss_unred = self.compute_kl_multivariate_gaussian_simm(self.aggr_logits,
                                                                                 self.aggr_relevance_judgments)
            self.ranking_loss = tf.multiply(self.ranking_loss_unred, self.rl_lengths_mask)
            thr = 0.2
            mask_n_rel = tf.cast(tf.less_equal(thr, self.aggr_relevance_judgments), tf.float32) * self.rl_lengths_mask
            mask_rel = tf.cast(tf.greater(thr, self.aggr_relevance_judgments), tf.float32) * self.rl_lengths_mask
            weight_non_rel = (1 + 1e-6) / (1e-6 + tf.reduce_sum(mask_n_rel, axis=-1))
            weight_rel = (1 + 1e-6) / (1e-6 + tf.reduce_sum(mask_rel, axis=-1))
            self.wnr = tf.multiply(mask_n_rel,
                                   tf.einsum('bs, b->bs', tf.ones_like(self.ranking_loss_unred), weight_non_rel))
            self.wr = tf.multiply(mask_rel, tf.einsum('bs, b->bs', tf.ones_like(self.ranking_loss_unred), weight_rel))
            self.w = self.wnr + self.wr
            self.ranking_loss_unred = tf.multiply(self.w, self.ranking_loss_unred)
            self.ranking_loss = tf.reduce_mean(
                tf.reduce_sum(self.ranking_loss_unred, axis=-1))
        elif loss_fn == 'KL_G_H' and consider_raw_rj_dists:
            reg_kl_coeff = 0.000002
            self.aggr_relevance_judgments = tf.multiply(self.aggr_relevance_judgments, self.rl_lengths_mask)
            self.aggr_relevance_judgments = self.aggr_relevance_judgments + tf.cast(
                tf.less_equal(self.rl_lengths_mask, 0), tf.float32) * (-1)
            pairs, weights = losses.compute_pairwise_kl_g_loss(self.aggr_logits, self.aggr_relevance_judgments)
            self.ranking_loss = tf.reduce_mean(tf.multiply(pairs, weights))
        elif loss_fn == 'KL_B_H' and consider_raw_rj_dists:
            reg_kl_coeff = 0.000002
            self.aggr_relevance_judgments = tf.multiply(self.aggr_relevance_judgments, self.rl_lengths_mask)
            self.aggr_relevance_judgments = self.aggr_relevance_judgments + tf.cast(
                tf.less_equal(self.rl_lengths_mask, 0), tf.float32) * (-1)
            pairs, weights = losses.compute_pairwise_kl_bin_loss(self.aggr_logits, self.aggr_relevance_judgments)
            self.ranking_loss = tf.reduce_mean(tf.multiply(pairs, weights))
        elif loss_fn == 'KL_B' and consider_raw_rj_dists:
            reg_kl_coeff = 0.00005  # 0.000002
            self.ranking_loss_unred = compute_kl_div_loss_bin_simm(self.aggr_logits, self.aggr_relevance_judgments)
            self.ranking_loss_unred = tf.multiply(self.ranking_loss_unred, self.rl_lengths_mask)

            thr = 0.2
            mask_n_rel = tf.cast(tf.less_equal(thr, self.aggr_relevance_judgments), tf.float32) * self.rl_lengths_mask
            mask_rel = tf.cast(tf.greater(thr, self.aggr_relevance_judgments), tf.float32) * self.rl_lengths_mask
            weight_non_rel = (1 + 1e-6) / (1e-6 + tf.reduce_sum(mask_n_rel, axis=-1))
            weight_rel = (1 + 1e-6) / (1e-6 + tf.reduce_sum(mask_rel, axis=-1))
            self.wnr = tf.multiply(mask_n_rel,
                                   tf.einsum('bs, b->bs', tf.ones_like(self.ranking_loss_unred), weight_non_rel))
            self.wr = tf.multiply(mask_rel, tf.einsum('bs, b->bs', tf.ones_like(self.ranking_loss_unred), weight_rel))
            self.w = self.wnr + self.wr
            self.ranking_loss_unred = tf.multiply(self.w, self.ranking_loss_unred)

            self.ranking_loss = tf.reduce_mean(
                tf.reduce_sum(self.ranking_loss_unred, axis=-1))
        ###########################################

        reg_losses = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        # reg_coeff = 0.0001
        # reg_kl_coeff = 0.000002  # 0.000002

        print('reg term coeff: {}'.format(reg_coeff))

        self.loss = self.ranking_loss + reg_losses * reg_coeff
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        # qdiff_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-2)
        # optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
        update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)

        train_op = optimizer.minimize(self.loss, global_step=self.global_step)
        self.init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                                tf.compat.v1.local_variables_initializer())
        self.train_op = tf.group([train_op, update_ops])
        # self.train_op = tf.group([qdiff_train_op, rerank_train_op, update_ops])
        self.saver = tf.train.Saver(max_to_keep=None)

    @staticmethod
    def compute_reg_term_for_att_matrices(attention_coeffs):
        reg_values = []
        att_vs = tf.unstack(attention_coeffs, axis=-1)
        for i in range(attention_coeffs.shape[-1]):
            for j in range(i + 1, attention_coeffs.shape[-1]):
                reg_term = tf.reduce_mean(tf.abs(tf.subtract(att_vs[i], att_vs[j])), axis=-1)
                reg_values.append(reg_term)
        reg_values = tf.stack(reg_values)
        return tf.reduce_mean(reg_values)

    def compute_kl_multivariate_gaussian(self, logits, labels):
        res = 0.5 * ((logits - labels) * 1 / tf.ones(self.list_size_train) * (logits - labels))
        return res

    def compute_kl_multivariate_gaussian_simm(self, logits, labels):
        return self.compute_kl_multivariate_gaussian(logits, labels) + self.compute_kl_multivariate_gaussian(labels,
                                                                                                             logits)

    def compute_kl_div_loss(self, logits, labels):
        d1 = tfp.distributions.MultivariateNormalDiag(loc=logits, scale_diag=tf.ones(
            self.list_size_train) * 0.5)
        d2 = tfp.distributions.MultivariateNormalDiag(loc=labels, scale_diag=tf.ones(
            self.list_size_train) * 0.5)
        return tfp.distributions.kl_divergence(d1, d2) + tfp.distributions.kl_divergence(d2, d1)


def compute_kl_div_loss_bin_simm(logits, labels, n=6):
    return compute_kl_div_loss_bin(logits, labels, n) + compute_kl_div_loss_bin(labels, logits, n)


def compute_kl_div_multinomial(logits, labels):
    return tf.reduce_sum(logits * tf.log((logits + 1e-6) / (labels + 1e-6)), axis=-1)


def compute_kl_div_loss_bin(logits, labels, n=6):
    loss = tf.log((1e-6 + labels) / (1e-6 + logits)) * n * labels + tf.log(
        (1e-6 + 1 - labels) / (1e-6 + 1 - logits)) * n * (1 - labels)
    return loss  # tf.reduce_mean(loss, axis=-1)


def smooth_max(sequence, gamma=10, axis=-1):
    return tf.log(tf.reduce_sum(tf.exp(gamma * (sequence + 1e-6)), axis=axis)) / (gamma + 1e-6)
