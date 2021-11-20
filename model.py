import tensorflow as tf
import tensorflow_probability as tfp

import losses
from layers import MultiHeadSelfAttLayer


class ReRanker:
    def __init__(self, seed, learning_rate, n_heads, num_features, loss_fn, list_size, coll_name, det_model=False,
                 norm_labels=False, max_label_value=4, n=32, consider_raw_rj_dists=False):
        tf.set_random_seed(seed)

        self.global_step = tf.Variable(0, trainable=False)
        self.training = tf.placeholder(tf.bool, None)
        self.list_size_train = list_size
        out_size = 1
        self.relevance_judgments = tf.placeholder(tf.float32, (None, list_size), name='relevance_judgments')
        self.rl_lengths_mask = tf.placeholder(tf.float32, (None, list_size), name='rl_lengths_mask')
        self.n_heads = n_heads
        self.n = n
        self.hidden_size = int(num_features / self.n_heads)
        self.input_docs = tf.placeholder(tf.float32, (None, list_size, num_features), name='raw_features')

        with tf.variable_scope('reranker'):
            self.multi_head_satt = MultiHeadSelfAttLayer(self.n_heads, self.input_docs.shape[-1], self.hidden_size, 0)
            self.bn0 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.4)
            hidd_size = 32
            if coll_name.startswith('MSLR-WEB'):
                hidd_size = 128
            # self.hidden_netw0 = tf.keras.layers.Dense(32, activation=tf.nn.leaky_relu)
            self.hidden_netw0 = tf.keras.layers.Dense(hidd_size, activation=tf.nn.leaky_relu)
            self.bn1 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.4)

            self.output_layer = tf.keras.layers.Dense(out_size, activation=None)

        ###########################################
        # self attention layer
        self.att_weights, self.hidd_doc_repr = self.multi_head_satt(self.input_docs, self.training)
        self.hidd_doc_repr = tf.concat((self.hidd_doc_repr, self.input_docs), axis=-1)
        self.hidd_doc_repr = self.bn0(self.hidd_doc_repr)
        self.hidd_doc_repr = self.hidden_netw0(self.hidd_doc_repr)
        self.hidd_doc_repr = self.bn1(self.hidd_doc_repr)

        self.logits = tf.squeeze(self.output_layer(self.hidd_doc_repr), axis=-1)
        self.logits = tf.sigmoid(self.logits)
        self.logits = tf.einsum('bs, bs->bs', self.rl_lengths_mask, self.logits)
        self.relevance_judgments = tf.multiply(self.relevance_judgments, self.rl_lengths_mask)
        ###########################################
        if loss_fn == 'KL_G_H':
            self.relevance_judgments = self.relevance_judgments + tf.cast(
                tf.less_equal(self.rl_lengths_mask, 0), tf.float32) * (-1)
            self.pairs, self.weights = losses.compute_pairwise_kl_g_loss(self.logits, self.relevance_judgments)
            self.ranking_loss = tf.reduce_mean(tf.multiply(self.pairs, self.weights))
        elif loss_fn == 'KL_B_H':
            self.relevance_judgments = self.relevance_judgments + tf.cast(
                tf.less_equal(self.rl_lengths_mask, 0), tf.float32) * (-1)
            self.pairs, self.weights = losses.compute_pairwise_kl_bin_loss(self.logits, self.relevance_judgments)
            self.ranking_loss = tf.reduce_mean(tf.multiply(self.pairs, self.weights))
        elif loss_fn == 'KL_G':
            # KL div loss
            # self.ranking_loss = self.compute_kl_multivariate_gaussian(self.logits, self.relevance_judgments) + self.compute_kl_multivariate_gaussian(self.relevance_judgments, self.logits)
            # self.ranking_loss = tf.reduce_mean(tf.reduce_sum(self.ranking_loss, axis=-1) / tf.reduce_sum(self.rl_lengths_mask, axis=-1))
            # print('tf prob klg loss')
            self.ranking_loss_unred = self.compute_kl_multivariate_gaussian_simm(self.logits, self.relevance_judgments)
            self.ranking_loss_unred = tf.multiply(self.ranking_loss_unred, self.rl_lengths_mask)

            thr = 0.2
            mask_n_rel = tf.cast(tf.less_equal(thr, self.relevance_judgments), tf.float32) * self.rl_lengths_mask
            mask_rel = tf.cast(tf.greater(thr, self.relevance_judgments), tf.float32) * self.rl_lengths_mask
            weight_non_rel = (1 + 1e-6) / (1e-6 + tf.reduce_sum(mask_n_rel, axis=-1))
            weight_rel = (1 + 1e-6) / (1e-6 + tf.reduce_sum(mask_rel, axis=-1))
            self.wnr = tf.multiply(mask_n_rel,
                                   tf.einsum('bs, b->bs', tf.ones_like(self.ranking_loss_unred), weight_non_rel))
            self.wr = tf.multiply(mask_rel, tf.einsum('bs, b->bs', tf.ones_like(self.ranking_loss_unred), weight_rel))
            self.w = self.wnr + self.wr
            self.ranking_loss_unred = tf.multiply(self.w, self.ranking_loss_unred)

            self.ranking_loss = tf.reduce_mean(
                tf.reduce_sum(self.ranking_loss_unred, axis=-1) / tf.reduce_sum(self.rl_lengths_mask, axis=-1))
        elif loss_fn == 'KL_B':
            # KL div loss
            # self.ranking_loss = tf.reduce_mean(
            #     compute_kl_div_loss_bin_simm(self.logits, self.relevance_judgments, self.n), axis=-1)
            self.ranking_loss_unred = compute_kl_div_loss_bin_simm(self.logits, self.relevance_judgments, self.n)

            self.ranking_loss_unred = tf.multiply(self.ranking_loss_unred, self.rl_lengths_mask)

            thr = 0.2
            mask_n_rel = tf.cast(tf.less_equal(thr, self.relevance_judgments), tf.float32) * self.rl_lengths_mask
            mask_rel = tf.cast(tf.greater(thr, self.relevance_judgments), tf.float32) * self.rl_lengths_mask
            weight_non_rel = (1 + 1e-6) / (1e-6 + tf.reduce_sum(mask_n_rel, axis=-1))
            weight_rel = (1 + 1e-6) / (1e-6 + tf.reduce_sum(mask_rel, axis=-1))
            self.wnr = tf.multiply(mask_n_rel,
                                   tf.einsum('bs, b->bs', tf.ones_like(self.ranking_loss_unred), weight_non_rel))
            self.wr = tf.multiply(mask_rel, tf.einsum('bs, b->bs', tf.ones_like(self.ranking_loss_unred), weight_rel))
            self.w = self.wnr + self.wr
            self.ranking_loss_unred = tf.multiply(self.w, self.ranking_loss_unred)

            self.ranking_loss = tf.reduce_mean(
                tf.reduce_sum(self.ranking_loss_unred, axis=-1) / tf.reduce_sum(self.rl_lengths_mask, axis=-1))

        reg_losses = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        reg_coeff = 0.0001
        print('reg term coeff: {}'.format(reg_coeff))

        self.loss = tf.reduce_mean(self.ranking_loss) + reg_losses * reg_coeff
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = optimizer.minimize(self.loss, global_step=self.global_step)
        self.init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                                tf.compat.v1.local_variables_initializer())
        self.train_op = tf.group([train_op, update_ops])
        # self.train_op = tf.group([qdiff_train_op, rerank_train_op, update_ops])
        self.saver = tf.train.Saver(max_to_keep=None)

    def compute_kl_multivariate_gaussian(self, logits, labels):
        res = 0.5 * ((logits - labels) * (1 / tf.ones(self.list_size_train) * 0.25) * (logits - labels))
        return res

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

    def compute_kl_multivariate_gaussian_simm(self, logits, labels):
        return self.compute_kl_multivariate_gaussian(logits, labels) + self.compute_kl_multivariate_gaussian(labels,
                                                                                                             logits)

    def compute_kl_div_loss(self, logits, labels, normalize_labels=False, max_label_value=4,
                            consider_raw_rj_dists=False):
        if not consider_raw_rj_dists:
            if normalize_labels:
                d1 = tfp.distributions.MultivariateNormalDiag(loc=logits,
                                                              scale_diag=tf.ones(
                                                                  shape=logits.shape[1]) / max_label_value)
                d2 = tfp.distributions.MultivariateNormalDiag(loc=labels,
                                                              scale_diag=tf.ones(
                                                                  shape=labels.shape[1]) / max_label_value)
            else:
                d1 = tfp.distributions.MultivariateNormalDiag(loc=logits, scale_diag=tf.ones(
                    self.list_size_train) / max_label_value)
                d2 = tfp.distributions.MultivariateNormalDiag(loc=labels, scale_diag=tf.ones(
                    self.list_size_train) / max_label_value)
                # d2 = tfp.distributions.MultivariateNormalDiag(loc=labels, scale_diag=tf.ones(self.labels.shape[1]))
        else:
            d1 = tfp.distributions.Normal(loc=logits, scale=tf.nn.softmax(tf.ones_like(logits) / max_label_value))
            d2 = tfp.distributions.Normal(loc=labels, scale=tf.nn.softmax(tf.ones_like(logits) / max_label_value))

        return tfp.distributions.kl_divergence(d1, d2) + tfp.distributions.kl_divergence(d2, d1)


def compute_kl_div_loss_bin_simm(logits, labels, n=6):
    return compute_kl_div_loss_bin(logits, labels, n) + compute_kl_div_loss_bin(labels, logits, n)


def compute_kl_div_loss_bin(logits, labels, n=6):
    loss = tf.log((1e-6 + labels) / (1e-6 + logits)) * n * labels + tf.log(
        (1e-6 + 1 - labels) / (1e-6 + 1 - logits)) * n * (1 - labels)
    return loss


def smooth_max(sequence, gamma=10, axis=-1):
    return tf.log(tf.reduce_sum(tf.exp(gamma * (sequence + 1e-6)), axis=axis)) / (gamma + 1e-6)
