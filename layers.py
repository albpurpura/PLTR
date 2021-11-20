import tensorflow as tf


class MultiHeadSelfAttLayer(tf.keras.layers.Layer):
    def __init__(self, n_heads, input_size, hidd_size, level):
        super(MultiHeadSelfAttLayer, self).__init__()
        self.hidd_size = hidd_size
        self.n_heads = n_heads
        self.w_output = tf.get_variable(name='w_output', shape=(hidd_size * n_heads, input_size),
                                        regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                                        # regularizer=tf.keras.regularizers.l2(l2=1.),
                                        dtype=tf.float32, initializer=tf.initializers.glorot_normal(seed=level),
                                        trainable=True)
        self.layernorm0 = tf.keras.layers.LayerNormalization(axis=-1)
        self.layernorm1 = tf.keras.layers.LayerNormalization(axis=-1)
        self.output_layer = Ffnn(hidd_size * n_heads, input_size * 3, input_size, level)
        self.layers = []
        for n in range(n_heads):
            with tf.variable_scope("self_att_layer_%d_%d" % (n, level)):
                # Create sublayers for each layer.
                self_attention_layer = SelfAttentionLayer(input_size, hidd_size, n)
                self.layers.append(self_attention_layer)

    def call(self, x, training):
        att_heads_results = []
        att_weights_results = []
        # multi-head attention
        for n, self_attention_layer in enumerate(self.layers):
            with tf.variable_scope("self_att_layer_%d" % n):
                interaction_weights, layer_out = self_attention_layer(x, training)
                att_heads_results.append(layer_out)
                att_weights_results.append(interaction_weights)
        # concat
        embedded_output = tf.stack(att_heads_results, axis=-1)
        hidd_doc_repr = tf.reshape(embedded_output, (-1, tf.shape(embedded_output)[1], self.hidd_size * self.n_heads))
        # add and norm
        hidd_doc_repr = self.layernorm0(hidd_doc_repr + x)
        hidd_doc_repr = tf.layers.dropout(hidd_doc_repr, rate=0.5, training=training)
        # position-ff
        output = self.output_layer(hidd_doc_repr, training)
        # add and norm
        output = self.layernorm1(output + hidd_doc_repr)
        output = tf.layers.dropout(output, rate=0.5, training=training)
        return tf.stack(att_weights_results, axis=-1), output


class Ffnn(tf.keras.layers.Layer):
    def __init__(self, input_size, w1_hidd_size, w2_hidd_size, seed):
        super(Ffnn, self).__init__()
        # self.bn = tf.keras.layers.BatchNormalization(axis=-1)
        self.w1 = tf.get_variable(name='w1', shape=(input_size, w1_hidd_size),
                                  regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                                  dtype=tf.float32, initializer=tf.initializers.glorot_normal(seed=seed),
                                  trainable=True)
        self.b1 = tf.get_variable(name='b1', shape=w1_hidd_size,
                                  regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                                  dtype=tf.float32, initializer=tf.initializers.glorot_normal(seed=seed),
                                  trainable=True)
        self.w2 = tf.get_variable(name='w2', shape=(w1_hidd_size, w2_hidd_size),
                                  regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                                  dtype=tf.float32, initializer=tf.initializers.glorot_normal(seed=seed),
                                  trainable=True)
        self.b2 = tf.get_variable(name='b2', shape=w2_hidd_size,
                                  # regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                                  regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                                  dtype=tf.float32, initializer=tf.initializers.glorot_normal(seed=seed),
                                  trainable=True)

    def call(self, x, training):
        p1 = tf.nn.leaky_relu(tf.einsum('bse, eo->bso', x, self.w1) + self.b1)
        # print('add dropout in ffnn in between the layers')
        # p1 = tf.layers.dropout(p1, training=training)
        # print('replaced l2 norm in ffnn with bn layer')
        p1 = tf.nn.l2_normalize(p1, axis=-1)
        # p1 = self.bn(p1)
        p2 = tf.einsum('bse, eo->bso', p1, self.w2) + self.b2
        return p2


class SelfAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, input_data_size, proj_space_size, seed):
        super(SelfAttentionLayer, self).__init__()
        self.proj_space_size = proj_space_size
        self.k = tf.get_variable(name='K', shape=(input_data_size, self.proj_space_size),
                                 regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                                 dtype=tf.float32, initializer=tf.initializers.glorot_normal(seed=seed), trainable=True)
        self.q = tf.get_variable(name='Q', shape=(input_data_size, self.proj_space_size),
                                 regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                                 dtype=tf.float32, initializer=tf.initializers.glorot_normal(seed=seed), trainable=True)
        self.v = tf.get_variable(name='V', shape=(input_data_size, self.proj_space_size),
                                 regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                                 dtype=tf.float32, initializer=tf.initializers.glorot_normal(seed=seed), trainable=True)

    def call(self, embdedded_features_vectors, training):
        Q = tf.einsum('eo, bse->bso', self.q, embdedded_features_vectors)
        K = tf.einsum('eo, bse->bso', self.k, embdedded_features_vectors)
        V = tf.einsum('eo, bse->bso', self.v, embdedded_features_vectors)

        QK = tf.matmul(Q, K, transpose_b=True)
        QK = QK / tf.sqrt(tf.cast(self.proj_space_size, tf.float32))
        interaction_weights = tf.reduce_sum(QK, axis=-1)
        att_w = tf.nn.softmax(interaction_weights, axis=-1)
        output = tf.layers.dropout(tf.einsum('bso,bs->bso', V, att_w), rate=0.5, training=training)
        output = tf.nn.l2_normalize(output)
        return att_w, output


class FFNetCombo(tf.keras.layers.Layer):
    def __init__(self, input_size, output_size, seed, rate=0.5):
        super(FFNetCombo, self).__init__()
        self.proj_matrix = tf.get_variable(name='W_ffncombo', shape=(input_size, output_size), dtype=tf.float32,
                                           initializer=tf.initializers.glorot_normal(seed=seed), trainable=True)
        self.bias = tf.get_variable(name='b_ffncombo', shape=output_size, dtype=tf.float32,
                                    initializer=tf.initializers.glorot_normal(seed=seed), trainable=True)
        self.dropout = tf.keras.layers.Dropout(rate)
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.4, axis=-1)
        # self.hidd_l = tfp.layers.DenseFlipout(1, activation=tf.nn.leaky_relu)

    def call(self, inputs, **kwargs):
        norm_inputs = self.bn(inputs)
        output = tf.nn.leaky_relu(tf.einsum('bsf, fo->bso', norm_inputs, self.proj_matrix) + self.bias)
        # output = self.hidd_l(norm_inputs)
        output = self.dropout(output)
        return output


class FCReluBN(tf.keras.layers.Layer):
    def __init__(self, input_size, output_size, seed, rate=0.5):
        super(FCReluBN, self).__init__()
        self.proj_matrix = tf.get_variable(name='W_ffncombo', shape=(input_size, output_size), dtype=tf.float32,
                                           initializer=tf.initializers.glorot_normal(seed=seed), trainable=True)
        self.bias = tf.get_variable(name='b_ffncombo', shape=output_size, dtype=tf.float32,
                                    initializer=tf.initializers.glorot_normal(seed=seed), trainable=True)
        self.dropout = tf.keras.layers.Dropout(rate)
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.4, axis=-1)
        # self.hidd_l = tfp.layers.DenseFlipout(1, activation=tf.nn.leaky_relu)

    def call(self, inputs, **kwargs):
        output = tf.nn.leaky_relu(tf.einsum('bsf, fo->bso', inputs, self.proj_matrix) + self.bias)
        output = self.bn(output)
        return output
