import tensorflow as tf

import utils

epsilon = 1e-8


def log2(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator


def compute_pairwise_kl_bin_loss(logits, labels):
    pairwise_labels, pairwise_logits = _pairwise_kl_bin_loss(labels, logits)
    pairwise_weights = pairwise_labels
    pairwise_weights = tf.stop_gradient(
        pairwise_weights, name='weights_stop_gradient')
    # return pairwise_logits, pairwise_weights
    return tf.nn.relu(1 - pairwise_logits), pairwise_weights


def compute_pairwise_kl_g_loss(logits, labels):
    pairwise_labels, pairwise_logits = _pairwise_kl_g_loss(labels, logits)
    pairwise_weights = pairwise_labels
    pairwise_weights = tf.stop_gradient(
        pairwise_weights, name='weights_stop_gradient')
    return tf.nn.relu(1 - pairwise_logits), pairwise_weights
    # return pairwise_logits, pairwise_weights


def compute_pairwise_kl_loss(logits, labels):
    pairwise_labels, pairwise_logits = _pairwise_kl(labels, logits)
    pairwise_weights = pairwise_labels
    pairwise_weights = tf.stop_gradient(
        pairwise_weights, name='weights_stop_gradient')
    # return pairwise_logits, pairwise_weights
    return tf.nn.relu(1 - pairwise_logits), pairwise_weights


def _pairwise_kl(labels, logits):
    pairwise_label_diff = _apply_pairwise_op(tf.subtract, labels)
    pairwise_logits = _apply_pairwise_op(simm_kl_multinomial, logits)
    pairwise_logits_diff = _apply_pairwise_op(tf.subtract, logits)
    print('added diff sign')
    pairwise_logits = tf.multiply(tf.sign(pairwise_logits_diff), pairwise_logits)
    # print('pw logits:')
    # print(pairwise_logits.shape)
    # Only keep the case when l_i > l_j.
    pairwise_labels = tf.cast(
        tf.greater(pairwise_label_diff, 0), dtype=tf.float32)
    is_valid = utils.is_label_valid(labels)
    valid_pair = _apply_pairwise_op(tf.logical_and, is_valid)
    pairwise_labels *= tf.cast(valid_pair, dtype=tf.float32)
    return pairwise_labels, pairwise_logits


def _pairwise_kl_g_loss(labels, logits):
    pairwise_label_diff = _apply_pairwise_op(tf.subtract, labels)
    pairwise_logits = _apply_pairwise_op(simm_kl_g, logits)
    pairwise_logits_diff = _apply_pairwise_op(tf.subtract, logits)
    print('added diff sign')
    pairwise_logits = tf.multiply(tf.sign(pairwise_logits_diff), pairwise_logits)
    # print('pw logits:')
    # print(pairwise_logits.shape)
    # Only keep the case when l_i > l_j.
    pairwise_labels = tf.cast(
        tf.greater(pairwise_label_diff, 0), dtype=tf.float32)
    is_valid = utils.is_label_valid(labels)
    valid_pair = _apply_pairwise_op(tf.logical_and, is_valid)
    pairwise_labels *= tf.cast(valid_pair, dtype=tf.float32)
    return pairwise_labels, pairwise_logits


def _pairwise_kl_bin_loss(labels, logits):
    pairwise_label_diff = _apply_pairwise_op(tf.subtract, labels)
    pairwise_logits = _apply_pairwise_op(simm_kl_bin, logits)
    pairwise_logits_diff = _apply_pairwise_op(tf.subtract, logits)
    print('added diff sign')
    pairwise_logits = tf.multiply(tf.sign(pairwise_logits_diff), pairwise_logits)
    # print('pw logits:')
    # print(pairwise_logits.shape)
    # Only keep the case when l_i > l_j.
    pairwise_labels = tf.cast(
        tf.greater(pairwise_label_diff, 0), dtype=tf.float32)
    is_valid = utils.is_label_valid(labels)
    valid_pair = _apply_pairwise_op(tf.logical_and, is_valid)
    pairwise_labels *= tf.cast(valid_pair, dtype=tf.float32)
    return pairwise_labels, pairwise_logits


def _compute_ranks(logits, is_valid):
    """Computes ranks by sorting valid logits.
    Args:
      logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
        ranking score of the corresponding item.
      is_valid: A `Tensor` of the same shape as `logits` representing validity of
        each entry.
    Returns:
      The `ranks` Tensor.
    """
    _check_tensor_shapes([logits, is_valid])
    # Only sort entries with is_valid = True.
    scores = tf.compat.v1.where(
        is_valid, logits, - 1e-6 * tf.ones_like(logits) +
                          tf.reduce_min(input_tensor=logits, axis=1, keepdims=True))
    return utils.sorted_ranks(scores)


def _check_tensor_shapes(tensors):
    """Checks the tensor shapes to be compatible."""
    for tensor in tensors:
        tensor = tf.convert_to_tensor(value=tensor)
        tensor.get_shape().assert_has_rank(2)
        tensor.get_shape().assert_is_compatible_with(
            tf.convert_to_tensor(value=tensors[0]).get_shape())


def _pairwise_comparison(labels, logits):
    r"""Returns pairwise comparison `Tensor`s.
    Given a list of n items, the labels of graded relevance l_i and the logits
    s_i, we form n^2 pairs. For each pair, we have the following:
                          /
                          | 1   if l_i > l_j for valid l_i and l_j.
    * `pairwise_labels` = |
                          | 0   otherwise
                          \
    * `pairwise_logits` = s_i - s_j
    Args:
      labels: A `Tensor` with shape [batch_size, list_size].
      logits: A `Tensor` with shape [batch_size, list_size].
    Returns:
      A tuple of (pairwise_labels, pairwise_logits) with each having the shape
      [batch_size, list_size, list_size].
    """
    # Compute the difference for all pairs in a list. The output is a Tensor with
    # shape [batch_size, list_size, list_size] where the entry [-1, i, j] stores
    # the information for pair (i, j).
    pairwise_label_diff = _apply_pairwise_op(tf.subtract, labels)
    pairwise_logits = _apply_pairwise_op(tf.subtract, logits)
    # Only keep the case when l_i > l_j.
    pairwise_labels = tf.cast(
        tf.greater(pairwise_label_diff, 0), dtype=tf.float32)
    is_valid = utils.is_label_valid(labels)
    valid_pair = _apply_pairwise_op(tf.logical_and, is_valid)
    pairwise_labels *= tf.cast(valid_pair, dtype=tf.float32)
    return pairwise_labels, pairwise_logits


def _apply_pairwise_op_ml(op, tensor):
    """Applies the op on tensor in the pairwise manner."""
    # _check_tensor_shapes([tensor])
    rval = op(tensor, tensor)
    return rval
    # return op(tf.expand_dims(tensor, 2), tf.expand_dims(tensor, 1))


def _apply_pairwise_op(op, tensor):
    """Applies the op on tensor in the pairwise manner."""
    # _check_tensor_shapes([tensor])
    return op(tf.expand_dims(tensor, 2), tf.expand_dims(tensor, 1))


def simm_kl_g(x, y):
    return kl_div_gaussian(x, y) + kl_div_gaussian(y, x)


def kl_div_gaussian(x, y):
    std_x = 0.25 * tf.ones_like(x)
    std_y = 0.25 * tf.ones_like(y)
    return 0.5 * tf.math.log(std_y / std_x) + (std_x ** 2 + (x - y) ** 2) / (2 * std_y ** 2) - 0.5


def simm_kl_bin(x, y):
    return compute_kl_div_loss_bin(x, y) + compute_kl_div_loss_bin(y, x)


def simm_kl_multinomial(x, y):
    return compute_kl_div_multinomial(x, y) + compute_kl_div_multinomial(y, x)


def compute_kl_div_multinomial(x, y):
    return tf.reduce_sum(x * tf.math.log((x + 1e-6) / (y + 1e-6)), axis=-1)


def compute_kl_div_loss_bin(logits, labels, n=32):
    loss = tf.log((1e-6 + labels) / (1e-6 + logits)) * n * labels + tf.log(
        (1e-6 + 1 - labels) / (1e-6 + 1 - logits)) * n * (1 - labels)
    return loss  # tf.reduce_mean(loss, axis=-1)


def simm_kl_div_bern(x, y):
    return kl_div_bern(x, y) + kl_div_bern(y, x)


def kl_div_bern(x, y):
    n = 300
    return tf.log((1e-6 + y) / (1e-6 + x)) * n * y + tf.log(
        (1e-6 + 1 - y) / (1e-6 + 1 - x)) * n * (1 - y)


def compute_approxNDCG_gumbel(logits, labels):
    # alpha = self._params.get('alpha', 10.0)
    alpha = 10.0
    # print('alpha from 10 to 0.1')
    # alpha = 0.1
    # the higher the alpha the more the sigmoid approximating the indicator function is steep in the nDCG approx.
    is_valid = utils.is_label_valid(labels)
    labels = tf.compat.v1.where(is_valid, labels, tf.zeros_like(labels))
    logits = tf.compat.v1.where(
        is_valid, logits, -1e3 * tf.ones_like(logits) +
                          tf.reduce_min(input_tensor=logits, axis=-1, keepdims=True))

    label_sum = tf.reduce_sum(input_tensor=labels, axis=1, keepdims=True)
    nonzero_mask = tf.greater(tf.reshape(label_sum, [-1]), 0.0)
    labels = tf.compat.v1.where(nonzero_mask, labels,
                                1e-10 * tf.ones_like(labels))
    gains = tf.pow(2., tf.cast(labels, dtype=tf.float32)) - 1.
    n_samples = 8
    _, sampled_logits = gumbel_neural_sort(logits, sample_size=n_samples)
    sampled_logits = tf.reshape(sampled_logits, (n_samples, -1, labels.shape[-1]))
    ranks = tf.map_fn(lambda l: utils.approx_ranks(l, alpha=alpha), elems=sampled_logits, dtype=tf.float32)
    discounts = 1. / tf.math.log1p(tf.cast(ranks, tf.float32))
    cost = tf.map_fn(lambda d: -tf.reduce_sum(input_tensor=gains * d, axis=-1, keepdims=True) * tf.expand_dims(
        utils.inverse_max_dcg(labels), axis=-2), elems=discounts, dtype=tf.float32)
    # discounts = 1. / tf.math.log1p(ranks)
    # dcg = tf.reduce_sum(input_tensor=gains * discounts, axis=-1, keepdims=True)
    # cost = -dcg * utils.inverse_max_dcg(labels)
    return tf.reduce_mean(cost)


def gumbel_neural_sort(logits,
                       name=None,
                       sample_size=8,
                       temperature=1.0,
                       seed=None):
    """Generate the permutation matrix from logits by stochastic neuralsort.
    By sampling logits from the Gumbel distribution,
      sampled_logits = logits + Gumbel(0, 1),
    the determinstic neural sort z of sampled_logits obeys the distribution with
      Prob(z|logits) = (exp(logit_z1) / Z) * (exp(logit_z2) / Z-exp(logit_z1)) *
                       ... * (exp(logit_zn) / Z-sum_i^(n-1)exp(logit_zi)),
    where Z = sum_i exp(logit_i).
    Args:
      logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
        ranking score of the corresponding item.
      name: A string used as the name for this loss.
      sample_size: An integer representing the number of samples drawn from the
        Concrete distribution defined by scores.
      temperature: The Gumbel-Softmax temperature.
      seed: Seed for pseudo-random number generator.
    Returns:
      A `Tensor` of permutation matrices whose dimension is [batch_size,
      sample_size, list_size, list_size].
    """
    with tf.compat.v1.name_scope(name, 'gumbel_neural_sort', [logits]):
        batch_size = tf.shape(input=logits)[0]
        list_size = tf.shape(input=logits)[1]

        # Sample logits from Concrete(logits).
        sampled_logits = tf.expand_dims(logits, 1)
        sampled_logits += _sample_gumbel([batch_size, sample_size, list_size],
                                         seed=seed)
        sampled_logits = tf.reshape(sampled_logits,
                                    [batch_size * sample_size, list_size])

        # Sort by constructing the relaxed permuation matrix from sampled logits.
        smooth_perm = neural_sort(sampled_logits, name, temperature)
        smooth_perm = tf.reshape(smooth_perm,
                                 [batch_size, sample_size, list_size, list_size])

        return smooth_perm, sampled_logits


def _sample_gumbel(shape, eps=1e-20, seed=None):
    u = tf.random.uniform(shape, minval=0, maxval=1, dtype=tf.float32, seed=seed)
    return -tf.math.log(-tf.math.log(u + eps) + eps)


def neural_sort(logits, name=None, temperature=1.0):
    r"""Generate the permutation matrix from logits by deterministic neuralsort.
    The sort on a list of logits can be approximated by a differentiable
    permutation matrix using Neural Sort (https://arxiv.org/abs/1903.08850).
    The approximation is achieved by constructing a list of functions on logits,
      fn_i(k) = (list_size + 1 - 2*i) * logit_k - sum_j |logit_k - logit_j|,
    whose value is maximal when k is at the ith largest logit.
    So that the permutation matrix can be expressed as
             / 1 if j = argmax_k fn_i(k)
      P_ij = |                           = one_hot(argmax(fn_i(j))).
             \ 0 otherwise
    And the differentiable approximation of the matrix is applied with softmax,
      P^_ij = softmax(fn_i(j) / temperature),
    where the parameter temperature tunes the smoothiness of the approximation.
    #### References
    [1]: Aditya Grover, Eric Wang, Aaron Zweig, Stefano Ermon.
         Stochastic Optimization of Sorting Networks via Continuous Relaxations.
         https://arxiv.org/abs/1903.08850
    Args:
      logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
        ranking score of the corresponding item. (We are using logits here,
        noticing the original paper is using probability weights, i.e., the
        exponentials of the logits).
      name: A string used as the name for this loss.
      temperature: The Softmax approximation temperature.
    Returns:
      A tensor of permutation matrices whose dimension is [batch_size, list_size,
      list_size].
    """
    with tf.compat.v1.name_scope(name, 'neural_sort', [logits]):
        list_size = tf.shape(input=logits)[1]

        logit_diff = tf.abs(tf.expand_dims(logits, 2) - tf.expand_dims(logits, 1))
        # shape = [batch_size, 1, list_size].
        logit_diff_sum = tf.reduce_sum(
            input_tensor=logit_diff, axis=1, keepdims=True)
        scaling = tf.cast(
            list_size + 1 - 2 * (tf.range(list_size) + 1), dtype=tf.float32)
        # shape = [1, list_size, 1].
        scaling = tf.expand_dims(tf.expand_dims(scaling, 1), 0)
        # shape = [batch_size, list_size, list_size].
        # Use broadcast to align the dims.
        scaled_logits = scaling * tf.expand_dims(logits, 1)

        p_logits = scaled_logits - logit_diff_sum
        smooth_perm = tf.nn.softmax(p_logits / temperature, -1)

        return smooth_perm


@tf.function
def compute_approxNDCG_unreduced_loss(logits, labels):
    # alpha = self._params.get('alpha', 10.0)
    alpha = 10.0
    # print('alpha from 10 to 0.1')
    # alpha = 0.1
    # the higher the alpha the more the sigmoid approximating the indicator function is steep in the nDCG approx.
    is_valid = utils.is_label_valid(labels)
    labels = tf.compat.v1.where(is_valid, labels, tf.zeros_like(labels))
    logits = tf.compat.v1.where(
        is_valid, logits, -1e3 * tf.ones_like(logits) +
                          tf.reduce_min(input_tensor=logits, axis=-1, keepdims=True))

    label_sum = tf.reduce_sum(input_tensor=labels, axis=1, keepdims=True)
    nonzero_mask = tf.greater(tf.reshape(label_sum, [-1]), 0.0)
    labels = tf.compat.v1.where(nonzero_mask, labels,
                                1e-10 * tf.ones_like(labels))
    gains = tf.pow(2., tf.cast(labels, dtype=tf.float32)) - 1.
    original_ranks = utils.approx_ranks(logits, alpha=alpha)
    ranks = original_ranks
    discounts = 1. / tf.math.log1p(ranks)
    dcg = tf.reduce_sum(input_tensor=gains * discounts, axis=-1, keepdims=True)
    cost = -dcg * utils.inverse_max_dcg(labels)
    return cost, tf.reshape(tf.cast(nonzero_mask, dtype=tf.float32), [-1, 1])




def compute_pairwise_hinge_loss(logits, labels):
    # is_valid = utils.is_label_valid(labels)
    # ranks = _compute_ranks(logits, is_valid)
    pairwise_labels, pairwise_logits = _pairwise_comparison(labels, logits)
    pairwise_weights = pairwise_labels
    # if self._lambda_weight is not None:
    #     pairwise_weights *= self._lambda_weight.pair_weights(labels, ranks)
    #     # For LambdaLoss with relative rank difference, the scale of loss becomes
    #     # much smaller when applying LambdaWeight. This affects the training can
    #     # make the optimal learning rate become much larger. We use a heuristic to
    #     # scale it up to the same magnitude as standard pairwise loss.
    #     pairwise_weights *= tf.cast(tf.shape(input=labels)[1], dtype=tf.float32)

    pairwise_weights = tf.stop_gradient(
        pairwise_weights, name='weights_stop_gradient')
    return tf.nn.relu(1 - pairwise_logits), pairwise_weights
