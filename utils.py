from hickle import hickle


def load_model(path):
    model = hickle.load(path)
    return model


def save_model(model, path):
    model = hickle.dump(model, path)
    return model


def pad_list(l, max_len):
    l = list(l)
    while len(l) < max_len:
        l.append(l[-1])
    l = l[0: min(len(l), max_len)]
    return l


import tensorflow as tf

"""
https://github.com/tensorflow/ranking/blob/4ed523746cc473652aba89c731019b505c1acc38/tensorflow_ranking/python/utils.py#L24
"""


def sort_by_scores(scores,
                   features_list,
                   topn=None,
                   shuffle_ties=True,
                   seed=None):
    """Sorts example features according to per-example scores.
    Args:
      scores: A `Tensor` of shape [batch_size, list_size] representing the
        per-example scores.
      features_list: A list of `Tensor`s with the same shape as scores to be
        sorted.
      topn: An integer as the cutoff of examples in the sorted list.
      shuffle_ties: A boolean. If True, randomly shuffle before the sorting.
      seed: The ops-level random seed used when `shuffle_ties` is True.
    Returns:
      A list of `Tensor`s as the list of sorted features by `scores`.
    """
    with tf.compat.v1.name_scope(name='sort_by_scores'):
        scores = tf.cast(scores, tf.float32)
        scores.get_shape().assert_has_rank(2)
        list_size = tf.shape(input=scores)[1]
        if topn is None:
            topn = list_size
        topn = tf.minimum(topn, list_size)
        shuffle_ind = None
        if shuffle_ties:
            shuffle_ind = _to_nd_indices(
                tf.argsort(
                    tf.random.uniform(tf.shape(input=scores), seed=seed),
                    stable=True))
            scores = tf.gather_nd(scores, shuffle_ind)
        _, indices = tf.math.top_k(scores, topn, sorted=True)
        nd_indices = _to_nd_indices(indices)
        if shuffle_ind is not None:
            nd_indices = tf.gather_nd(shuffle_ind, nd_indices)
        return [tf.gather_nd(f, nd_indices) for f in features_list]


def _to_nd_indices(indices):
    """Returns indices used for tf.gather_nd or tf.scatter_nd.
    Args:
      indices: A `Tensor` of shape [batch_size, size] with integer values. The
        values are the indices of another `Tensor`. For example, `indices` is the
        output of tf.argsort or tf.math.top_k.
    Returns:
      A `Tensor` with shape [batch_size, size, 2] that can be used by tf.gather_nd
      or tf.scatter_nd.
    """
    indices.get_shape().assert_has_rank(2)
    batch_ids = tf.ones_like(indices) * tf.expand_dims(
        tf.range(tf.shape(input=indices)[0]), 1)
    return tf.stack([batch_ids, indices], axis=-1)


def is_label_valid(labels):
    """Returns a boolean `Tensor` for label validity."""
    labels = tf.convert_to_tensor(value=labels)
    return tf.greater_equal(labels, 0.)
    # return tf.greater_equal(labels, -1.)


def approx_ranks(logits, alpha=10.):
    r"""Computes approximate ranks given a list of logits.
    Given a list of logits, the rank of an item in the list is simply
    one plus the total number of items with a larger logit. In other words,
      rank_i = 1 + \sum_{j \neq i} I_{s_j > s_i},
    where "I" is the indicator function. The indicator function can be
    approximated by a generalized sigmoid:
      I_{s_j < s_i} \approx 1/(1 + exp(-\alpha * (s_j - s_i))).
    This function approximates the rank of an item using this sigmoid
    approximation to the indicator function. This technique is at the core
    of "A general approximation framework for direct optimization of
    information retrieval measures" by Qin et al.
    Args:
      logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
        ranking score of the corresponding item.
      alpha: Exponent of the generalized sigmoid function.
    Returns:
      A `Tensor` of ranks with the same shape as logits.
    """
    list_size = tf.shape(input=logits)[1]
    x = tf.tile(tf.expand_dims(logits, 2), [1, 1, list_size])
    y = tf.tile(tf.expand_dims(logits, 1), [1, list_size, 1])
    pairs = tf.sigmoid(alpha * (y - x))
    return tf.reduce_sum(input_tensor=pairs, axis=-1) + .5


def inverse_max_dcg(labels,
                    gain_fn=lambda labels: tf.pow(2.0, labels) - 1.,
                    rank_discount_fn=lambda rank: 1. / tf.math.log1p(rank),
                    topn=None):
    """Computes the inverse of max DCG.
    Args:
      labels: A `Tensor` with shape [batch_size, list_size]. Each value is the
        graded relevance of the corresponding item.
      gain_fn: A gain function. By default this is set to: 2^label - 1.
      rank_discount_fn: A discount function. By default this is set to:
        1/log(1+rank).
      topn: An integer as the cutoff of examples in the sorted list.
    Returns:
      A `Tensor` with shape [batch_size, 1].
    """
    ideal_sorted_labels, = sort_by_scores(labels, [labels], topn=topn)
    rank = tf.range(tf.shape(input=ideal_sorted_labels)[1]) + 1
    discounted_gain = gain_fn(ideal_sorted_labels) * rank_discount_fn(
        tf.cast(rank, dtype=tf.float32))
    discounted_gain = tf.reduce_sum(
        input_tensor=discounted_gain, axis=1, keepdims=True)
    return tf.compat.v1.where(
        tf.greater(discounted_gain, 0.), 1. / discounted_gain,
        tf.zeros_like(discounted_gain))


def sorted_ranks(scores, shuffle_ties=True, seed=None):
    """Returns an int `Tensor` as the ranks (1-based) after sorting scores.
    Example: Given scores = [[1.0, 3.5, 2.1]], the returned ranks will be [[3, 1,
    2]]. It means that scores 1.0 will be ranked at position 3, 3.5 will be ranked
    at position 1, and 2.1 will be ranked at position 2.
    Args:
      scores: A `Tensor` of shape [batch_size, list_size] representing the
        per-example scores.
      shuffle_ties: See `sort_by_scores`.
      seed: See `sort_by_scores`.
    Returns:
      A 1-based int `Tensor`s as the ranks.
    """
    with tf.compat.v1.name_scope(name='sorted_ranks'):
        batch_size, list_size = tf.unstack(tf.shape(scores))
        # The current position in the list for each score.
        positions = tf.tile(tf.expand_dims(tf.range(list_size), 0), [batch_size, 1])
        # For score [[1.0, 3.5, 2.1]], sorted_positions are [[1, 2, 0]], meaning the
        # largest score is at poistion 1, the second is at postion 2 and third is at
        # position 0.
        sorted_positions = sort_by_scores(
            scores, [positions], shuffle_ties=shuffle_ties, seed=seed)[0]
        # The indices of sorting sorted_postions will be [[2, 0, 1]] and ranks are
        # 1-based and thus are [[3, 1, 2]].
        ranks = tf.argsort(sorted_positions) + 1
        return ranks
