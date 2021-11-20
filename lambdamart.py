import argparse
import multiprocessing
import os
import time

import hickle
import lightgbm as lgb
import numpy as np
import pyltr
import tensorflow.compat.v1 as tf
import logging
from tqdm import tqdm

from evaluation import compute_mean_ndcg, compute_P_at_k, compute_MAP, create_trec_eval_format_run_qrels

# from loss_functions_lambdamart import *

np.random.seed(0)

flags = tf.app.flags
FLAGS = flags.FLAGS

from multiprocessing import Array

npar_proc = 8
# max_len = 10000
max_len = 2500000  # 800K WEB10K
# global shared_qids
# global shared_preds
# global shared_labels

shared_qids = Array('f', max_len, lock=False)
shared_preds = Array('f', max_len, lock=False)
shared_labels = Array('f', max_len, lock=False)

np.random.seed(0)


def add_arguments(parser):
    """Build ArgumentParser."""
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--coll_name", type=str, default='MSLR-WEB30K', help="Collection name")
    parser.add_argument("--data_folder", type=str, default='../LETOR_data/MSLR-WEB30K', help="Data folder.")
    parser.add_argument("--simulate_labels", type=bool, default=False)
    # parser.add_argument("--lambdamart_preds_path", type=str, default='../LETOR_data/MQ2008/lambdamart_runs', help="LM data folder.")
    parser.add_argument("--lambdamart_preds_path", type=str, default='./output/lm_tests', help="LM data folder.")


def simulate_labels(real_labels, n=32):
    real_labels = real_labels / np.max(real_labels)
    sampled_labels = np.zeros(len(real_labels))
    for i in range(len(real_labels)):
        sampled_labels[i] = np.random.binomial(n, real_labels[i]) / n
    return sampled_labels


def grad_kl_bin(x, y, n=32):
    if (not (np.min(x) > 0 and np.max(x) < 1)) or (not (np.min(y) > 0 and np.max(y) < 1)):
        print('GRAD: x min: {}, x max: {}, y min: {}, y max: {}'.format(np.min(x), np.max(x), np.min(y), np.max(y)))
    # assert np.min(x) > 0 and np.max(x) < 1
    # assert np.min(y) > 0 and np.max(y) < 1
    return np.multiply(n, np.divide(-y, x) + (1 - y) / (1 - x) + np.log(x / y) - np.log((x - 1) / (y - 1)))


def hess_kl_bin(x, y, n=32):
    if (not (np.min(x) > 0 and np.max(x) < 1)) or (not (np.min(y) > 0 and np.max(y) < 1)):
        print('HESSIAN: x min: {}, x max: {}, y min: {}, y max: {}'.format(np.min(x), np.max(x), np.min(y), np.max(y)))
    # assert np.min(x) > 0 and np.max(x) < 1
    # assert np.min(y) > 0 and np.max(y) < 1
    return n * (y / (np.power(x, 2)) + (1 - y) / np.power(1 - x, 2) + 1 / x - 1 / (1 - x))


def grad_kl_normal(x, y):
    return 2 * (x - y)


def hess_kl_normal(x, y):
    return 2 * np.ones_like(x - y)


def sigmoid(x, alpha=1):
    return 1 / (1 + np.exp(-alpha * x))


def kl_bin_hinge_loss_original_value(x, y, n=32):
    return np.maximum(0, 1 - (
            np.log(y / x) * n * y + np.log((1 - y) / (1 - x)) * n * (1 - y) + np.log(x / y) * n * x + np.log(
        (1 - x) / (1 - y)) * n * (1 - x)) * np.sign(x - y))


def kl_norm_hinge_loss_original_value(x, y):
    return np.maximum(0, 1 - (0.5 + np.power(x - y, 2) - 0.5))


def kl_bin_pt_wise(preds, train_data):
    preds = (sigmoid(preds) + 1e-4) / (1 + 1e-2)
    if FLAGS.simulate_labels:
        labels = simulate_labels(train_data.get_label(), n=32)
        assert np.min(labels) >= 0 and np.max(labels) <= 1
    else:
        labels = train_data.get_label()

    labels = (labels + 1e-4) / (np.max(labels) + 1e-2)

    n_rel = sum([1 for lab in labels if lab > 0.1])
    n_not_rel = len(labels) - n_rel

    # w = [1 / n_rel if lab > 0.1 else 1 / n_not_rel for lab in labels]
    w = [n_rel / len(labels) if lab > 0.1 else n_not_rel / len(labels) for lab in labels]

    grad = grad_kl_bin(preds, labels) * w
    hess = hess_kl_bin(preds, labels)
    return grad, hess


def kl_normal_pt_wise(preds, train_data):
    preds = (sigmoid(preds) + 1e-4) / (1 + 1e-2)
    if FLAGS.simulate_labels:
        labels = simulate_labels(train_data.get_label(), n=32)
        assert np.min(labels) >= 0 and np.max(labels) <= 1
    else:
        labels = train_data.get_label()

    labels = (labels + 1e-4) / (np.max(labels) + 1e-2)
    n_rel = sum([1 for lab in labels if lab > 0.1])
    n_not_rel = len(labels) - n_rel

    w = [n_rel / len(labels) if lab > 0.1 else n_not_rel / len(labels) for lab in labels]
    grad = grad_kl_normal(preds, labels) * w
    hess = hess_kl_normal(preds, labels)
    return grad, hess


def kl_normal_list_wise(preds, train_data):
    preds = (sigmoid(preds) + 1e-4) / (1 + 1e-2)
    if FLAGS.simulate_labels:
        labels = simulate_labels(train_data.get_label(), n=32)
        assert np.min(labels) >= 0 and np.max(labels) <= 1
    else:
        labels = train_data.get_label()

    labels = (labels + 1e-4) / (np.max(labels) + 1e-2)
    n_rel = sum([1 for lab in labels if lab > 0.1])
    n_not_rel = len(labels) - n_rel
    w = [n_rel / len(labels) if lab > 0.1 else n_not_rel / len(labels) for lab in labels]

    grad = grad_kl_normal(preds, labels) * w / 0.25
    hess = hess_kl_normal(preds, labels)
    return grad, hess


def kl_normal_pair_wise(preds, train_data):
    preds = (sigmoid(preds, alpha=5) + 1e-4) / (1 + 1e-2)
    if FLAGS.simulate_labels:
        labels = simulate_labels(train_data.get_label(), n=32)
        assert np.min(labels) >= 0 and np.max(labels) <= 1
    else:
        labels = train_data.get_label()

    labels = (labels + 1e-4) / (np.max(labels) + 1e-2)
    groups = train_data.get_group()

    group_mask = np.zeros(shape=(len(preds), len(preds)))
    for k in range(len(groups)):
        group_mask[sum(groups[:k]):sum(groups[:k]) + groups[k], sum(groups[:k]):sum(groups[:k]) + groups[k]] = 1

    lab_diffs = np.expand_dims(labels, axis=-1) - np.expand_dims(labels, axis=-2)
    pred_diffs_sign = np.sign(np.expand_dims(preds, axis=-1) - np.expand_dims(preds, axis=-2))

    # coeff = np.where(lab_diffs >= 0, np.ones_like(lab_diffs), 0)
    coeff = np.where(lab_diffs < 0, np.ones_like(lab_diffs), 0)
    coeff *= group_mask * 1 / np.sum(group_mask, axis=0)

    normal_l_orig = kl_norm_hinge_loss_original_value(np.expand_dims(preds, axis=-1), np.expand_dims(preds, axis=-2))
    gvalue = grad_kl_normal(np.expand_dims(preds, axis=-1), np.expand_dims(preds, axis=-2)) * pred_diffs_sign
    hvalue = hess_kl_normal(np.expand_dims(preds, axis=-1), np.expand_dims(preds, axis=-2)) * pred_diffs_sign
    grads_matrix = np.where(normal_l_orig > 0, - gvalue, 0)
    hess_matrix = np.where(normal_l_orig > 0, - hvalue, 0)

    grads_matrix *= coeff
    hess_matrix *= coeff
    if np.max(grads_matrix) == np.min(grads_matrix):
        print('USELESS GRADS')
        grad = 2 * (preds - labels)
        hess = np.ones_like(preds) * 2
    else:
        grad = np.sum(grads_matrix, axis=1)
        hess = np.sum(hess_matrix, axis=1)
    return grad, hess


def compute_qid_gmask(curr_qid, all_qids):
    return np.where(np.array(all_qids) == curr_qid, np.ones(len(all_qids)), 0)


# @jit(nopython=True, parallel=True)
def process_row_data_kl_bin_pwise(idx, len_limit):
    curr_preds = np.array(shared_preds[:len_limit])
    curr_labels = np.array(shared_labels[:len_limit])
    curr_qids = np.array(shared_qids[:len_limit])

    group_mask = compute_qid_gmask(curr_qids[idx], curr_qids)
    lab_diffs = np.ones_like(curr_labels) * curr_labels[idx] - curr_labels
    pred_diffs_sign = np.sign(np.ones_like(curr_preds) * curr_preds[idx] - curr_preds)
    coeff = np.where(lab_diffs < 0, np.ones_like(lab_diffs), 0)
    coeff *= np.where(group_mask > 0, np.multiply(group_mask, 1 / np.sum(group_mask)), 0)
    hinge_l_orig = kl_bin_hinge_loss_original_value(np.ones_like(curr_preds) * curr_preds[idx], curr_preds)

    gvalue = grad_kl_bin(np.ones_like(curr_preds) * curr_preds[idx], curr_preds) * pred_diffs_sign
    hvalue = hess_kl_bin(np.ones_like(curr_preds) * curr_preds[idx], curr_preds) * pred_diffs_sign

    grads_matrix_row = np.where(hinge_l_orig > 0, - gvalue, 0)
    hess_matrix_row = np.where(hinge_l_orig > 0, - hvalue, 0)
    grads_matrix_row *= coeff
    hess_matrix_row *= coeff
    return np.sum(grads_matrix_row), np.sum(hess_matrix_row)


def process_row_data_kl_normal_pwise(idx, len_limit):
    curr_preds = np.array(shared_preds[:len_limit])
    curr_labels = np.array(shared_labels[:len_limit])
    curr_qids = np.array(shared_qids[:len_limit])

    group_mask = compute_qid_gmask(curr_qids[idx], curr_qids)
    lab_diffs = np.ones_like(curr_labels) * curr_labels[idx] - curr_labels
    pred_diffs_sign = np.sign(np.ones_like(curr_preds) * curr_preds[idx] - curr_preds)
    coeff = np.where(lab_diffs < 0, np.ones_like(lab_diffs), 0)
    coeff *= np.where(group_mask > 0, np.multiply(group_mask, 1 / np.sum(group_mask)), 0)
    hinge_l_orig = kl_norm_hinge_loss_original_value(np.ones_like(curr_preds) * curr_preds[idx], curr_preds)

    gvalue = grad_kl_normal(np.ones_like(curr_preds) * curr_preds[idx], curr_preds) * pred_diffs_sign
    hvalue = hess_kl_normal(np.ones_like(curr_preds) * curr_preds[idx], curr_preds) * pred_diffs_sign

    grads_matrix_row = np.where(hinge_l_orig > 0, - gvalue, 0)
    hess_matrix_row = np.where(hinge_l_orig > 0, - hvalue, 0)
    grads_matrix_row *= coeff
    hess_matrix_row *= coeff
    return np.sum(grads_matrix_row), np.sum(hess_matrix_row)


def initpool(p, q, l):
    global shared_preds
    global shared_qids
    global shared_labels
    shared_preds[:len(p)] = p
    shared_qids[:len(q)] = q
    shared_labels[:len(l)] = l


def kl_bin_pair_wise_parallel(preds, train_data):
    preds = (sigmoid(preds, alpha=5) + 1e-4) / (1 + 1e-2)
    if FLAGS.simulate_labels:
        labels = simulate_labels(train_data.get_label(), n=32)
        assert np.min(labels) >= 0 and np.max(labels) <= 1
    else:
        labels = train_data.get_label()

    labels = (labels + 1e-4) / (np.max(labels) + 1e-2)
    groups = train_data.get_group()
    assert np.min(preds) > 0 and np.max(preds) < 1
    assert np.min(labels) > 0 and np.max(labels) < 1
    assert len(preds) < max_len
    qids = []
    curr_qid = 0
    for glen in groups:
        for i in range(glen):
            qids.append(curr_qid)
        curr_qid += 1
    s = time.time()

    # shared_qids[:len(qids)] = qids
    # shared_preds[:len(preds)] = preds
    # shared_labels[:len(labels)] = labels
    # seq_results = []
    # for i in tqdm(range(len(preds))):
    #     r = process_row_data_kl_bin_pwise(i, len(preds))
    #     seq_results.append(r)

    pool = multiprocessing.Pool(processes=npar_proc, initializer=initpool, initargs=(preds, qids, labels))

    results = pool.starmap(process_row_data_kl_bin_pwise,
                           [(i, len(preds)) for i in range(len(preds))])
    grad = np.array(results)[:, 0]
    hess = np.array(results)[:, 1]
    print('time to compute gradients = {}'.format(time.time() - s))
    if np.max(grad) == np.min(hess):
        print('USELESS GRADS')
        grad = 2 * (preds - labels)
        hess = np.ones_like(preds) * 2
    return grad, hess


def kl_normal_pair_wise_parallel(preds, train_data):
    preds = (sigmoid(preds, alpha=5) + 1e-4) / (1 + 1e-2)
    if FLAGS.simulate_labels:
        labels = simulate_labels(train_data.get_label(), n=32)
        assert np.min(labels) >= 0 and np.max(labels) <= 1
    else:
        labels = train_data.get_label()

    labels = (labels + 1e-4) / (np.max(labels) + 1e-2)
    groups = train_data.get_group()
    assert np.min(preds) > 0 and np.max(preds) < 1
    assert np.min(labels) > 0 and np.max(labels) < 1
    assert len(preds) < max_len
    qids = []
    curr_qid = 0
    for glen in groups:
        for i in range(glen):
            qids.append(curr_qid)
        curr_qid += 1
    s = time.time()

    # shared_qids[:len(qids)] = qids
    # shared_preds[:len(preds)] = preds
    # shared_labels[:len(labels)] = labels
    # seq_results = []
    # for i in tqdm(range(len(preds))):
    #     r = process_row_data_kl_bin_pwise(i, len(preds))
    #     seq_results.append(r)

    pool = multiprocessing.Pool(processes=npar_proc, initializer=initpool, initargs=(preds, qids, labels))

    results = pool.starmap(process_row_data_kl_normal_pwise,
                           [(i, len(preds)) for i in range(len(preds))])
    grad = np.array(results)[:, 0]
    hess = np.array(results)[:, 1]
    print('time to compute gradients = {}'.format(time.time() - s))
    if np.max(grad) == np.min(hess):
        print('USELESS GRADS')
        grad = 2 * (preds - labels)
        hess = np.ones_like(preds) * 2
    return grad, hess


def kl_bin_pair_wise(preds, train_data):
    preds = (sigmoid(preds, alpha=5) + 1e-4) / (1 + 1e-2)
    if FLAGS.simulate_labels:
        labels = simulate_labels(train_data.get_label(), n=32)
        assert np.min(labels) >= 0 and np.max(labels) <= 1
    else:
        labels = train_data.get_label()

    labels = (labels + 1e-4) / (np.max(labels) + 1e-2)
    groups = train_data.get_group()

    s = time.time()
    group_mask = np.zeros(shape=(len(preds), len(preds)))
    for k in range(len(groups)):
        group_mask[sum(groups[:k]):sum(groups[:k]) + groups[k], sum(groups[:k]):sum(groups[:k]) + groups[k]] = 1

    lab_diffs = np.expand_dims(labels, axis=-1) - np.expand_dims(labels, axis=-2)
    pred_diffs_sign = np.sign(np.expand_dims(preds, axis=-1) - np.expand_dims(preds, axis=-2))

    coeff = np.where(lab_diffs < 0, np.ones_like(lab_diffs), 0)
    coeff *= np.multiply(group_mask, 1 / np.sum(group_mask, axis=0))
    hinge_l_orig = kl_bin_hinge_loss_original_value(np.expand_dims(preds, axis=-1), np.expand_dims(preds, axis=-2))
    gvalue = grad_kl_bin(np.expand_dims(preds, axis=-1), np.expand_dims(preds, axis=-2)) * pred_diffs_sign
    hvalue = hess_kl_bin(np.expand_dims(preds, axis=-1), np.expand_dims(preds, axis=-2)) * pred_diffs_sign
    #############################################################
    # qids = []
    # curr_qid = 0
    # for glen in groups:
    #     for i in range(glen):
    #         qids.append(curr_qid)
    #     curr_qid += 1
    # for idx in range(len(preds)):
    #     alt_group_mask = compute_qid_gmask(qids[idx], qids)
    #     alt_lab_diffs = np.ones_like(labels) * labels[idx] - labels
    #     alt_pred_diffs_sign = np.sign(np.ones_like(preds) * preds[idx] - preds)
    #     alt_coeff = np.where(alt_lab_diffs < 0, np.ones_like(alt_lab_diffs), 0)
    #     alt_coeff *= np.where(alt_group_mask > 0, np.multiply(alt_group_mask, 1 / np.sum(alt_group_mask)), 0)
    #     alt_hinge_l_orig = kl_bin_hinge_loss_original_value(np.ones_like(preds) * preds[idx], preds)
    #     assert False not in (alt_group_mask == group_mask[idx])
    #     assert False not in (alt_lab_diffs == lab_diffs[idx])
    #     assert False not in (alt_pred_diffs_sign == pred_diffs_sign[idx])
    #     assert False not in (alt_coeff == coeff[idx])
    #     assert False not in (alt_hinge_l_orig == hinge_l_orig[idx])
    #     alt_gvalue = grad_kl_bin(np.ones_like(preds) * preds[idx], preds) * alt_pred_diffs_sign
    #     alt_hvalue = hess_kl_bin(np.ones_like(preds) * preds[idx], preds) * alt_pred_diffs_sign
    #     assert False not in (alt_gvalue == gvalue[idx])
    #     assert False not in (alt_hvalue == hvalue[idx])
    ############################################################

    grads_matrix = np.where(hinge_l_orig > 0, - gvalue, 0)
    hess_matrix = np.where(hinge_l_orig > 0, - hvalue, 0)
    grads_matrix *= coeff
    hess_matrix *= coeff
    if np.max(grads_matrix) == np.min(grads_matrix):
        print('USELESS GRADS')
        grad = 2 * (preds - labels)
        hess = np.ones_like(preds) * 2
    else:
        grad = np.sum(grads_matrix, axis=1)
        hess = np.sum(hess_matrix, axis=1)
    print('time to compute gradients = {}'.format(time.time() - s))
    return grad, hess


def compute_lambdamart_preds_w_custom_loss(FLAGS, loss_fn):
    print('LOSS: {}'.format(loss_fn.__name__))
    num_round = 200
    # experiment_suff = str(uuid.uuid4())
    # param = {'learning_rate': 0.5, 'num_leaves': 12, 'num_trees': 48, 'num_threads': 8, 'metric': 'ndcg', 'max_depth': 6,
    #  'objective': 'rank_xendcg', 'min_data_in_leaf': 0, 'min_sum_hessian_in_leaf': 100, 'ndcg_eval_at': 1} best MQ2007

    if FLAGS.coll_name == 'MQ2008':
        param = {'learning_rate': 0.1, 'num_leaves': 20, 'num_trees': 48, 'num_threads': 8, 'metric': 'ndcg',
                 'objective': 'rank_xendcg', 'min_data_in_leaf': 0, 'min_sum_hessian_in_leaf': 100,
                 'ndcg_eval_at': 1}  # best on MQ2008
        if loss_fn.__name__ == 'kl_normal_pt_wise' or loss_fn.__name__ == 'kl_normal_list_wise':
            param = {'learning_rate': 0.9, 'num_leaves': 12, 'num_trees': 64, 'num_threads': 8, 'metric': 'ndcg',
                     'max_depth': 6, 'objective': 'rank_xendcg', 'min_data_in_leaf': 0, 'min_sum_hessian_in_leaf': 100,
                     'ndcg_eval_at': 1}
    elif FLAGS.coll_name == 'MQ2007':
        if loss_fn.__name__ == 'kl_normal_list_wise':
            param = {'learning_rate': 0.5, 'num_leaves': 16, 'num_trees': 48, 'num_threads': 8, 'metric': 'ndcg',
                     'max_depth': 6, 'objective': 'rank_xendcg', 'min_data_in_leaf': 0, 'min_sum_hessian_in_leaf': 100,
                     'ndcg_eval_at': 5}
        elif loss_fn.__name__ == 'kl_bin_pt_wise':
            param = {'learning_rate': 0.5, 'num_leaves': 12, 'num_trees': 48, 'num_threads': 8, 'metric': 'ndcg',
                     'max_depth': 6, 'objective': 'rank_xendcg', 'min_data_in_leaf': 0, 'min_sum_hessian_in_leaf': 100,
                     'ndcg_eval_at': 1}
        elif loss_fn.__name__ == 'kl_normal_pt_wise':
            param = {'learning_rate': 0.9, 'num_leaves': 12, 'num_trees': 64, 'num_threads': 8, 'metric': 'ndcg',
                     'max_depth': 6, 'objective': 'rank_xendcg', 'min_data_in_leaf': 0, 'min_sum_hessian_in_leaf': 100,
                     'ndcg_eval_at': 1}
        else:
            param = {'learning_rate': 0.5, 'num_leaves': 12, 'num_trees': 48, 'num_threads': 8, 'metric': 'ndcg',
                     'max_depth': 6, 'objective': 'rank_xendcg', 'min_data_in_leaf': 0, 'min_sum_hessian_in_leaf': 100,
                     'ndcg_eval_at': 1}

    elif FLAGS.coll_name == 'MSLR-WEB30K' or FLAGS.coll_name == 'MSLR-WEB10K':
        param = {'learning_rate': 0.8, 'num_leaves': 12, 'num_trees': 250, 'num_threads': 8, 'metric': 'ndcg',
                 'max_depth': 6, 'objective': 'rank_xendcg', 'min_data_in_leaf': 0, 'min_sum_hessian_in_leaf': 100,
                 'ndcg_eval_at': 1}
    elif FLAGS.coll_name == 'OHSUMED':
        param = {'learning_rate': 0.05, 'num_leaves': 20, 'num_trees': 64, 'num_threads': 8, 'metric': 'ndcg',
                 'objective': loss_fn, 'min_data_in_leaf': 0, 'min_sum_hessian_in_leaf': 100, 'ndcg_eval_at': 1}
    if 'pair' in loss_fn.__name__:
        param['n_iter_no_change'] = 100
    print(param)
    avg_measures = {'ndcg1': [], 'ndcg3': [], 'ndcg5': [], 'ndcg10': [], 'ndcg20': [], 'p1': [], 'p3': [], 'p5': [],
                    'MAP': []}
    if FLAGS.coll_name == 'MSLR-WEB10K' or FLAGS.coll_name == 'MSLR-WEB30K':
        fold_folders = ['Fold1']
    else:
        fold_folders = ['Fold1', 'Fold2', 'Fold3', 'Fold4', 'Fold5']
    all_preds = []
    all_dids = []
    all_qids = []
    all_rjs = []
    curr_qid = 0
    for fold_f in fold_folders:
        print('*** FOLDER: %s' % fold_f)
        save_preds_path = os.path.join(FLAGS.lambdamart_preds_path, FLAGS.coll_name + '_lightgbm_' + fold_f + '.hkl')
        if FLAGS.coll_name == 'OHSUMED':
            docs_train, lab_train, qids_train, _ = pyltr.data.letor.read_dataset(
                open(os.path.join(FLAGS.data_folder, 'Feature-min/' + fold_f + '/train.txt')))
            docs_test, lab_test, qids_test, _ = pyltr.data.letor.read_dataset(
                open(os.path.join(FLAGS.data_folder, 'Feature-min/' + fold_f + '/test.txt')))
            docs_val, lab_val, qids_val, _ = pyltr.data.letor.read_dataset(
                open(os.path.join(FLAGS.data_folder, 'Feature-min/' + fold_f + '/vali.txt')))
        else:
            docs_train, lab_train, qids_train, _ = pyltr.data.letor.read_dataset(
                open(os.path.join(FLAGS.data_folder, fold_f + '/train.txt')))
            docs_test, lab_test, qids_test, _ = pyltr.data.letor.read_dataset(
                open(os.path.join(FLAGS.data_folder, fold_f + '/test.txt')))
            docs_val, lab_val, qids_val, _ = pyltr.data.letor.read_dataset(
                open(os.path.join(FLAGS.data_folder, fold_f + '/vali.txt')))

        # if FLAGS.simulate_labels:
        #     lab_train = simulate_labels(lab_train, n=32)

        docs_train = np.array(docs_train)
        docs_test = np.array(docs_test)
        docs_val = np.array(docs_val)
        lab_test = np.array(lab_test)
        lab_train = np.array(lab_train)
        lab_val = np.array(lab_val)

        train_data = lgb.Dataset(docs_train, lab_train, group=get_f_groups(qids_train))
        validation_data = lgb.Dataset(docs_val, lab_val, group=get_f_groups(qids_val))
        bst = lgb.train(param, train_data, num_round, fobj=loss_fn, valid_sets=[validation_data])
        preds = bst.predict(docs_test, num_iteration=bst.best_iteration)
        # dumping preds to file
        hickle.dump(preds, open(save_preds_path, 'w'))
        print('preds saved to path: %s' % save_preds_path)
        grouped_preds, grouped_rj = group_ranklib_preds_in_ranking_lists(qids_test, preds, lab_test)
        ideal_rel_j_lists = [np.array(rl)[np.argsort(-np.array(rl))] for rl in grouped_rj]

        ndcg1, _ = compute_mean_ndcg(grouped_preds, grouped_rj, ideal_rel_j_lists, 1)
        ndcg3, _ = compute_mean_ndcg(grouped_preds, grouped_rj, ideal_rel_j_lists, 3)
        ndcg5, _ = compute_mean_ndcg(grouped_preds, grouped_rj, ideal_rel_j_lists, 5)
        ndcg10, _ = compute_mean_ndcg(grouped_preds, grouped_rj, ideal_rel_j_lists, 10)
        ndcg20, _ = compute_mean_ndcg(grouped_preds, grouped_rj, ideal_rel_j_lists, 20)
        p1 = compute_P_at_k(grouped_preds, grouped_rj, 1)
        p3 = compute_P_at_k(grouped_preds, grouped_rj, 3)
        p5 = compute_P_at_k(grouped_preds, grouped_rj, 5)
        map_v = compute_MAP(grouped_preds, grouped_rj)

        print('{} ndcg@1: {}'.format(fold_f, ndcg1))
        print('{} ndcg@3: {}'.format(fold_f, ndcg3))
        print('{} ndcg@5: {}'.format(fold_f, ndcg5))
        print('{} ndcg@10: {}'.format(fold_f, ndcg10))
        print('{} ndcg@20: {}'.format(fold_f, ndcg20))
        avg_measures['ndcg1'].append(ndcg1)
        avg_measures['ndcg3'].append(ndcg3)
        avg_measures['ndcg5'].append(ndcg5)
        avg_measures['ndcg10'].append(ndcg10)
        avg_measures['ndcg20'].append(ndcg20)
        avg_measures['p1'].append(p1)
        avg_measures['p3'].append(p3)
        avg_measures['p5'].append(p5)
        avg_measures['MAP'].append(map_v)

        for ranked_list in grouped_preds:
            curr_qid += 1
            fake_dids = []
            for i in range(len(ranked_list)):
                fake_dids.append('{}_{}'.format(curr_qid, i))
            all_dids.append(fake_dids)
            all_qids.append(curr_qid)
        all_preds.extend(grouped_preds)
        all_rjs.extend(grouped_rj)

    create_trec_eval_format_run_qrels(all_preds, all_dids, all_qids, all_rjs,
                                      'PR_{}_lambdaMART_loss={}_simulate_labels={}'.format(FLAGS.coll_name,
                                                                                           loss_fn.__name__,
                                                                                           FLAGS.simulate_labels),
                                      './output')

    print()
    print()
    print('AVG ndcg@1: {}'.format(np.mean(avg_measures['ndcg1'])))
    print('AVG ndcg@3: {}'.format(np.mean(avg_measures['ndcg3'])))
    print('AVG ndcg@5: {}'.format(np.mean(avg_measures['ndcg5'])))
    print('AVG ndcg@10: {}'.format(np.mean(avg_measures['ndcg10'])))
    print('AVG ndcg@20: {}'.format(np.mean(avg_measures['ndcg20'])))
    print('AVG P@1: {}'.format(np.mean(avg_measures['p1'])))
    print('AVG P@3: {}'.format(np.mean(avg_measures['p3'])))
    print('AVG P@5: {}'.format(np.mean(avg_measures['p5'])))
    print('AVG MAP: {}'.format(np.mean(avg_measures['MAP'])))


def grid_search():
    num_trees_set = [300, 500, 1000]
    num_leaves_set = [200, 500, 1000]
    learning_rate_set = [0.001, 0.05, 0.1, 0.5]

    fold_f = 'Fold1'
    docs_train, lab_train, qids_train, _ = pyltr.data.letor.read_dataset(
        open(os.path.join(FLAGS.data_folder, fold_f + '/train.txt')))
    docs_test, lab_test, qids_test, _ = pyltr.data.letor.read_dataset(
        open(os.path.join(FLAGS.data_folder, fold_f + '/test.txt')))
    docs_val, lab_val, qids_val, _ = pyltr.data.letor.read_dataset(
        open(os.path.join(FLAGS.data_folder, fold_f + '/vali.txt')))

    all_perfs = []
    all_params = []
    for nt in tqdm(num_trees_set):
        for nl in num_leaves_set:
            for lr in learning_rate_set:
                param = {'learning_rate': lr, 'num_leaves': nl, 'num_trees': nt, 'num_threads': 8, 'metric': 'ndcg',
                         'objective': 'lambdarank', 'ndcg_eval_at': 1}

                docs_train = np.array(docs_train)
                docs_test = np.array(docs_test)
                docs_val = np.array(docs_val)
                lab_test = np.array(lab_test)
                lab_train = np.array(lab_train)
                lab_val = np.array(lab_val)

                ndcg1 = run_lambdamart(param, docs_train, lab_train, qids_train, docs_val, lab_val, qids_val,
                                       docs_test, qids_test, lab_test)
                all_perfs.append(ndcg1)
                all_params.append(param)

    best_params = all_params[np.argmax(all_perfs)]
    print('BEST params: {}'.format(best_params))
    return best_params


def run_lambdamart(param, docs_train, lab_train, qids_train, docs_val, lab_val, qids_val, docs_test, qids_test,
                   lab_test):
    train_data = lgb.Dataset(docs_train, lab_train, group=get_f_groups(qids_train))
    validation_data = lgb.Dataset(docs_val, lab_val, group=get_f_groups(qids_val))
    bst = lgb.train(param, train_data, valid_sets=[validation_data])
    preds = bst.predict(docs_test, num_iteration=bst.best_iteration)

    grouped_preds, grouped_rj = group_ranklib_preds_in_ranking_lists(qids_test, preds, lab_test)
    ideal_rel_j_lists = [np.array(rl)[np.argsort(-np.array(rl))] for rl in grouped_rj]

    ndcg1, _ = compute_mean_ndcg(grouped_preds, grouped_rj, ideal_rel_j_lists, 1)
    return ndcg1


def compute_lambdamart_preds(FLAGS, loss_fn='lambdarank'):
    num_round = 200
    # print('LOSS: rank_xendcg')
    print('LOSS: {}'.format(loss_fn))
    # experiment_suff = str(uuid.uuid4())
    if FLAGS.coll_name == 'MSLR-WEB30K' or FLAGS.coll_name == 'MSLR-WEB10K':
        param = {'learning_rate': 0.8, 'num_leaves': 12, 'num_trees': 48, 'num_threads': 16, 'metric': 'ndcg',
                 'objective': loss_fn, 'min_data_in_leaf': 0, 'min_sum_hessian_in_leaf': 100, 'ndcg_eval_at': 1}
    else:
        param = {'learning_rate': 0.05, 'num_leaves': 12, 'num_trees': 48, 'num_threads': 16, 'metric': 'ndcg',
                 'objective': loss_fn, 'min_data_in_leaf': 0, 'min_sum_hessian_in_leaf': 100, 'ndcg_eval_at': 1}
    print(param)
    avg_measures = {'ndcg1': [], 'ndcg3': [], 'ndcg5': [], 'ndcg10': [], 'ndcg20': [], 'p1': [], 'p3': [], 'p5': [],
                    'MAP': []}
    if FLAGS.coll_name == 'MSLR-WEB10K' or FLAGS.coll_name == 'MSLR-WEB30K':
        fold_folders = ['Fold1']
    else:
        fold_folders = ['Fold1', 'Fold2', 'Fold3', 'Fold4', 'Fold5']
    all_preds = []
    all_dids = []
    all_qids = []
    all_rjs = []
    curr_qid = 0
    for fold_f in fold_folders:
        print('*** FOLDER: %s' % fold_f)
        save_preds_path = os.path.join(FLAGS.lambdamart_preds_path, FLAGS.coll_name + '_lightgbm_' + fold_f + '.hkl')
        if FLAGS.coll_name == 'OHSUMED':
            docs_train, lab_train, qids_train, _ = pyltr.data.letor.read_dataset(
                open(os.path.join(FLAGS.data_folder, 'Feature-min/' + fold_f + '/train.txt')))
            docs_test, lab_test, qids_test, _ = pyltr.data.letor.read_dataset(
                open(os.path.join(FLAGS.data_folder, 'Feature-min/' + fold_f + '/test.txt')))
            docs_val, lab_val, qids_val, _ = pyltr.data.letor.read_dataset(
                open(os.path.join(FLAGS.data_folder, 'Feature-min/' + fold_f + '/vali.txt')))
        else:
            docs_train, lab_train, qids_train, _ = pyltr.data.letor.read_dataset(
                open(os.path.join(FLAGS.data_folder, fold_f + '/train.txt')))
            docs_test, lab_test, qids_test, _ = pyltr.data.letor.read_dataset(
                open(os.path.join(FLAGS.data_folder, fold_f + '/test.txt')))
            docs_val, lab_val, qids_val, _ = pyltr.data.letor.read_dataset(
                open(os.path.join(FLAGS.data_folder, fold_f + '/vali.txt')))

        if FLAGS.simulate_labels:
            lab_train = simulate_labels(lab_train, n=32)

        docs_train = np.array(docs_train)
        docs_test = np.array(docs_test)
        docs_val = np.array(docs_val)
        lab_test = np.array(lab_test)
        lab_train = np.array(lab_train)
        lab_val = np.array(lab_val)

        train_data = lgb.Dataset(docs_train, lab_train, group=get_f_groups(qids_train))
        validation_data = lgb.Dataset(docs_val, lab_val, group=get_f_groups(qids_val))
        bst = lgb.train(param, train_data, num_round, valid_sets=[validation_data])
        preds = bst.predict(docs_test, num_iteration=bst.best_iteration)
        # dumping preds to file
        hickle.dump(preds, open(save_preds_path, 'w'))
        print('preds saved to path: %s' % save_preds_path)
        grouped_preds, grouped_rj = group_ranklib_preds_in_ranking_lists(qids_test, preds, lab_test)
        ideal_rel_j_lists = [np.array(rl)[np.argsort(-np.array(rl))] for rl in grouped_rj]

        ndcg1, _ = compute_mean_ndcg(grouped_preds, grouped_rj, ideal_rel_j_lists, 1)
        ndcg3, _ = compute_mean_ndcg(grouped_preds, grouped_rj, ideal_rel_j_lists, 3)
        ndcg5, _ = compute_mean_ndcg(grouped_preds, grouped_rj, ideal_rel_j_lists, 5)
        ndcg10, _ = compute_mean_ndcg(grouped_preds, grouped_rj, ideal_rel_j_lists, 10)
        ndcg20, _ = compute_mean_ndcg(grouped_preds, grouped_rj, ideal_rel_j_lists, 20)
        p1 = compute_P_at_k(grouped_preds, grouped_rj, 1)
        p3 = compute_P_at_k(grouped_preds, grouped_rj, 3)
        p5 = compute_P_at_k(grouped_preds, grouped_rj, 5)
        map_v = compute_MAP(grouped_preds, grouped_rj)

        print('{} ndcg@1: {}'.format(fold_f, ndcg1))
        print('{} ndcg@3: {}'.format(fold_f, ndcg3))
        print('{} ndcg@5: {}'.format(fold_f, ndcg5))
        print('{} ndcg@10: {}'.format(fold_f, ndcg10))
        print('{} ndcg@20: {}'.format(fold_f, ndcg20))
        avg_measures['ndcg1'].append(ndcg1)
        avg_measures['ndcg3'].append(ndcg3)
        avg_measures['ndcg5'].append(ndcg5)
        avg_measures['ndcg10'].append(ndcg10)
        avg_measures['ndcg20'].append(ndcg20)
        avg_measures['p1'].append(p1)
        avg_measures['p3'].append(p3)
        avg_measures['p5'].append(p5)
        avg_measures['MAP'].append(map_v)

        for ranked_list in grouped_preds:
            curr_qid += 1
            fake_dids = []
            for i in range(len(ranked_list)):
                fake_dids.append('{}_{}'.format(curr_qid, i))
            all_dids.append(fake_dids)
            all_qids.append(curr_qid)
        all_preds.extend(grouped_preds)
        all_rjs.extend(grouped_rj)

    create_trec_eval_format_run_qrels(all_preds, all_dids, all_qids, all_rjs,
                                      'PR_{}_lambdaMART_loss={}_simulate_labels={}'.format(FLAGS.coll_name,
                                                                                           loss_fn,
                                                                                           FLAGS.simulate_labels),
                                      './output')
    # perfs = re_evaluate('./output/run-PR_MQ2008_lambdaMART_loss=rank_xendcg_simulate_labels=False.txt',
    #                     './output/qrels-PR_MQ2008_lambdaMART_loss=rank_xendcg_simulate_labels=False.txt', 'MQ2008')
    all_ideal_rel_j_lists = [np.array(rl)[np.argsort(-np.array(rl))] for rl in all_rjs]
    ndcg1, _ = compute_mean_ndcg(all_preds, all_rjs, all_ideal_rel_j_lists, 1)
    ndcg3, _ = compute_mean_ndcg(all_preds, all_rjs, all_ideal_rel_j_lists, 3)
    ndcg5, _ = compute_mean_ndcg(all_preds, all_rjs, all_ideal_rel_j_lists, 5)
    ndcg10, _ = compute_mean_ndcg(all_preds, all_rjs, all_ideal_rel_j_lists, 10)
    ndcg20, _ = compute_mean_ndcg(all_preds, all_rjs, all_ideal_rel_j_lists, 20)
    p1 = compute_P_at_k(all_preds, all_rjs, 1)
    p3 = compute_P_at_k(all_preds, all_rjs, 3)
    p5 = compute_P_at_k(all_preds, all_rjs, 5)

    print('{} p@1: {}'.format('ALL', p1))
    print('{} p@3: {}'.format('ALL', p3))
    print('{} p@5: {}'.format('ALL', p5))
    print('{} ndcg@10: {}'.format('ALL', ndcg10))
    print('{} ndcg@20: {}'.format('ALL', ndcg20))
    # perfs_by_q_base = compute_performance_measures_by_q(all_preds, all_rjs, all_ideal_rel_j_lists,
    #                                                     [len(rl) for rl in all_preds], 2.0)
    print()
    print()
    print('AVG ndcg@1: {}'.format(np.mean(avg_measures['ndcg1'])))
    print('AVG ndcg@3: {}'.format(np.mean(avg_measures['ndcg3'])))
    print('AVG ndcg@5: {}'.format(np.mean(avg_measures['ndcg5'])))
    print('AVG ndcg@10: {}'.format(np.mean(avg_measures['ndcg10'])))
    print('AVG ndcg@20: {}'.format(np.mean(avg_measures['ndcg20'])))
    print('AVG P@1: {}'.format(np.mean(avg_measures['p1'])))
    print('AVG P@3: {}'.format(np.mean(avg_measures['p3'])))
    print('AVG P@5: {}'.format(np.mean(avg_measures['p5'])))
    print('AVG MAP: {}'.format(np.mean(avg_measures['MAP'])))


def get_f_groups(qids):
    groups = []
    pqid = None
    group_cnt = 0
    for i in range(len(qids)):
        if pqid == qids[i] or pqid is None:
            pqid = qids[i]
            group_cnt += 1
        else:
            groups.append(group_cnt)
            group_cnt = 1
            pqid = qids[i]
    groups.append(group_cnt)
    return groups


def group_ranklib_preds_in_ranking_lists(qids, preds, lab_test):
    ranking_lists = {}
    for i in range(len(qids)):
        qid = qids[i]
        pred = preds[i]
        label = lab_test[i]
        if qid in ranking_lists.keys():
            ranking_lists[qid].append((pred, label))
        else:
            ranking_lists[qid] = [(pred, label)]
    doc_scores = []
    doc_rj = []
    for k, ranking_list in ranking_lists.items():
        curr_scores = []
        curr_rj = []
        for i in range(len(ranking_list)):
            curr_scores.append(ranking_list[i][0])
            curr_rj.append(ranking_list[i][1])
        doc_scores.append(curr_scores)
        doc_rj.append(curr_rj)
    return doc_scores, doc_rj


def read_mslr_data(fp):
    docs = []
    qids = []
    labels = []
    for line in tqdm(open(fp)):
        data = line.split()
        labels.append(int(float(data[0])))
        qids.append(int(data[1].split(':')[1]))
        docv = np.zeros(max([int(idx.split(':')[0]) for idx in data[2:]]))
        for idx, v in [(int(index.split(':')[0]), index.split(':')[1]) for index in data[2:]]:
            docv[int(idx - 1)] = float(v)
        docs.append(docv)
    return docs, labels, qids


if __name__ == '__main__':
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    logging.getLogger("tensorflow").setLevel(logging.CRITICAL)

    arg_parser = argparse.ArgumentParser()
    add_arguments(arg_parser)
    FLAGS, unparsed = arg_parser.parse_known_args()
    for arg in vars(FLAGS):
        print(arg, ":", getattr(FLAGS, arg))
    grid_search()
    # compute_lambdamart_preds(FLAGS)
    # compute_lambdamart_preds(FLAGS, 'lambdarank')
    # compute_lambdamart_preds(FLAGS, 'rank_xendcg')
    # compute_lambdamart_preds_w_custom_loss(FLAGS, kl_bin_pt_wise)
    # compute_lambdamart_preds_w_custom_loss(FLAGS, kl_normal_pt_wise)
    # compute_lambdamart_preds_w_custom_loss(FLAGS, kl_normal_list_wise)
    # compute_lambdamart_preds_w_custom_loss(FLAGS, kl_bin_pair_wise)  # takes up a ton of ram on MQ2007
    # compute_lambdamart_preds_w_custom_loss(FLAGS, kl_normal_pair_wise)  # takes up a ton of ram on MQ2007
    # compute_lambdamart_preds_w_custom_loss(FLAGS, kl_bin_pair_wise_parallel)  # takes up a ton of ram on MQ2007
    # compute_lambdamart_preds_w_custom_loss(FLAGS, kl_normal_pair_wise_parallel)  # takes up a ton of ram on MQ2007
