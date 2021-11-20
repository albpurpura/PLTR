import argparse
import logging
import os
import time
import uuid

import numpy as np
import pyltr
import tensorflow as tf

from evaluation import compute_mean_ndcg, compute_perf_metrics, create_trec_eval_format_run_qrels
from lambdamart import compute_lambdamart_preds
from model import ReRanker
from simulate_unsupervised_rj import compare_artif_rj_with_real_ones, sample_labels
from utils import load_model, pad_list, save_model

flags = tf.app.flags
FLAGS = flags.FLAGS


def add_arguments(parser):
    """Build ArgumentParser."""
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--coll_name", type=str, default='MQ2008', help="Collection name")
    parser.add_argument("--data_folder", type=str, default='../LETOR_data/MQ2008/', help="Data folder.")
    parser.add_argument("--simulate_labels", type=str, default=False,
                        help="Whether to train with simulated labels or not.")
    parser.add_argument("--expand_training_data", type=str, default=False,
                        help="Whether to expand training data or not.")
    parser.add_argument("--det_model", type=str, default=True, help="Whether to use probabilistic layers or not.")
    parser.add_argument("--rerank_lambdamart", type=str, default=False,
                        help="Whether to rerank lambdamart preds or from scratch.")
    parser.add_argument("--lambdamart_preds_path", type=str, default='../LETOR_data/MQ2008/lambdamart_runs',
                        help="LM data folder.")
    # model parameters
    parser.add_argument("--seed", type=float, default=0, help="The random seed to use.")
    parser.add_argument("--n_binomial_samples", type=float, default=32,
                        help="The number of binomial samples to simulate.")
    # parser.add_argument("--loss", type=str, default='KL_B', help="The loss to use to train the model.")
    # parser.add_argument("--loss", type=str, default='KL_G', help="The loss to use to train the model.")
    # parser.add_argument("--loss", type=str, default='KL_G_H', help="The loss to use to train the model.")
    # parser.add_argument("--loss", type=str, default='KL_B_H', help="The loss to use to train the model.")
    # parser.add_argument("--loss", type=str, default='ApproxNDCG', help="The loss to use to train the model.")
    parser.add_argument("--loss", type=str, default='ApproxNDCG_G', help="The loss to use to train the model.")
    parser.add_argument("--norm_labels", type=bool, default=False,
                        help="Whether to normalize within [0,1] the relevance labels.")
    parser.add_argument("--num_features", type=int, default=46, help="Number of features per document.")
    parser.add_argument("--num_epochs", type=int, default=100, help="The number of epochs for training.")
    parser.add_argument("--n_heads", type=int, default=2, help="Num heads.")
    parser.add_argument("--batch_size", type=int, default=4, help="The batch size for training.")  # MQ2007
    # parser.add_argument("--batch_size", type=int, default=2, help="The batch size for training.")  # MQ2008
    # parser.add_argument("--list_size_test", type=int, default=147, help="List size.")  # MQ2007
    parser.add_argument("--list_size_test", type=int, default=120, help="List size.")  # MQ2008
    parser.add_argument("--list_size_train", type=int, default=147, help="List size.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer.")  # MQ2008 and MQ2007
    parser.add_argument("--model_ckpt_path", type=str, default='./output/chkpts/',
                        help="Output path for checkpoint saving.")


def remove_queries_without_rel_docs(rj, docs, rl_lengths, dids):
    indices_to_remove = []
    for i in range(len(rj)):
        if max(rj[i]) == 0:
            indices_to_remove.append(i)
    rj = [rj[i] for i in range(len(rj)) if i not in indices_to_remove]
    docs = [docs[i] for i in range(len(docs)) if i not in indices_to_remove]
    rl_lengths = [rl_lengths[i] for i in range(len(rl_lengths)) if i not in indices_to_remove]
    dids = [dids[i] for i in range(len(dids)) if i not in indices_to_remove]
    return rj, docs, rl_lengths, dids


def group_docs_with_lambdamart_preds(preds, qids, docs, labels, max_list_size):
    grouped = {}
    for i in range(len(qids)):
        if qids[i] in grouped.keys():
            grouped[qids[i]].append((preds[i], docs[i], labels[i]))
        else:
            grouped[qids[i]] = [(preds[i], docs[i], labels[i])]
    grouped_docs = []
    grouped_labels = []
    rl_lengths = []
    for group in grouped.values():
        g = np.array(group)
        lmp = g[:, 0]
        indices = np.argsort(-lmp)

        ranked_list = list(g[:, 1][indices])
        ranked_labels = list(g[:, 2][indices])

        while len(ranked_list) < max_list_size:
            ranked_list.append(np.zeros(FLAGS.num_features))
            ranked_labels.append(0.0)
        ranked_list = ranked_list[:max_list_size]
        ranked_labels = ranked_labels[:max_list_size]

        grouped_docs.append(ranked_list)
        grouped_labels.append(ranked_labels)
        rl_lengths.append(min(max_list_size, len(lmp)))
    return grouped_docs, grouped_labels, rl_lengths


def read_data(data_folder, fold_f):
    # data_fpath = './data_proc/{}_{}_listSize={}_rerank_lambdamart={}.hkl'.format(FLAGS.coll_name, fold_f,
    #                                                                              FLAGS.list_size_test,
    #                                                                              FLAGS.rerank_lambdamart)
    # if not os.path.isfile(data_fpath) or not FLAGS.load_proc_data:

    training_file_path = os.path.join(os.path.join(data_folder, fold_f), 'train.txt')
    valid_file_path = os.path.join(os.path.join(data_folder, fold_f), 'vali.txt')
    test_file_path = os.path.join(os.path.join(data_folder, fold_f), 'test.txt')

    docs_train, lab_train, qids_train, _ = pyltr.data.letor.read_dataset(open(training_file_path))
    docs_val, lab_val, qids_val, _ = pyltr.data.letor.read_dataset(open(valid_file_path))
    docs_test, lab_test, qids_test, _ = pyltr.data.letor.read_dataset(open(test_file_path))

    dids_train = ['fake_did_{}'.format(i) for i in range(len(docs_train))]
    dids_test = ['fake_did_{}'.format(i) for i in range(len(docs_test))]
    dids_val = ['fake_did_{}'.format(i) for i in range(len(docs_val))]

    max_l = np.max(lab_train)
    print('max label: {}'.format(max_l))
    lab_train = np.array(lab_train) / max_l
    lab_val = np.array(lab_val) / max_l
    lab_test = np.array(lab_test) / max_l

    assert 0 <= max(lab_test) <= 1
    assert 0 <= max(lab_train) <= 1
    assert 0 <= max(lab_val) <= 1

    # without lambdamart
    ranking_lists_train, all_labels_train, rl_lengths_train, resp_qids_train, resp_dids_train = \
        group_data_in_ranking_lists(docs_train, lab_train, qids_train, dids_train, FLAGS.list_size_test)
    ranking_lists_val, all_labels_val, rl_lengths_val, resp_qids_val, resp_dids_val = \
        group_data_in_ranking_lists(docs_val, lab_val, qids_val, dids_val, FLAGS.list_size_test)
    ranking_lists_test, all_labels_test, rl_lengths_test, resp_qids_test, resp_dids_test = \
        group_data_in_ranking_lists(docs_test, lab_test, qids_test, dids_test, FLAGS.list_size_test)

    # if FLAGS.load_proc_data:
    #     print('dumping data')
    #     save_model(((ranking_lists_train, all_labels_train, rl_lengths_train, resp_dids_train),
    #                 (ranking_lists_test, all_labels_test, rl_lengths_test, resp_dids_test, resp_qids_test),
    #                 (ranking_lists_val, all_labels_val, rl_lengths_val, resp_dids_val, resp_qids_val),
    #                 (np.array(lab_val, dtype=np.float32), np.array(lab_test, dtype=np.float32),
    #                  np.array(qids_val, dtype=np.float32), np.array(qids_test, dtype=np.float32))),
    #                data_fpath)
    # else:
    #     print('loading data')
    #     (ranking_lists_train, all_labels_train, rl_lengths_train, resp_dids_train), \
    #     (ranking_lists_test, all_labels_test, rl_lengths_test, resp_dids_test, resp_qids_test), \
    #     (ranking_lists_val, all_labels_val, rl_lengths_val, resp_dids_val, resp_qids_val), \
    #     (lab_val, lab_test, qids_val, qids_test) = load_model(data_fpath)

    all_labels_train, ranking_lists_train, rl_lengths_train, resp_dids_train = remove_queries_without_rel_docs(
        all_labels_train, ranking_lists_train, rl_lengths_train, resp_dids_train)
    if FLAGS.simulate_labels:
        # artif_labels = compute_simulated_labels(ranking_lists_train, rl_lengths_train, all_labels_train)
        artif_labels = sample_labels(all_labels_train, rl_lengths_train, FLAGS.n_binomial_samples)
        compare_artif_rj_with_real_ones(artif_labels, all_labels_train, rl_lengths_train)
        all_labels_train = artif_labels

        all_labels_train, ranking_lists_train, rl_lengths_train, resp_dids_train = remove_queries_without_rel_docs(
            all_labels_train, ranking_lists_train, rl_lengths_train, resp_dids_train)

    # avg_n_rel_docs = np.mean([np.sum([1 for rj in rl if rj > 0]) for rl in all_labels_train])
    # print('avg number of relevant documents per ranked list in training data: {}'.format(avg_n_rel_docs))

    if FLAGS.expand_training_data:
        ranking_lists_train, all_labels_train, rl_lengths_train = augment_training_data(ranking_lists_train,
                                                                                        all_labels_train,
                                                                                        rl_lengths_train)
        all_labels_train, ranking_lists_train, rl_lengths_train, resp_dids_train = remove_queries_without_rel_docs(
            all_labels_train, ranking_lists_train, rl_lengths_train, resp_dids_train)
    else:
        FLAGS.list_size_train = FLAGS.list_size_test

    return ranking_lists_train, all_labels_train, rl_lengths_train, resp_dids_train, \
           ranking_lists_val, all_labels_val, rl_lengths_val, resp_dids_val, resp_qids_val, \
           ranking_lists_test, all_labels_test, rl_lengths_test, resp_dids_test, resp_qids_test, \
           lab_val, lab_test, qids_val, qids_test


def augment_training_data(training_docs, training_rj, rl_lengths):
    training_rj = np.array(training_rj)
    rl_lengths = np.array(rl_lengths)
    n_samples_per_rl = 5
    new_ranked_lists = []
    new_rj = []
    new_lengths = []
    for i in range(len(training_docs)):
        docs_to_sample = np.array(training_docs[i][:rl_lengths[i]])
        for _ in range(n_samples_per_rl):
            sel_indices = np.random.choice([idx for idx in range(len(docs_to_sample))], size=FLAGS.list_size_train,
                                           replace=True)
            new_ranked_lists.append(docs_to_sample[sel_indices])
            new_rj.append(training_rj[i][sel_indices])
            new_lengths.append(FLAGS.list_size_train)
    return new_ranked_lists, new_rj, new_lengths


def load_lambdaMART_preds(fold_f, lambdamart_preds_path):
    """
    Fold	Training.txt	Validation.txt Test.txt
    Fold1	S1, S2, S3	S4		S5
    Fold2	S2, S3, S4	S5		S1
    Fold3	S3, S4, S5	S1		S2
    Fold4	S4, S5, S1	S2		S3
    Fold5	S5, S1, S2	S3		S4
    """
    test_preds_path = os.path.join(lambdamart_preds_path, FLAGS.coll_name + '_lightgbm_' + fold_f + '.hkl')
    if not os.path.isfile(test_preds_path):
        compute_lambdamart_preds(FLAGS)
    test_preds = load_model(test_preds_path)
    training_folds = []
    validation_folds = []
    if fold_f == 'Fold1':
        training_folds = ['Fold2', 'Fold3', 'Fold4']
        validation_folds = ['Fold5']
    elif fold_f == 'Fold2':
        training_folds = ['Fold3', 'Fold4', 'Fold5']
        validation_folds = ['Fold1']
    elif fold_f == 'Fold3':
        training_folds = ['Fold4', 'Fold5', 'Fold1']
        validation_folds = ['Fold2']
    elif fold_f == 'Fold4':
        training_folds = ['Fold5', 'Fold1', 'Fold2']
        validation_folds = ['Fold3']
    elif fold_f == 'Fold5':
        training_folds = ['Fold1', 'Fold2', 'Fold3']
        validation_folds = ['Fold4']

    training_preds = []
    for ff in training_folds:
        tmp_model_path = os.path.join(lambdamart_preds_path, FLAGS.coll_name + '_lightgbm_' + ff + '.hkl')
        training_preds.extend(load_model(tmp_model_path))
    val_preds_path = os.path.join(lambdamart_preds_path, FLAGS.coll_name + '_lightgbm_' + validation_folds[0] + '.hkl')
    val_preds = load_model(val_preds_path)
    return training_preds, test_preds, val_preds


def group_data_in_ranking_lists(vectors, labels, qids, dids, list_size):
    assert len(qids) == len(labels)
    assert len(qids) == len(vectors)
    data_indices_grouped_by_qid = {}
    for i in range(len(qids)):
        curr_qid = qids[i]
        if curr_qid not in data_indices_grouped_by_qid.keys():
            data_indices_grouped_by_qid[curr_qid] = [i]
        else:
            data_indices_grouped_by_qid[curr_qid].append(i)

    print('mean ranking list length: %2.4f' % np.mean([len(item) for item in data_indices_grouped_by_qid.values()]))
    print('max ranking list length: %2.4f' % np.max([len(item) for item in data_indices_grouped_by_qid.values()]))
    print('min ranking list length: %2.4f' % np.min([len(item) for item in data_indices_grouped_by_qid.values()]))

    ranking_lists = []
    all_labels = []
    rl_lengths = []
    resp_qids = []
    all_dids = []
    for qid, indices_group in data_indices_grouped_by_qid.items():
        curr_dids = [dids[i] for i in indices_group]
        vecs = [vectors[i] for i in indices_group]
        curr_labels = [labels[i] for i in indices_group]
        original_rl_len = len(curr_labels)
        # pad ranking lists now
        vecs = pad_list(vecs, list_size)
        curr_labels = curr_labels[0: min(list_size, len(curr_labels))]
        curr_labels = curr_labels + [0.0] * (list_size - len(curr_labels))
        resp_qids.append(qid)
        curr_dids = curr_dids[0: min(list_size, len(curr_dids))]
        curr_dids.extend('padding_did_{}'.format(i) for i in range(list_size - len(curr_dids)))

        # append to output values
        all_labels.append(curr_labels)
        ranking_lists.append(vecs)
        all_dids.append(curr_dids)
        rl_lengths.append(min(list_size, original_rl_len))

    return ranking_lists, all_labels, rl_lengths, resp_qids, all_dids


def group_rj_in_ranking_lists_no_pad_trim(qids, labs):
    ranking_lists = {}
    for i in range(len(qids)):
        qid = qids[i]
        label = labs[i]
        if qid in ranking_lists.keys():
            ranking_lists[qid].append(label)
        else:
            ranking_lists[qid] = [label]
    doc_scores = []
    doc_rj = []
    for k, ranking_list in ranking_lists.items():
        curr_scores = []
        curr_rj = []
        for i in range(len(ranking_list)):
            curr_rj.append(ranking_list[i])
        doc_scores.append(curr_scores)
        doc_rj.append(curr_rj)
    return doc_rj


def compute_ranking_lists_rl_length_masks(rl_lengths, list_size):
    rl_masks = []
    for i in range(len(rl_lengths)):
        curr_v = np.zeros(list_size)
        for j in range(min(len(curr_v), rl_lengths[i])):
            curr_v[j] = 1
        rl_masks.append(curr_v)
    return rl_masks


def remove_training_rl_without_rel_docs(train_rj, train_docs, rl_lengths_train):
    indices_to_remove_train = []
    for i in range(len(train_rj)):
        if max(train_rj[i]) == 0:
            indices_to_remove_train.append(i)

    train_rj = [train_rj[i] for i in range(len(train_rj)) if i not in indices_to_remove_train]
    train_docs = [train_docs[i] for i in range(len(train_docs)) if i not in indices_to_remove_train]
    rl_lengths_train = [rl_lengths_train[i] for i in range(len(rl_lengths_train)) if i not in indices_to_remove_train]
    return train_rj, train_docs, rl_lengths_train


def test_model(sess, model, model_path, test_rj, test_docs, rl_lengths, qids_test, labels_test_non_grouped,
               silent=False):
    rl_test_masks = compute_ranking_lists_rl_length_masks(rl_lengths, FLAGS.list_size_test)

    # initialize graph and session
    # tf.reset_default_graph()
    # sess_config = tf.ConfigProto()
    # sess_config.gpu_options.allow_growth = True
    # sess = tf.Session(config=sess_config, graph=tf.get_default_graph())

    # initialize model
    # model = ReRanker(FLAGS.seed, FLAGS.learning_rate, det_model=FLAGS.det_model, n_heads=FLAGS.n_heads,
    #                  num_features=FLAGS.num_features, n=FLAGS.n_binomial_samples,
    #                  loss_fn=FLAGS.loss, list_size=FLAGS.list_size_train, max_label_value=4,
    #                  norm_labels=FLAGS.norm_labels)
    tf.set_random_seed(FLAGS.seed)
    # sess.run(model.init_op)
    model.saver.restore(sess, model_path)
    sess.graph.finalize()
    # compute_predictions
    msamples = 50
    if FLAGS.det_model:
        msamples = 1
    all_preds = np.zeros(shape=(msamples, len(test_docs), FLAGS.list_size_test))
    for k in range(msamples):
        scores = sess.run(model.logits,
                          {model.training: False, model.input_docs: test_docs, model.rl_lengths_mask: rl_test_masks})
        if FLAGS.loss == 'ML':
            all_preds[k] = np.argmax(scores, axis=-1)
        else:
            all_preds[k] = scores

    avg_preds = np.mean(all_preds, axis=0)
    var_preds = np.var(all_preds, axis=0)

    for i in range(len(rl_test_masks)):
        for j in range(len(rl_test_masks[i])):
            if rl_test_masks[i][j] == 0:
                rl_test_masks[i][j] = 0  # -np.inf
            else:
                rl_test_masks[i][j] = 0
    avg_preds = rl_test_masks + avg_preds
    var_preds = rl_test_masks + var_preds

    grouped_rj = group_rj_in_ranking_lists_no_pad_trim(qids_test, labels_test_non_grouped)
    ideal_rel_j_lists = [np.array(rl)[np.argsort(-np.array(rl))] for rl in grouped_rj]
    ndcg_1, base_1 = compute_mean_ndcg(avg_preds, test_rj, ideal_rel_j_lists, 1)

    return avg_preds, ndcg_1, var_preds, compute_perf_metrics(avg_preds, test_rj, ideal_rel_j_lists, silent, rl_lengths)


def get_batches(all_docs, all_labels, rl_lengths_mask):
    db = []
    rb = []
    lb = []
    for i in range(len(all_docs)):
        db.append(all_docs[i])
        rb.append(all_labels[i])
        lb.append(rl_lengths_mask[i])
        if len(db) == FLAGS.batch_size:
            yield db, rb, lb
            db = []
            rb = []
            lb = []
    if len(db) > 0:
        yield db, rb, lb


def train_model(sess, model, train_docs, train_rj, rl_train_masks, test_rj, test_docs, rl_lengths_test,
                labels_test_non_grouped, qids_test, model_suffix):
    ckpt_paths = []
    perfs = []
    max_patience = 20
    patience = 20
    ploss = None
    early_stopping = False
    for epoch in range(1, FLAGS.num_epochs + 1):
        if early_stopping:
            break
        print('*** EPOCH: %d/%d' % (epoch, FLAGS.num_epochs))
        start = time.time()
        for db, rjb, lenb in get_batches(train_docs, train_rj, rl_train_masks):
            _, step, loss = sess.run(
                [model.train_op, model.global_step, model.loss],
                feed_dict={model.input_docs: db,
                           model.relevance_judgments: rjb,
                           model.rl_lengths_mask: lenb,
                           model.training: True})
            if ploss is None:
                ploss = loss
            else:
                if loss >= ploss:
                    patience -= 1
                    if patience == 0:
                        early_stopping = True
                        print('early stopping')
                        break
                else:
                    patience = max_patience
            if step % 50 == 0:
                end = time.time()
                print('step: %d, loss: %2.6f, time: %2.3fs' % (step, loss, (end - start)))
        step = sess.run(model.global_step)
        # save_path = model.saver.save(sess, os.path.join(FLAGS.model_ckpt_path, 'ckpt_' + model_suffix),
        #                              global_step=step)
        for _ in range(100):
            try:
                save_path = model.saver.save(sess, os.path.join(FLAGS.model_ckpt_path, 'ckpt_' + model_suffix),
                                             global_step=step)
            except:
                print('exception, retrying')
                continue
            break

        print("Model saved in path: %s" % save_path)
        preds, ndcg_1, var_preds, _ = test_model(sess, model, save_path, test_rj, test_docs, rl_lengths_test, qids_test,
                                                 labels_test_non_grouped, silent=False)

        perfs.append(ndcg_1)
        ckpt_paths.append(save_path)

    return ckpt_paths, perfs


def train_eval_model(train_rj, train_docs, test_rj, test_docs, rl_lengths_train, rl_lengths_test,
                     labels_test_non_grouped, qids_test, model_suffix=str(uuid.uuid4())):
    rl_train_masks = compute_ranking_lists_rl_length_masks(rl_lengths_train, FLAGS.list_size_train)
    print('max ranking list length in training data: %d' % max(rl_lengths_train))
    print('max ranking list length in test data: %d' % max(rl_lengths_test))

    # initialize graph and session
    tf.reset_default_graph()
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config, graph=tf.get_default_graph())

    # initialize model
    model = ReRanker(FLAGS.seed, FLAGS.learning_rate, coll_name=FLAGS.coll_name, det_model=FLAGS.det_model,
                     n_heads=FLAGS.n_heads,
                     num_features=FLAGS.num_features, n=FLAGS.n_binomial_samples,
                     loss_fn=FLAGS.loss, list_size=FLAGS.list_size_train, max_label_value=4,
                     norm_labels=FLAGS.norm_labels)
    tf.set_random_seed(FLAGS.seed)
    sess.run(model.init_op)
    sess.graph.finalize()
    start_training = time.time()
    ckpt_paths, perfs = train_model(sess, model, train_docs, train_rj, rl_train_masks, test_rj, test_docs,
                                    rl_lengths_test, labels_test_non_grouped, qids_test, model_suffix)
    print('Model trained in: %2.4fs' % (time.time() - start_training))

    # load and evaluate best model
    best_model_path = ckpt_paths[np.argmax(perfs)]
    print('Best ckpt model path: %s' % best_model_path)
    return best_model_path, sess, model


def run():
    fold_folders = ['Fold1', 'Fold2', 'Fold3', 'Fold4', 'Fold5']
    # fold_folders = ['Fold1']
    all_preds = []
    all_rjs = []
    all_qids_test = []
    all_qids_test_non_g = []
    all_dids_test = []
    all_lab_test_non_grouped = []
    all_rl_lengths = []
    perfs_across_folds = {}
    for fold_f in fold_folders:
        ranking_lists_train, all_labels_train, rl_lengths_train, resp_dids_train, \
        ranking_lists_val, all_labels_val, rl_lengths_val, resp_dids_val, resp_qids_val, \
        ranking_lists_test, all_labels_test, rl_lengths_test, resp_dids_test, resp_qids_test, \
        lab_val_non_grouped, lab_test_non_grouped, qids_val, qids_test = read_data(data_folder=FLAGS.data_folder,
                                                                                   fold_f=fold_f)
        # print(qids_test)
        best_model_path, sess, model = train_eval_model(all_labels_train, ranking_lists_train, all_labels_val,
                                                        ranking_lists_val,
                                                        rl_lengths_train, rl_lengths_val, lab_val_non_grouped, qids_val)
        avg_preds, ndcg_1, var_preds, all_perf = test_model(sess, model, best_model_path, all_labels_test,
                                                            ranking_lists_test, rl_lengths_test, qids_test,
                                                            lab_test_non_grouped)
        all_preds.extend(avg_preds)
        all_rjs.extend(all_labels_test)
        all_qids_test.extend(resp_qids_test)
        all_qids_test_non_g.extend(qids_test)
        all_dids_test.extend(resp_dids_test)
        all_lab_test_non_grouped.extend(lab_test_non_grouped)
        all_rl_lengths.extend(rl_lengths_test)

        for k, v in all_perf.items():
            if k in perfs_across_folds.keys():
                perfs_across_folds[k].append(v)
            else:
                perfs_across_folds[k] = [v]

    for k, v in perfs_across_folds.items():
        print('{}: {}'.format(k, np.mean(v)))
    # save_model((all_preds, all_rjs, all_qids_test, all_dids_test, all_qids_test_non_g, all_lab_test_non_grouped),
    #            './output/final_preds_data_{}_{}_{}.hkl'.format(FLAGS.coll_name, FLAGS.loss, FLAGS.simulate_labels))
    grouped_rj = group_rj_in_ranking_lists_no_pad_trim(all_qids_test_non_g, all_lab_test_non_grouped)
    ideal_rel_j_lists = [np.array(rl)[np.argsort(-np.array(rl))] for rl in grouped_rj]
    all_rjs = np.array(all_rjs) * int(1.0 / sorted(set(all_lab_test_non_grouped))[1])
    ideal_rel_j_lists = np.array(ideal_rel_j_lists) * int(1.0 / sorted(set(all_lab_test_non_grouped))[1])
    print('\nFINAL PERF AVGD ACROSS FOLDS:')
    # import pdb
    # pdb.set_trace()
    compute_perf_metrics(all_preds, all_rjs, ideal_rel_j_lists, False, all_rl_lengths, max_rj=2.0)
    create_trec_eval_format_run_qrels(all_preds, all_dids_test, all_qids_test, all_rjs,
                                      'PR_{}_loss={}_simulate_labels={}_det_model={}'.format(FLAGS.coll_name,
                                                                                             FLAGS.loss,
                                                                                             FLAGS.simulate_labels,
                                                                                             FLAGS.det_model),
                                      './output')

    return


def create_trec_format_run(qids, dids, preds, ofpath):
    out = open(ofpath, 'w')
    for ranked_list_idx in range(len(preds)):
        sorted_indices = np.argsort(preds[ranked_list_idx])
        for item_idx in sorted_indices:
            run_line = '{} Q0 {} {} {} {}\n'.format(qids[ranked_list_idx], dids[ranked_list_idx][item_idx],
                                                    item_idx + 1, preds[ranked_list_idx][item_idx], 'PFusion')
            out.write(run_line)
    out.close()


def flatten_stuff_provide_fake_qids(all_preds, all_rjs):
    preds = []
    labels = []
    qids = []
    for i in range(len(all_preds)):
        preds.extend(all_preds[i])
        labels.extend(all_rjs[i])
        qids.extend([i] * len(all_preds[i]))
    return np.array(preds), np.array(labels), np.array(qids)


if __name__ == '__main__':
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    logging.getLogger("tensorflow").setLevel(logging.CRITICAL)

    arg_parser = argparse.ArgumentParser()
    add_arguments(arg_parser)
    FLAGS, unparsed = arg_parser.parse_known_args()
    for arg in vars(FLAGS):
        print(arg, ":", getattr(FLAGS, arg))

    if not os.path.exists(FLAGS.model_ckpt_path):
        os.makedirs(FLAGS.model_ckpt_path)
    np.random.seed(FLAGS.seed)
    tf.random.set_random_seed(FLAGS.seed)
    run()
    print(FLAGS.loss)
    print('DONE')
