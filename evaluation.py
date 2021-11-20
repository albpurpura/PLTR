import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from tqdm import tqdm

from globals import PADDING_PREFIX


def compute_idcg(rel_j_list, cutoff):
    rel_j_list = np.array(rel_j_list)
    rel_j_list = -np.sort(-rel_j_list)
    idcg = 0
    for i in range(len(rel_j_list)):
        rank = i + 1
        # idcg += (np.power(2, rel_j_list[i]) - 1) / (np.log2(rank + 1))
        idcg += rel_j_list[i] / (np.log2(rank + 1))
        if rank == cutoff:
            break
    return idcg


def compute_ndcg(returned_rel_j_list, ideal_rel_j_list, cutoff, alt_ndcg=False):
    dcg = 0
    for i in range(len(returned_rel_j_list)):
        rank = i + 1
        if returned_rel_j_list[i] > 0:
            if alt_ndcg:
                dcg += (np.power(2, returned_rel_j_list[i]) - 1) / (np.log2(rank + 1))
            else:
                dcg += returned_rel_j_list[i] / (np.log2(rank + 1))
        if rank == cutoff:
            break

    idcg = 0
    for i in range(len(ideal_rel_j_list)):
        rank = i + 1
        if ideal_rel_j_list[i] > 0:
            if alt_ndcg:
                idcg += (np.power(2, ideal_rel_j_list[i]) - 1) / (np.log2(rank + 1))
            else:
                idcg += ideal_rel_j_list[i] / (np.log2(rank + 1))
        if rank == cutoff:
            break
    if idcg == 0:
        return 0
    return dcg / idcg


def compute_mean_ndcg(test_preds, test_rj, ideal_rel_j_lists, cutoff):
    baseline = compute_mean_ndcg_no_reranking(test_rj, ideal_rel_j_lists, cutoff)
    ndcg_by_rl = []
    for i in range(len(test_preds)):
        rel_j = np.array(test_rj[i])
        predictions = np.array(test_preds[i])
        predictions_rel_j = rel_j[np.argsort(-predictions)]
        # sorted_indices = np.argsort(-rel_j)
        # ideal_ranking_rj = rel_j[sorted_indices]
        curr_ndcg = compute_ndcg(predictions_rel_j, ideal_rel_j_lists[i], cutoff)
        ndcg_by_rl.append(curr_ndcg)
    return np.mean(ndcg_by_rl), baseline


def get_ranks(values):
    ranks = np.zeros(len(values))
    new_indices = np.argsort(-values)
    for i in range(len(values)):
        ranks[new_indices[i]] = i
    return ranks


def ndcg_from_ranks(pred_ranks, rj):
    all_ndcg_values = []
    for i in range(len(pred_ranks)):
        dcgs = []
        idcgs = []
        curr_pred_ranks = pred_ranks[i]
        curr_best_ranks = get_ranks(rj[i])
        for j in range(len(curr_pred_ranks)):
            dcg = (np.power(2, rj[i, j]) - 1) / np.log2((curr_pred_ranks[j] + 2))
            idcg = (np.power(2, rj[i, j]) - 1) / np.log2((curr_best_ranks[j] + 2))
            dcgs.append(dcg)
            idcgs.append(idcg)
        ndcg = 0
        idcg = np.sum(idcgs)
        dcg = np.sum(dcgs)
        if idcg > 0:
            ndcg = dcg / idcg
        all_ndcg_values.append(ndcg)
    return all_ndcg_values


def compute_MAP(test_preds, test_rj):
    all_avg_precs = []
    for i in range(len(test_preds)):
        curr_preds = np.array(test_preds[i])
        curr_rj = np.array(test_rj[i])

        sorted_indices = np.argsort(-curr_preds)
        sorted_preds = curr_preds[sorted_indices]
        sorted_rj = curr_rj[sorted_indices]
        avg_prec = 0
        n_rel_docs = 0
        for j in range(len(sorted_preds)):
            if sorted_rj[j] > 0:
                n_rel_docs += 1
                avg_prec += (n_rel_docs / (j + 1))
        if n_rel_docs == 0:
            avg_prec = 0  # undefined in this case
        else:
            avg_prec /= n_rel_docs
        all_avg_precs.append(avg_prec)
    return np.mean(all_avg_precs)


def compute_P_at_k(test_preds, test_rj, k):
    all_avg_precs = []
    for i in range(len(test_preds)):
        curr_preds = np.array(test_preds[i])
        curr_rj = np.array(test_rj[i])

        sorted_indices = np.argsort(-curr_preds)
        sorted_preds = curr_preds[sorted_indices]
        sorted_rj = curr_rj[sorted_indices]
        n_docs_seen = 0
        n_rel_docs = 0
        for j in range(min(k, len(sorted_preds))):
            n_docs_seen += 1
            if sorted_rj[j] > 0:
                n_rel_docs += 1
        avg_prec = n_rel_docs / n_docs_seen
        all_avg_precs.append(avg_prec)
    return np.mean(all_avg_precs)


def compute_mean_ndcg_no_reranking(test_rj, ideal_rel_j_lists, cutoff):
    ndcg_by_rl = []
    for i in range(len(test_rj)):
        rel_j = np.array(test_rj[i])
        # sorted_indices = np.argsort(-rel_j)
        # ideal_ranking_rj = rel_j[sorted_indices]
        curr_ndcg = compute_ndcg(rel_j, ideal_rel_j_lists[i], cutoff)
        ndcg_by_rl.append(curr_ndcg)
    return np.mean(ndcg_by_rl)


def create_trec_eval_format_run_qrels_alt(preds, resp_dids, resp_qids, resp_rel_j, all_dids, all_qids, all_rjs,
                                          run_name, output_dir):
    run_out = open(os.path.join(output_dir, 'run-{}.txt'.format(run_name)), 'w')
    qrels_out = open(os.path.join(output_dir, 'qrels-{}.txt'.format(run_name)), 'w')
    preds = np.array(preds)
    # rel_j = np.array(resp_rel_j)
    # import pdb
    # pdb.set_trace()
    for i in range(len(preds)):
        qid = resp_qids[i][0]
        sorted_doc_indices = np.argsort(-preds[i])
        curr_docs = preds[i]
        pos = 1
        for doc_idx in sorted_doc_indices:
            if doc_idx < len(resp_dids[i]):
                did = resp_dids[i][doc_idx]
                if str(did).startswith(PADDING_PREFIX):
                    continue
                # pos = sorted_doc_indices[doc_idx] + 1
                # pos = doc_idx + 1
                score = curr_docs[doc_idx]
                # label = curr_rj[doc_idx]
                run_line = '{} Q0 {} {} {} {}\n'.format(qid, did, pos, score, run_name)
                # qrels_line = '{} 0 {} {}\n'.format(qid, did, label)
                pos += 1
                run_out.write(run_line)
                # qrels_out.write(qrels_line)
    run_out.close()

    for i in range(len(all_dids)):
        did = all_dids[i]
        if str(did).startswith(PADDING_PREFIX):
            continue
        qid = all_qids[i]
        label = all_rjs[i]
        qrels_line = '{} 0 {} {}\n'.format(qid, did, label)
        qrels_out.write(qrels_line)
    qrels_out.close()


def create_trec_eval_format_run_qrels(preds, resp_dids, resp_qids, rel_j, run_name, output_dir):
    run_out = open(os.path.join(output_dir, 'run-{}.txt'.format(run_name)), 'w')
    qrels_out = open(os.path.join(output_dir, 'qrels-{}.txt'.format(run_name)), 'w')
    preds = np.array(preds)
    rel_j = np.array(rel_j)
    for i in range(len(preds)):
        qid = resp_qids[i]
        sorted_doc_indices = np.argsort(-np.array(preds[i]))
        curr_docs = preds[i]
        curr_rj = rel_j[i]
        # curr_docs = preds[i][sorted_doc_indices]
        # curr_rj = rel_j[i][sorted_doc_indices]
        pos = 1
        for doc_idx in sorted_doc_indices:
            if doc_idx < len(resp_dids[i]):
                did = resp_dids[i][doc_idx]
                if str(did).startswith(PADDING_PREFIX):
                    continue
                # pos = sorted_doc_indices[doc_idx] + 1
                # pos = doc_idx + 1
                score = curr_docs[doc_idx]
                label = curr_rj[doc_idx]
                run_line = '{} Q0 {} {} {} {}\n'.format(qid, did, pos, score, run_name)
                qrels_line = '{} 0 {} {}\n'.format(qid, did, label)
                pos += 1
                run_out.write(run_line)
                qrels_out.write(qrels_line)
    run_out.close()
    qrels_out.close()


def get_idcg(scores, labels, cutoff=None):
    scores = np.array(scores)
    labels = np.array(labels)
    sorted_indices = np.argsort(-scores)
    sorted_labels = labels[sorted_indices]
    gain = 0
    for i in range(len(sorted_labels)):
        # gain += ((np.power(2, sorted_labels[i]) - 1) / np.log2(i + 2))
        gain += (sorted_labels[i] / np.log2(i + 2))  # this is the version used in trec eval
        if i + 1 == cutoff:
            break
    # normalize gain
    ideal_sorted_labels = labels[np.argsort(-labels)]
    discount = 0
    for i in range(len(ideal_sorted_labels)):
        discount += (ideal_sorted_labels[i] / np.log2(i + 2))  # this is the version used in trec eval
        if i + 1 == cutoff:
            break
    if discount == 0:
        return 0
    return gain / discount


def get_mean_ndcg(preds, labels, cutoff):
    ndcg_values = []
    for i in range(len(preds)):
        ndcg_values.append(get_idcg(preds[i], labels[i], cutoff))
    return np.mean(ndcg_values)


def compute_avg_groups_representation(runfile, pos_thr=None):
    docs_prefixes = []
    for line in open(runfile):
        data = line.split()
        did = data[2]
        if not did.startswith('['):
            docs_prefixes.append(did[:2])

    all_prefs = set(docs_prefixes)

    aggr_counts = {pref: 0 for pref in all_prefs}
    aggr_counts_total = {pref: 0 for pref in all_prefs}
    repr_docs_freqs_by_qid = {}
    for line in open(runfile):
        data = line.split()
        pref = data[2][:2]
        qid = int(data[0])
        if pos_thr is None or int(data[3]) <= pos_thr:
            if pref in all_prefs:
                aggr_counts[pref] += 1
                if qid not in repr_docs_freqs_by_qid.keys():
                    repr_docs_freqs_by_qid[qid] = {pref: 0 for pref in all_prefs}
                repr_docs_freqs_by_qid[qid][pref] += 1
        if pref in all_prefs:
            aggr_counts_total[pref] += 1

    normalized_counts = {k: aggr_counts[k] / aggr_counts_total[k] for k in aggr_counts_total.keys()}

    return repr_docs_freqs_by_qid, aggr_counts, aggr_counts_total, normalized_counts


def compute_repr_index(runfile, pos_thr=None):
    docs_prefixes = []
    for line in open(runfile):
        data = line.split()
        did = data[2]
        if not did.startswith('['):
            docs_prefixes.append(did[:2])

    all_prefs = set(docs_prefixes)
    # print(all_prefs)
    repr_docs_freqs_by_qid = {}
    repr_docs_freqs_seqs_by_qid = {}
    aggr_counts_total = {p: 0 for p in all_prefs}
    for line in open(runfile):
        data = line.split()
        pref = data[2][:2]
        qid = int(data[0])
        if pos_thr is None or int(data[3]) <= pos_thr:
            if pref in all_prefs:
                aggr_counts_total[pref] += 1
                if qid not in repr_docs_freqs_by_qid.keys():
                    repr_docs_freqs_by_qid[qid] = {pref: 0 for pref in all_prefs}
                    repr_docs_freqs_seqs_by_qid[qid] = []
                repr_docs_freqs_by_qid[qid][pref] += 1
                repr_docs_freqs_seqs_by_qid[qid].append(sum([1 for v in repr_docs_freqs_by_qid[qid].values() if v > 0]))
        if pref in all_prefs:
            if qid not in aggr_counts_total.keys():
                aggr_counts_total[qid] = {pref: 0 for pref in all_prefs}
            aggr_counts_total[qid][pref] += 1

    for qid in repr_docs_freqs_seqs_by_qid.keys():
        # divide by the total number of repr groups in a ranked list
        repr_docs_freqs_seqs_by_qid[qid] = np.array(repr_docs_freqs_seqs_by_qid[qid]) / sum(
            [1 for v in aggr_counts_total[qid].values() if v > 0])

    return repr_docs_freqs_seqs_by_qid


def compute_ERR(preds, trues, true_rl_lengths, max_rj):
    # max_rj = 2
    if not np.min(trues) >= 0 and np.max(trues) <= max_rj:
        trues = np.array(trues) * max_rj
    assert np.min(trues) >= 0 and np.max(trues) <= max_rj
    all_errs = []
    for ranked_list_index in range(len(preds)):
        curr_preds = preds[ranked_list_index]
        curr_trues = trues[ranked_list_index]
        sorted_indices = np.argsort(-curr_preds)[:true_rl_lengths[ranked_list_index]]
        sorted_labels = np.array(curr_trues)[sorted_indices]
        all_errs.append(ERR(sorted_labels, max_rj))
    return np.mean(all_errs)


def ERR(ranked_rj, max_rj):
    def rfunc(value, max_rj):
        return (np.power(2, value) - 1) / np.power(2, max_rj)

    p = 1
    rval = 0
    for r in range(1, len(ranked_rj) + 1):
        idx = r - 1
        R = rfunc(ranked_rj[idx], max_rj)
        rval += p * R / r
        p = p * (1 - R)
    return rval


def compute_perf_metrics(avg_preds, test_rj, ideal_rel_j_lists, silent, rl_lengths, max_rj=2):
    ndcg_1, base1 = compute_mean_ndcg(avg_preds, test_rj, ideal_rel_j_lists, 1)
    ndcg_3, base3 = compute_mean_ndcg(avg_preds, test_rj, ideal_rel_j_lists, 3)
    ndcg_5, base5 = compute_mean_ndcg(avg_preds, test_rj, ideal_rel_j_lists, 5)
    ndcg_10, base10 = compute_mean_ndcg(avg_preds, test_rj, ideal_rel_j_lists, 10)
    map_v = compute_MAP(avg_preds, test_rj)
    p_1 = compute_P_at_k(avg_preds, test_rj, 1)
    p_3 = compute_P_at_k(avg_preds, test_rj, 3)
    p_5 = compute_P_at_k(avg_preds, test_rj, 5)
    err = compute_ERR(avg_preds, test_rj, rl_lengths, max_rj)
    if not silent:
        print('ndcg@1: %2.4f (input: %2.4f)' % (ndcg_1, base1))
        print('ndcg@3: %2.4f (input: %2.4f)' % (ndcg_3, base3))
        print('ndcg@5: %2.4f (input: %2.4f)' % (ndcg_5, base5))
        print('ndcg@10: %2.4f (input: %2.4f)' % (ndcg_10, base10))
        print('MAP: %2.4f' % map_v)
        print('P@1: %2.4f' % p_1)
        print('P@3: %2.4f' % p_3)
        print('P@5: %2.4f' % p_5)
        print('ERR: %2.4f' % err)
    return {'ndcg@1': ndcg_1, 'ndcg@3': ndcg_3, 'ndcg@5': ndcg_5, 'ndcg@10': ndcg_10, 'MAP': map_v, 'P@1': p_1,
            'P@3': p_3, 'P@5': p_5, 'ERR': err}


def compute_group_fairness_tests():
    fpaths = ['./output/run-PR_MLIA_consider_raw_rj_dists=True_loss=ApproxNDCG_det_model=False.txt',
              './output/run-PR_MLIA_consider_raw_rj_dists=True_loss=KL_G_det_model=False.txt',
              './output/run-PR_MLIA_consider_raw_rj_dists=True_loss=KL_B_det_model=False.txt',
              './output/run-PR_MLIA_consider_raw_rj_dists=True_loss=ML_det_model=False.txt']

    for pos_thr in [150]:
        avgd_repr_docs_freqs_seq = []
        for fpath in fpaths:
            repr_docs_freqs_by_qid, aggr_counts, aggr_counts_total, normalized_counts = compute_avg_groups_representation(
                fpath, pos_thr=pos_thr)
            repr_docs_freqs_seq_by_qid = compute_repr_index(fpath)
            sequences = [list(v[:min(len(v), pos_thr)]) for v in repr_docs_freqs_seq_by_qid.values()]
            padded_seqs = [v + [v[-1]] * (pos_thr - len(v)) for v in sequences]
            print('run={}, pos_thr={}: {} (normalized: {})'.format(fpath, pos_thr, aggr_counts, normalized_counts))
            avgd_repr_docs_freqs_seq.append(np.mean(np.array(padded_seqs), axis=0))
        fig, ax = plt.subplots()
        for i in range(len(fpaths)):
            ax.plot(avgd_repr_docs_freqs_seq[i])
        ax.set(xlabel='position', ylabel='% represented groups', title='thr={}'.format(pos_thr))
        ax.grid()
        plt.legend(fpaths)
        plt.show()


def load_trec_format_run(qrels, run):
    test_rj = []
    ideal_rel_j_lists = []
    rl_lengths = []
    preds = []

    grouped_preds_by_qid = {}
    grouped_qrels_by_qid = {}

    for line in open(qrels):
        qid, _, did, rj = line.split()
        if qid in grouped_qrels_by_qid.keys():
            grouped_qrels_by_qid[qid][did] = float(rj)
        else:
            grouped_qrels_by_qid[qid] = {did: float(rj)}
    # print('opening: {}'.format(run))
    for line in open(run):
        # print(line)
        qid, _, did, rank, score, run_name = line.split()
        if qid in grouped_preds_by_qid.keys():
            grouped_preds_by_qid[qid][did] = float(score)
        else:
            grouped_preds_by_qid[qid] = {did: float(score)}

    for k in grouped_qrels_by_qid.keys():
        did_rel_pairs = grouped_qrels_by_qid[k]
        did_score_pairs = grouped_preds_by_qid[k]
        curr_rs = []
        curr_rj = []
        curr_ideal_rel_j_list = []
        curr_rl_len = 0
        for did in did_rel_pairs.keys():
            curr_d_rj = did_rel_pairs[did]
            curr_ideal_rel_j_list.append(curr_d_rj)

        for did in did_score_pairs.keys():
            if not did.startswith(PADDING_PREFIX):
                curr_rl_len += 1

                if did in did_rel_pairs.keys():
                    curr_d_rj = did_rel_pairs[did]
                else:
                    curr_d_rj = 0.0

                curr_rs.append(did_score_pairs[did])
                curr_rj.append(curr_d_rj)

        preds.append(curr_rs)
        test_rj.append(curr_rj)
        rl_lengths.append(curr_rl_len)
        ideal_rel_j_lists.append(curr_ideal_rel_j_list)
    return preds, test_rj, ideal_rel_j_lists, rl_lengths


def compute_paired_t_test(baseline_qrels, baseline_run, other_qrels, other_run, coll='MLIA'):
    if coll == 'MLIA':
        max_rj = 1
    elif coll == 'MQ2007' or coll == 'MQ2008' or coll == 'OHSUMED':
        max_rj = 2
    elif coll == 'MSLR-WEB10K':
        # there will be a problem when computing the qrels of this dataset
        max_rj = 4
    else:
        max_rj = 1.0
    preds_base, test_rj_base, ideal_rel_j_lists_base, rl_lengths_base = load_trec_format_run(baseline_qrels,
                                                                                             baseline_run)

    preds_other, test_rj_other, ideal_rel_j_lists_other, rl_lengths_other = load_trec_format_run(other_qrels, other_run)

    perfs_by_q_base = compute_performance_measures_by_q(preds_base, test_rj_base, ideal_rel_j_lists_base,
                                                        rl_lengths_base, max_rj)
    perfs_by_q_other = compute_performance_measures_by_q(preds_other, test_rj_other, ideal_rel_j_lists_other,
                                                         rl_lengths_other, max_rj)
    p_values = {}
    for m in perfs_by_q_base.keys():
        tstat, pvalue = stats.ttest_rel(perfs_by_q_base[m], perfs_by_q_other[m])
        p_values['pvalue_' + m] = pvalue
    return p_values


def compute_performance_measures_by_q(preds, test_rj, ideal_rel_j_lists, rl_lengths, max_rj, alt_ndcg=False):
    ndcg_1 = compute_ndcg_by_q(preds, test_rj, ideal_rel_j_lists, 1, alt_ndcg)
    ndcg_3 = compute_ndcg_by_q(preds, test_rj, ideal_rel_j_lists, 3, alt_ndcg)
    ndcg_5 = compute_ndcg_by_q(preds, test_rj, ideal_rel_j_lists, 5, alt_ndcg)
    ndcg_10 = compute_ndcg_by_q(preds, test_rj, ideal_rel_j_lists, 10, alt_ndcg)
    p_1 = compute_P_at_k_by_q(preds, test_rj, 1)
    p_3 = compute_P_at_k_by_q(preds, test_rj, 3)
    p_5 = compute_P_at_k_by_q(preds, test_rj, 5)
    p_10 = compute_P_at_k_by_q(preds, test_rj, 10)
    map = compute_MAP_by_q(preds, test_rj, ideal_rel_j_lists)
    err = compute_ERR_by_q(np.array(preds), np.array(test_rj), rl_lengths, max_rj)
    return {'ndcg@1': ndcg_1, 'ndcg@3': ndcg_3, 'ndcg@5': ndcg_5, 'ndcg@10': ndcg_10, 'P@1': p_1, 'P@3': p_3,
            'P@5': p_5, 'P@10': p_10, 'ERR': err, 'MAP': map}


def compute_MAP_by_q(test_preds, test_rj, ideal_rel_j_lists):
    all_avg_precs = []
    for i in range(len(test_preds)):
        curr_preds = np.array(test_preds[i])
        curr_rj = np.array(test_rj[i])
        ideal_rj = ideal_rel_j_lists[i]
        true_nrel = sum([1 for item in ideal_rj if item > 0])
        sorted_indices = np.argsort(-curr_preds)
        sorted_preds = curr_preds[sorted_indices]
        sorted_rj = curr_rj[sorted_indices]
        avg_prec = 0
        n_rel_docs = 0
        for j in range(len(sorted_preds)):
            if sorted_rj[j] > 0:
                n_rel_docs += 1
                avg_prec += (n_rel_docs / (j + 1))
        if n_rel_docs == 0:
            avg_prec = 0  # undefined in this case
        else:
            avg_prec /= true_nrel
        all_avg_precs.append(avg_prec)
    return all_avg_precs


def compute_ndcg_by_q(test_preds, test_rj, ideal_rel_j_lists, cutoff, alt_ndcg):
    ndcg_by_rl = []
    for i in range(len(test_preds)):
        rel_j = np.array(test_rj[i])
        predictions = np.array(test_preds[i])
        predictions_rel_j = rel_j[np.argsort(-predictions)]
        curr_ndcg = compute_ndcg(predictions_rel_j, np.array(ideal_rel_j_lists[i])[np.argsort(-np.array(ideal_rel_j_lists[i]))], cutoff, alt_ndcg)
        ndcg_by_rl.append(curr_ndcg)
    return ndcg_by_rl


def compute_P_at_k_by_q(test_preds, test_rj, k):
    all_avg_precs = []
    for i in range(len(test_preds)):
        curr_preds = np.array(test_preds[i])
        curr_rj = np.array(test_rj[i])

        sorted_indices = np.argsort(-curr_preds)
        sorted_preds = curr_preds[sorted_indices]
        sorted_rj = curr_rj[sorted_indices]
        n_docs_seen = 0
        n_rel_docs = 0
        for j in range(min(k, len(sorted_preds))):
            n_docs_seen += 1
            if sorted_rj[j] > 0:
                n_rel_docs += 1
        avg_prec = n_rel_docs / n_docs_seen
        all_avg_precs.append(avg_prec)
    return all_avg_precs


def compute_ERR_by_q(preds, trues, true_rl_lengths, max_rj):
    # trues = np.array(trues) * max_rj
    assert np.min([np.min(t) for t in trues]) >= 0 and np.max([np.max(t) for t in trues]) <= max_rj
    all_errs = []
    for ranked_list_index in range(len(preds)):
        curr_preds = np.array(preds[ranked_list_index])
        curr_trues = np.array(trues[ranked_list_index])
        sorted_indices = np.argsort(-curr_preds)[:true_rl_lengths[ranked_list_index]]
        sorted_labels = np.array(curr_trues)[sorted_indices]
        all_errs.append(ERR(sorted_labels, max_rj))
    return all_errs


def re_evaluate(run, qrels, coll, silent=False, alt_ndcg=False):
    if coll == 'MLIA':
        max_rj = 1
    elif coll == 'MQ2007' or coll == 'MQ2008' or coll == 'OHSUMED':
        max_rj = 2
    elif coll == 'MSLR-WEB10K' or coll == 'MSLR-WEB30K':
        # there will be a problem when computing the qrels of this dataset
        max_rj = 4
    else:
        max_rj = 1.0
    preds, test_rj, ideal_rel_j_lists, rl_lengths = load_trec_format_run(qrels, run)
    ideal_rel_j_lists = [np.array(l)[np.argsort(-np.array(l))] for l in ideal_rel_j_lists]
    perfs_by_q_base = compute_performance_measures_by_q(preds, test_rj, ideal_rel_j_lists, rl_lengths, max_rj, alt_ndcg)
    # import pdb
    # pdb.set_trace()
    if not silent:
        print('RUN {}:\t{}'.format(run, ', '.join([k + ' : ' + str(np.mean(v))
                                                   for k, v in perfs_by_q_base.items()]).strip(', ')))
    return perfs_by_q_base
