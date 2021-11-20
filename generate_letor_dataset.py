import os

import numpy as np
import ujson
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm

qids_dids_labels_path = './MLIA/data/qids_dids_labels.jsonl'
docs_scores_by_qid_path = './MLIA/data/docs_scores_by_qid.jsonl'


def aggregate_prob_dist(probs_vec):
    aggr = np.array([probs_vec[0], (probs_vec[1] * 0.5 + probs_vec[2]) / 1.5])
    return aggr / np.sum(aggr)


def compute_aggregated_rel_score(aggregated_prob_vector):
    return (np.sum(np.array([-1, 0.5, 1]) * aggregated_prob_vector, axis=-1) + 1) / 2


def is_to_ignore(fname):
    kw_to_ignore = ['nostem', 'lmjm', 'nostop', 'upd_lucene']
    for k in kw_to_ignore:
        if k in fname:
            return True
    return False


def compute_data_MLIA(fold_f, consider_raw_rj_dists):
    if os.path.isfile(qids_dids_labels_path):
        qids_dids_labels = load_jsonl_data(qids_dids_labels_path)
        docs_scores_by_qid = load_jsonl_data(docs_scores_by_qid_path)
    else:
        runs_dir = '/Users/alberto/PycharmProjects/MLIA/base_runs/en'
        pool = '/Users/alberto/PycharmProjects/MLIA/pools/crowd/pool-en-crowd-MV.txt'

        fpaths = [os.path.join(runs_dir, f) for f in os.listdir(runs_dir) if
                  not f.startswith('.') and not is_to_ignore(f)]
        ndims = len(fpaths)
        print('NDIMS={}'.format(ndims))
        min_feature_values = np.ones(ndims)
        docs_scores_by_qid = {}
        for i in tqdm(range(len(fpaths))):
            for line in open(fpaths[i]):
                data = line.split()
                qid = int(data[0])
                did = data[2]
                dscore = float(data[4])
                if qid not in docs_scores_by_qid.keys():
                    docs_scores_by_qid[qid] = {}
                if did not in docs_scores_by_qid[qid]:
                    docs_scores_by_qid[qid][did] = [None] * ndims
                docs_scores_by_qid[qid][did][i] = dscore
                if min_feature_values[i] > dscore:
                    min_feature_values[i] = dscore

        # prune docs with none features in each query
        for qid in tqdm(docs_scores_by_qid.keys()):
            doc_vecs_by_id = docs_scores_by_qid[qid]
            all_dids = doc_vecs_by_id.keys()
            for did in list(all_dids):
                # fill nones with min feature value
                if None in doc_vecs_by_id[did] and doc_vecs_by_id[did].count(None) / len(doc_vecs_by_id[did]) < 0.2:
                    for feature_idx in range(len(doc_vecs_by_id[did])):
                        if doc_vecs_by_id[did][feature_idx] is None:
                            doc_vecs_by_id[did][feature_idx] = min_feature_values[feature_idx]
                if None in doc_vecs_by_id[did]:
                    doc_vecs_by_id.pop(did)
            docs_scores_by_qid[qid] = doc_vecs_by_id

        # read pool
        qids_dids_labels = {}
        for line in open(pool):
            data = line.split()
            qid = int(data[0])
            did = data[2]
            rj = float(data[3])
            if qid not in qids_dids_labels.keys():
                qids_dids_labels[qid] = {}
            if did not in qids_dids_labels[qid]:
                qids_dids_labels[qid][did] = rj
        np.random.seed(0)

        if consider_raw_rj_dists:
            qids_dids_labels = get_judgements_distribution()
        else:
            qids_dids_labels = get_judgements_distribution()
            for qid in qids_dids_labels.keys():
                for did in qids_dids_labels[qid].keys():
                    # qids_dids_labels[qid][did] = np.argmax(qids_dids_labels[qid][did], axis=-1)
                    # qids_dids_labels[qid][did] = np.sum(np.array([-1, 1, 2]) * qids_dids_labels[qid][did], axis=-1)
                    # new_rj_split = aggregate_prob_dist(qids_dids_labels[qid][did])
                    new_rj_split = qids_dids_labels[qid][did]
                    qids_dids_labels[qid][did] = compute_aggregated_rel_score(new_rj_split)

        dump_qids_dids_labels_and_doc_scores_by_qid(qids_dids_labels, docs_scores_by_qid)
    return get_data_per_fold(qids_dids_labels, docs_scores_by_qid, consider_raw_rj_dists, fold_f)


def load_jsonl_data(fpath):
    return ujson.loads(''.join([l for l in open(fpath)]))


def dump_qids_dids_labels_and_doc_scores_by_qid(qids_dids_labels, docs_scores_by_qid):
    out = open(qids_dids_labels_path, 'w')
    out.write(
        ujson.dumps({k: {did: list(v) for did, v in qids_dids_labels[k].items()} for k in qids_dids_labels.keys()}))
    # for item in qids_dids_labels:
    #     line = ujson.dumps(item)
    #     out.write(line + '\n')
    out.close()
    out = open(docs_scores_by_qid_path, 'w')
    out.write(ujson.dumps(docs_scores_by_qid))
    # for item in docs_scores_by_qid:
    #     line = ujson.dumps(item)
    #     out.write(line + '\n')
    out.close()


def get_data_per_fold(qids_dids_labels, docs_scores_by_qid, consider_raw_rj_dists, fold_f='Fold1'):
    print('5 splits')
    print('shuffle = False')
    all_qids = list(qids_dids_labels.keys())
    kf = KFold(n_splits=5, random_state=0, shuffle=True)
    for fold_idx, (train_qids_indices, test_qids_indices) in enumerate(kf.split(list(qids_dids_labels.keys()))):
        train_qids = np.array(all_qids)[train_qids_indices]
        test_qids = np.array(all_qids)[test_qids_indices]
        if 'Fold{}'.format(fold_idx + 1) == fold_f:
            train_qids, val_qids = train_test_split(train_qids, test_size=0.2, random_state=1)

            train_docs, train_labs, train_fl_qids, train_fl_dids = flatten_data(docs_scores_by_qid, qids_dids_labels,
                                                                                train_qids, consider_raw_rj_dists)
            test_docs, test_labs, test_fl_qids, test_fl_dids = flatten_data(docs_scores_by_qid, qids_dids_labels,
                                                                            test_qids, consider_raw_rj_dists)
            val_docs, val_labs, val_fl_qids, val_fl_dids = flatten_data(docs_scores_by_qid, qids_dids_labels, val_qids,
                                                                        consider_raw_rj_dists)
            print('test_qids: ' + str(test_qids))
            return train_docs, train_labs, train_fl_qids, train_fl_dids, test_docs, test_labs, test_fl_qids, test_fl_dids, val_docs, val_labs, val_fl_qids, val_fl_dids


def get_judgements_distribution():
    # fpath = '/Users/alberto/ExperimentalCollections/MLIA_qrels/qrels_it_merged.txt'
    fpath = '/Users/alberto/ExperimentalCollections/MLIA_qrels/qrels_en_merged.txt'
    judgements_conv_map = {'relevant': 2, 'partially relevant': 1, 'not relevant': 0}
    # this first reading step is supposed to remove duplicates
    rj_by_q = {}
    for line in open(fpath):
        data = line.split('\t')
        # lang = data[0]
        qid = int(data[1])
        did = data[2]
        rj = judgements_conv_map[data[3]]
        user = data[-1].strip()
        if qid not in rj_by_q.keys():
            rj_by_q[qid] = {}
        if did not in rj_by_q[qid]:
            rj_by_q[qid][did] = {}
        rj_by_q[qid][did][user] = rj

    # avg judges per document and evaluated docs per topic:
    mean_docs_per_topic = np.mean([len(topic_data) for topic_data in rj_by_q.values()])
    std_docs_per_topic = np.std([len(topic_data) for topic_data in rj_by_q.values()])
    mean_judges_per_doc = np.mean(
        [len(judgements.keys()) for topic_data in rj_by_q.values() for judgements in topic_data.values()])
    std_judges_per_doc = np.std(
        [len(judgements.keys()) for topic_data in rj_by_q.values() for judgements in topic_data.values()])
    print('avg judged docs per topic: {:.4f}, std: {:.4f}'.format(mean_docs_per_topic, std_docs_per_topic))
    print('avg judges per topic: {:.4f}, std: {:.4f}'.format(mean_judges_per_doc, std_judges_per_doc))
    inter_annotator_agreement(rj_by_q)
    doc_rj_dists = {}
    # read data and build labels distr
    for qid in rj_by_q.keys():
        for did in rj_by_q[qid].keys():
            curr_dist = np.zeros(3)
            for user, rel_score in rj_by_q[qid][did].items():
                curr_dist[rel_score] += 1
            # normalize dist:
            curr_dist /= np.sum(curr_dist)
            if qid not in doc_rj_dists.keys():
                doc_rj_dists[qid] = {}
            doc_rj_dists[qid][did] = curr_dist
    return doc_rj_dists


def inter_annotator_agreement(rj_by_q):
    annotator_names = [[[name for name in annotators.keys()] for annotators in judgements.values()] for did, judgements
                       in rj_by_q.items()]
    col_names = []
    for l1 in annotator_names:
        for l2 in l1:
            for l3 in l2:
                col_names.append(l3)
    col_names = list(set(col_names))
    iaas = []
    for topic_data in rj_by_q.values():
        docids = list(topic_data.keys())
        annotation_matrix = np.zeros((len(docids), len(col_names)))

        for docid in docids:
            for annotator_name in col_names:
                if annotator_name in topic_data[docid].keys():
                    res = topic_data[docid][annotator_name]
                else:
                    res = np.nan
                annotation_matrix[docids.index(docid), col_names.index(annotator_name)] = res
            all_judgements = list(topic_data[docid].values())
            iaas.append(compute_iaa(all_judgements))
    print('IAA: {}'.format(np.mean(iaas)))


def compute_iaa(judgements):
    same = 0
    different = 0
    for i in range(len(judgements)):
        for j in range(len(judgements)):
            if i != j:
                same += 1 if judgements[i] == judgements[j] else 0
                different += 1 if judgements[i] != judgements[j] else 0
    if len(judgements) == 1:
        return 1.0
    else:
        return same / (len(judgements) * (len(judgements) - 1))


def flatten_data(docs_scores_by_qid, qids_dids_labels, curr_qids, consider_raw_rj_dists):
    qids = []
    doc_vecs = []
    rlabels = []
    dids = []
    ndocs_per_qid = {}
    for qid, doc_vecs_by_id in docs_scores_by_qid.items():
        if qid not in ndocs_per_qid.keys():
            ndocs_per_qid[qid] = 0
        if qid in curr_qids:
            for did, dv in doc_vecs_by_id.items():
                # qids.append(qid)
                # doc_vecs.append(dv)
                # dids.append(did)
                is_rel = False
                if consider_raw_rj_dists:
                    # rel_label = np.zeros(3)
                    rel_label = np.array([1.0, 0.0, 0.0])
                else:
                    rel_label = 0
                if did in qids_dids_labels[qid].keys():
                    rel_label = qids_dids_labels[qid][did]
                    is_rel = True
                if ndocs_per_qid[qid] < 120 or is_rel:
                    rlabels.append(rel_label)
                    qids.append(qid)
                    doc_vecs.append(dv)
                    dids.append(did)
                    ndocs_per_qid[qid] += 1
    return np.array(doc_vecs), np.array(rlabels), np.array(qids), np.array(dids)


if __name__ == '__main__':
    get_judgements_distribution()
    # compute_data_MLIA('Fold1', True)
    # compute_data_MLIA('Fold2')
    # compute_data_MLIA('Fold3')
    # compute_data_MLIA('Fold4')
    # compute_data_MLIA('Fold5')
