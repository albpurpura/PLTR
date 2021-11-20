import numpy as np
import scipy.stats as ss


def get_most_freq_entry(values):
    # majority voting
    distinct_values = set(values)
    freqs = [values.count(v) for v in distinct_values]
    return values[np.argmax(freqs)]


def compute_simulated_labels_give_features_list(data):
    n_pos = 5  # 5
    thr = 0.2  # 0.2
    # multiplier_idx = 40  # pagerank
    print('SIMULATE LABELS')
    ranking_lists, rl_lengths, true_labels, features_to_use_for_ranking_indices = data
    final_rl_len = len(ranking_lists[0])
    artificial_labels = []
    for i in range(len(ranking_lists)):
        docs_positions_with_different_ranking_models = []
        for ranking_feature in features_to_use_for_ranking_indices:
            # pg_ranks = np.array(ranking_lists[i])[:, multiplier_idx][:rl_lengths[i]]
            # pg_ranks /= max(pg_ranks)
            # curr_f_scores = np.array(ranking_lists[i])[:, ranking_feature][:rl_lengths[i]]
            # curr_f_scores /= max(curr_f_scores)
            feat_values = np.array(ranking_lists[i])[:, ranking_feature][:rl_lengths[i]]
            positions = ss.rankdata(-feat_values, method='max')
            docs_positions_with_different_ranking_models.append(positions)
        curr_rl_labels_alternatives = []
        for predictions_seq in docs_positions_with_different_ranking_models:
            curr_labels_preds = []
            for j in range(len(predictions_seq)):
                if predictions_seq[j] <= n_pos:
                    curr_labels_preds.append(n_pos - (predictions_seq[j]))
                else:
                    curr_labels_preds.append(0)
            curr_rl_labels_alternatives.append(curr_labels_preds)
        curr_rl_labels_alternatives = np.array(curr_rl_labels_alternatives)
        final_labels = np.mean(np.array(curr_rl_labels_alternatives), axis=0) / (n_pos - 1)
        # final_labels = np.array([get_most_freq_entry(list(curr_rl_labels_alternatives[:, k])) for k in
        #                          range(curr_rl_labels_alternatives.shape[1])]) / (n_pos - 1)
        final_labels = [l if l >= thr else 0.0 for l in final_labels]
        curr_true_labels = true_labels[i][:rl_lengths[i]]
        assert len(final_labels) == len(curr_true_labels)

        artificial_labels.append(final_labels)
    # pad new labels lists
    for i in range(len(artificial_labels)):
        artificial_labels[i] = list(artificial_labels[i])
        while len(artificial_labels[i]) < final_rl_len:
            artificial_labels[i].append(0.0)
    tp, fp, tn, fn = compare_artif_rj_with_real_ones(artificial_labels, true_labels, rl_lengths)
    return artificial_labels, tp, fp, tn, fn, features_to_use_for_ranking_indices


def compute_simulated_labels(ranking_lists, rl_lengths, true_labels):
    features_to_use_for_ranking_indices = [24, 29, 39]  # [24, 29, 39]
    artificial_labels, tp, fp, tn, fn, _ = compute_simulated_labels_give_features_list(
        (ranking_lists, rl_lengths, true_labels, features_to_use_for_ranking_indices))

    tpr = tp / (tp + fn)
    print('tp: {}'.format(tp))
    print('fp: {}'.format(fp))
    print('fn: {}'.format(fn))
    print('tpr: {}'.format(tpr))
    return artificial_labels


def compare_artif_rj_with_real_ones(artif_rj, real_rj, rl_lengths):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    real_rj = np.array(real_rj) / 4
    for ranked_list_index in range(len(real_rj)):
        curr_artif_rj = artif_rj[ranked_list_index]
        curr_real_rj = real_rj[ranked_list_index]

        for j in range(rl_lengths[ranked_list_index]):
            if curr_real_rj[j] > 0 and curr_artif_rj[j] > 0:
                tp += 1
            if curr_real_rj[j] > 0 and curr_artif_rj[j] == 0:
                fn += 1
            if curr_real_rj[j] == 0 and curr_artif_rj[j] == 0:
                tn += 1
            if curr_real_rj[j] == 0 and curr_artif_rj[j] > 0:
                fp += 1
    return tp, fp, tn, fn


def sample_labels(real_rj, rl_lengths, n_samples):
    # could do 100 extraction and average
    print('SAMPLE labels')
    all_sampled_labels = []
    for i in range(len(real_rj)):
        curr_sampled_labels = []
        for j in range(len(real_rj[i])):
            if j < rl_lengths[i]:
                sampled_label = np.random.binomial(n_samples, real_rj[i][j]) / n_samples
            else:
                sampled_label = real_rj[i][j]
            curr_sampled_labels.append(sampled_label)
        all_sampled_labels.append(curr_sampled_labels)
    return all_sampled_labels
