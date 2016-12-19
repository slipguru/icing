"""Validation utils for clustering algorithms.

Author: Federico Tomasi
Copyright (c) 2016, Federico Tomasi.
Licensed under the FreeBSD license (see LICENSE.txt).
"""
import numpy as np
import pandas as pd
import seaborn as sns


def get_clones_real_estimated(filename):
    # res = '/home/fede/projects_local/icing_learning_step_simulated/results/'
    # 'icing_clones_20.tab_2016-12-12_18.08.55,509543/db_file_clusters.tab'
    df = pd.read_csv(filename, dialect='excel-tab', header=0)

    df['CLONE_ID'] = df['SEQUENCE_ID'].str.split('_').apply(lambda x: x[3])

    # confusion matrix
    clone_ids = np.array(df['CLONE_ID'], dtype=str)
    found_ids = np.array(df['CLONE'], dtype=str)
    return clone_ids, found_ids


# from sklearn import preprocessing
# le = preprocessing.LabelEncoder()
# le.fit(clone_ids)
# print(le.classes_)
# print(le.transform(clone_ids))

# let = preprocessing.LabelEncoder()
# let.fit(found_ids)
# print(let.classes_)
# print(let.transform(found_ids))


def confusion_matrix(true_labels, estimated_labels, squared=True,
                     ordered=True):
    """Return a confusion matrix in a multiclass / multilabel problem."""
    rows = np.unique(true_labels)
    cols = np.unique(estimated_labels)
    if squared:
        dim = max(rows.shape[0], cols.shape[0])
        dims = (dim, dim)
    else:
        dims = (rows.shape[0], cols.shape[0])
    cm = np.zeros(dims)
    from collections import Counter
    for i, row in enumerate(rows):
        idx_rows = clone_ids == row
        counter = Counter(found_ids[idx_rows])
        for g in counter:
            idx_col = np.where(cols == g)[0][0]
            cm[i, idx_col] += counter[g]
    if squared:
        mins = min(rows.shape[0], cols.shape[0])
        add_rows = cols.shape[0] - mins
        add_cols = rows.shape[0] - mins
        rows = np.append(rows, ['pad'] * add_rows)
        cols = np.append(cols, ['pad'] * add_cols)
    if ordered:
        cm, rr, cc = order_cm(cm)
        rows, cols = rows[rr], cols[cc]
    return cm, rows, cols


def order_cm(a):
    # reorder rows
    idx_rows = np.max(a, axis=1).argsort()[::-1]
    b = a[idx_rows, :]

    max_idxs = np.ones(b.shape[1], dtype=bool)
    final_idxs = []
    for i, row in enumerate(b.copy()):
        if i == b.shape[0] or not max_idxs.any():
            break
        row[~max_idxs] = np.min(a) - 5000
        max_idx = np.argmax(row)
        final_idxs.append(max_idx)
        max_idxs[max_idx] = False

    idx_cols = np.append(np.array(final_idxs, dtype=int), np.argwhere(max_idxs).T[0])
    return b[:, idx_cols], idx_rows, idx_cols


def multiclass_TP_FP_FN(a, return_dict=True):
    def _single_measures(a, i):
        TP = a[i, i]
        FP = np.sum(a[:, i]) - TP
        FN = np.sum(a[i, :]) - TP
        return TP, FP, FN
    singles = zip(*[_single_measures(a, i) for i in range(min(a.shape))])
    maps = map(lambda x: np.array(list(x)).mean(), singles)
    if return_dict:
        return dict(zip(['TP', 'FP', 'FN'], maps))
    return maps


def show_heatmap(filename):
    true_labels, estimated_labels = get_clones_real_estimated(filename)
    cm, rows, cols = confusion_matrix(true_labels, estimated_labels)
    df = pd.DataFrame(cm, index=rows, columns=cols)
    sns.heatmap(df)
    sns.plt.show()


def make_square(a, rows=None, cols=None):
    if a.shape[0] == a.shape[1]:
        return a
    mins = min(a.shape)
    diff_rows = a.shape[1] - mins
    diff_cols = a.shape[0] - mins
    if rows is not None and diff_rows > 0:
        rows = np.append(rows, ['pad'] * diff_rows)
    if cols is not None and diff_cols > 0:
        cols = np.append(cols, ['pad'] * diff_cols)
    return np.pad(a, ((0, diff_rows), (0, diff_cols)),
                  mode='constant', constant_values=0), rows, cols


# For class x:
# True positive: diagonal position, cm(x, x).
# False positive: sum of column x (without main diagonal), sum(cm(:, x))-cm(x, x).
# False negative: sum of row x (without main diagonal), sum(cm(x, :), 2)-cm(x, x).
# You can compute precision, recall and F1 score following course formula.

# Averaging over all classes (with or without weighting) gives values for the entire model.

def test_validation():
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 5, 7]])
    a, _, _ = make_square(a)
    print("Unordered random confusion matrix", a)
    a, _, _ = order_cm(a)
    print("Ordered random confusion matrix", a)
    print(multiclass_TP_FP_FN(a))


if __name__ == '__main__':
    test_validation()
