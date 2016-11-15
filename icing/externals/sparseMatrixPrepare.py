"""Prepare Sparse Matrix for Sparse Affinity Propagation Clustering (SAP).

Script from pysapc.
"""
# Authors: Huojun Cao <bioinfocao at gmail.com>
# License: BSD 3 clause

import numpy as np


def rmSingleSamples(rows, cols, data, n_samples):
    """Remove single samples from the sparse matrix.

    Affinity/similarity matrix does not need be symmetric.
    For the FSAPC to work, specifically in computation of R and A matrix,
    each row/column of Affinity/similarity matrix should have at least two
    datapoints. Samples do not meet this condition are removed from
    computation.

    Also, samples with only one symmetric datapoint are removed,
    For example for sample 'B' only s(B,C) exist and for sample 'C' only
    s(C,B) exists. In these two cases, these samples are removed from
    computation and their examplers are set to themself.

    For samples that only have one data (affinity/similarity) with others, For
    example if for sample 'A', the only datapoint of [s(A,A),s(A,B),s(A,C)...]
    is s(A,B), and there exist at least one value in [s(A,A),s(C,A),s(D,A)...]
    (except s(B,A), because if we copy s(B,A), for 'A' we still only have one
    data point) then we copy the minimal value of [s(A,A),s(C,A),s(D,A)...].

    Parameters:
    -----------
    rows, cols, data : array-like
        Rows and columns for the sparse matrix. Data is row-based.
    n_samples : int
        Number of samples in the original input data.
    """
    # find rows and cols that only have one datapoint
    unique, counts = np.unique(rows, return_counts=True)
    single_rows = set(unique[counts < 2])
    unique, counts = np.unique(cols, return_counts=True)
    single_cols = set(unique[counts < 2])

    if len(single_rows) == 0 and len(single_cols) == 0:
        # No modification
        return rows, cols, data, None, None, n_samples

    # remove samples that only have affinity/similarity with itself
    # or only have one symmetric datapoint, for example for sample 'B' only
    # s(B,C) exist and for sample 'C' only s(C,B) exist
    # in these two cases, these samples are removed from FSAPC computation and
    # their examplers are set to themself.
    single_samples = single_rows & single_cols
    rowLeftOriDict = None
    if len(single_samples) > 0:
        # row indexs that left after remove single samples
        # rowLeft_old = sorted(list(set(range(n_samples)) - single_samples))
        mask = np.ones(n_samples, dtype=bool)
        mask[np.array(single_samples)] = False
        rowLeft = np.arange(n_samples)[mask]
        # assert rowLeft == rowLeft_old
        # map of original row index to current row index(after remove rows/cols
        # that only have single item)
        rowOriLeftDict, rowLeftOriDict = {}, {}
        for left, ori in enumerate(rowLeft):
            rowOriLeftDict[ori] = left
            rowLeftOriDict[left] = ori

        # remove single elements outside the diagonal
        mask = np.ones(rows.shape[0], dtype=bool)
        mask[single_samples] = False
        idx_to_remove = np.logical_or(rows == cols, mask)
        rows = rows[idx_to_remove]
        cols = cols[idx_to_remove]
        data = data[idx_to_remove]

        # rows, cols, data = sparseAP_cy.removeSingleSamples(
        #     rows, cols, data, single_samples)
        # assert set(rows) == set(rows)
        # assert set(cols) == set(cols)
        # assert set(data) == set(data)

    # for samples that need copy a minimal value to have at least two
    # datapoints in row/column
    # for samples that row have single data point, copy minimal value of this
    # sample's column

    def _copySym(rows, cols, data, single_rows):
        # mask = np.logical_and(rows != cols, np.in1d(cols, list(single_rows)))
        rows_copy = np.empty(0, dtype=int)
        cols_copy = np.empty(0, dtype=int)
        data_copy = np.empty(0)
        for ind in single_rows:
            mask = np.logical_and(cols == ind, cols != rows)
            idx = np.argmin(data[mask])
            rows_copy = np.append(rows_copy, cols[mask][idx])
            cols_copy = np.append(cols_copy, rows[mask][idx])
            data_copy = np.append(data_copy, data[mask][idx])
        rows = np.concatenate((rows, rows_copy))
        cols = np.concatenate((cols, cols_copy))
        data = np.concatenate((data, data_copy))
        return rows, cols, data

    single_rows = single_rows - single_samples
    if len(single_rows) > 0:
        rows, cols, data = _copySym(rows, cols, data, single_rows)
    # for samples that col have single data point, copy minimal value of this row
    single_cols = single_cols - single_samples
    if len(single_cols) > 0:
        cols, rows, data = _copySym(cols, rows, data, single_cols)

    # change row, col index if there is any sample removed
    if len(single_samples) > 0:
        changeIndV = np.vectorize(lambda x: rowOriLeftDict[x])
        rows = changeIndV(rows)
        cols = changeIndV(cols)

    idx_sorted_left_ori = np.lexsort((cols, rows))
    rows = rows[idx_sorted_left_ori]
    cols = cols[idx_sorted_left_ori]
    data = data[idx_sorted_left_ori]
    return rows, cols, data, rowLeftOriDict, \
        single_samples, n_samples - len(single_samples)
