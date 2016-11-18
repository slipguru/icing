#!/usr/bin/env python
"""Utilities for Sparse Affinity Propagation Clustering (SAP).

These functions are based on functions in the pysapc module, but optimised
with numpy efficient computations.

Author: Federico Tomasi
Copyright (c) 2016, Federico Tomasi.
Licensed under the FreeBSD license (see LICENSE.txt).
"""
import numpy as np

from scipy.sparse import coo_matrix, csr_matrix, lil_matrix


def matrix_to_row_col_data(X):
    """Convert sparse affinity matrix to arrays.

    .. note:: Deprecated.
          It will be removed in icing 0.2. This is now done by check_array from
          numpy.
    """
    # convert to coo format (from lil,csr,csc)
    if isinstance(X, coo_matrix):
        X_coo = X
    elif (isinstance(X, csr_matrix)) or (isinstance(X, lil_matrix)):
        X_coo = X.tocoo()
    else:  # others like numpy matrix could be convert to coo matrix
        X_coo = coo_matrix(X)
    # Upcast matrix to a floating point format (if necessary)
    X_coo = X_coo.asfptype()
    return X_coo.row.astype(np.int), X_coo.col.astype(np.int), X_coo.data


def parse_preference(preference, n_samples, data):
    """Set array of preferences.

    Parameters
    ----------
    preference : str, array-like or float
        If preference is array-like, it must have same dimensions as data.
    n_samples : int
        Number of samples in the analysis.
    data : array-like
        Array of dense data.
    """
    if (isinstance(preference, list) or isinstance(preference, np.ndarray)):
        if len(preference) != n_samples:
            raise ValueError("Preference array of incorrect size.")
        return np.asarray(preference)

    preference = preference == 'min' and data.min() or \
        preference == 'median' and np.median(data) or preference
    return np.ones(n_samples) * preference


def _set_sparse_diagonal(rows, cols, data, preferences):
    idx = np.where(rows == cols)
    data[idx] = preferences[rows[idx]]
    mask = np.ones(preferences.shape, dtype=bool)
    mask[rows[idx]] = False
    diag_other = np.argwhere(mask).T[0]
    rows = np.concatenate((rows, diag_other))
    cols = np.concatenate((cols, diag_other))
    data = np.concatenate((data, preferences[mask]))

    # return data sorted by row
    idx_sorted_left_ori = np.lexsort((cols, rows))
    rows = rows[idx_sorted_left_ori]
    cols = cols[idx_sorted_left_ori]
    data = data[idx_sorted_left_ori]
    return rows, cols, data


def _sparse_row_maxindex(data, row_indptr):
    # data and row_idx must have same dimensions.
    # row_idx is ordered
    tmp = np.empty(row_indptr.shape[0] - 1, dtype=int)
    for i in range(row_indptr.shape[0] - 1):
        i_start = row_indptr[i]
        i_end = row_indptr[i + 1]
        # tmp[i_start:i_end] = np.sort(data[i_start:i_end])[::-1]
        tmp[i] = np.argmax(data[i_start:i_end]) + i_start
    return tmp


# def _sparse_maxindex(data, rows):
#     # data and row_idx must have same dimensions.
#     # row_idx is ordered
#     ma = np.ma.MaskedArray(data)
#     tmp = np.empty(np.unique(rows).shape[0], dtype=int)
#     for i in np.unique(rows):
#         ma.mask = np.where(rows == i, False, True)
#         tmp[i] = np.ma.argmax(ma)
#     return tmp


def _sparse_row_sum_update(data, row_indptr, diag_idxs):
    # data and row_idx must have same dimensions.
    # row_idx is ordered
    for i in range(row_indptr.shape[0] - 1):
        i_start = row_indptr[i]
        i_end = row_indptr[i + 1]
        data[i_start:i_end] -= np.sum(data[i_start:i_end])
        kk_ind = diag_idxs[i]
        diag = data[kk_ind]
        data[i_start:i_end].clip(0, np.inf, data[i_start:i_end])
        data[kk_ind] = diag


def _sparse_sum_update(data, rows, diag_idxs):
    for i in np.unique(rows):
        idx = np.where(rows == i)
        data.flat[idx] -= np.sum(data.flat[idx])
        kk_ind = diag_idxs[i]
        diag = data[kk_ind]
        data.flat[idx] = np.clip(data.flat[idx], 0, np.inf)
        data[kk_ind] = diag


def _update_r_max_row(data, row_indptr):
    max_row = np.empty(data.shape[0])
    for i in range(row_indptr.shape[0] - 1):
        i_start = row_indptr[i]
        i_end = row_indptr[i + 1]
        s_max_i, max_i = (np.argsort(data[i_start:i_end]) + i_start)[-2:]
        max_row[i_start:i_end] = data[max_i]
        max_row[max_i] = data[s_max_i]
    return max_row


def _copy_symmetric_elem(rows, cols, data, single_rows):
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


def remove_single_samples(rows, cols, data, n_samples):
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

    Parameters
    -----------
    rows, cols, data : array-like
        Rows and columns for the sparse matrix. Data is row-based.
    n_samples : int
        Number of samples in the original input data.
    """
    # find rows and cols that only have one datapoint
    unique, counts = np.unique(rows, return_counts=True)
    single_rows = unique[counts < 2]
    unique, counts = np.unique(cols, return_counts=True)
    single_cols = unique[counts < 2]
    if single_rows.shape[0] == 0 and single_cols.shape[0] == 0:
        return rows, cols, data, None, None, n_samples

    single_rows, single_cols = set(single_rows), set(single_cols)
    single_samples = single_rows & single_cols
    left_ori_dict = None
    if len(single_samples) > 0:
        # row indexs that left after remove single samples
        mask = np.ones(n_samples, dtype=bool)
        mask[np.array(single_samples)] = False
        # map of original row index to current row index(after remove rows/cols
        # that only have single item)
        ori_left_dict, left_ori_dict = {}, {}
        for left, ori in enumerate(np.arange(n_samples)[mask]):
            ori_left_dict[ori] = left
            left_ori_dict[left] = ori

        # remove single elements outside the diagonal
        mask = np.ones(rows.shape[0], dtype=bool)
        mask[single_samples] = False
        idx_to_leave = np.logical_or(rows == cols, mask)
        rows = rows[idx_to_leave]
        cols = cols[idx_to_leave]
        data = data[idx_to_leave]

    # If some row have a single data point, copy the min value in col
    single_rows = single_rows - single_samples
    if len(single_rows) > 0:
        rows, cols, data = _copy_symmetric_elem(rows, cols, data, single_rows)

    # If some col have a single data point, copy the min value in row
    single_cols = single_cols - single_samples
    if len(single_cols) > 0:
        cols, rows, data = _copy_symmetric_elem(cols, rows, data, single_cols)

    # change row, col index if there is any sample removed
    if len(single_samples) > 0:
        change_idx = np.vectorize(lambda x: ori_left_dict[x])
        rows = change_idx(rows)
        cols = change_idx(cols)
        n_samples -= len(single_samples)

    idx_sorted_left_ori = np.lexsort((cols, rows))
    rows = rows[idx_sorted_left_ori]
    cols = cols[idx_sorted_left_ori]
    data = data[idx_sorted_left_ori]
    return rows, cols, data, left_ori_dict, single_samples, n_samples
