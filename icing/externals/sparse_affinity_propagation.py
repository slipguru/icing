"""Affinity Propagation clustering algorithm.

This module is an extension of sklearn's Affinity Propagation to take into
account sparse matrices, which are not currently supported.
"""

# Author: Alexandre Gramfort alexandre.gramfort@inria.fr
#        Gael Varoquaux gael.varoquaux@normalesup.org

# License: BSD 3 clause

import numpy as np

from scipy.sparse import coo_matrix, csr_matrix, lil_matrix
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import as_float_array, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import euclidean_distances
from sklearn.metrics import pairwise_distances_argmin


def rmSingleSamples(rows, cols, rowBased_data, n_samples):
    """
    Affinity/similarity matrix does not need be symmetric, that is s(A,B) does not need be same as s(B,A).
    Also since Affinity/similarity matrix is sparse, it could be that s(A,B) exist but s(B,A) does not exist in the sparse matrix.
    For the FSAPC to work, specifically in computation of R and A matrix, each row/column of Affinity/similarity matrix should have at least two datapoints.
    So in FSAPC, we first remove samples that do not have affinity/similarity with other samples, that is samples that only have affinity/similarity with itself
    And we remove samples only have one symmetric datapoint, for example for sample 'B' only s(B,C) exist and for sample 'C' only s(C,B) exist
    In these two cases, these samples are removed from FSAPC computation and their examplers are set to themself.
    For samples that only have one data (affinity/similarity) with others, For example if for sample 'A', the only datapoint of [s(A,A),s(A,B),s(A,C)...] is s(A,B),
    and there exist at least one value in [s(A,A),s(C,A),s(D,A)...] (except s(B,A), because if we copy s(B,A), for 'A' we still only have one data point)
    then we copy the minimal value of [s(A,A),s(C,A),s(D,A)...]
    n_samples is the number of samples of orignail input data
    """
    # find rows and cols that only have one datapoint
    # singleRowInds=set(sparseAP_cy.singleItems(rows))
    # singleColInds=set(sparseAP_cy.singleItems(cols))

    unique, counts = np.unique(rows, return_counts=True)
    singleRowInds = set(unique[counts < 2])
    unique, counts = np.unique(cols, return_counts=True)
    singleColInds = set(unique[counts < 2])

    # assert sorted(singleRowInds) == sorted(rows_one_element)
    # in case every col/row have more than one datapoint, return original data
    if len(singleRowInds) == 0 and len(singleColInds) == 0:
        return rows, cols, rowBased_data, \
            None, None, n_samples

    # samples that have one datapoint in row and col are samples only have
    # affinity/similarity with itself
    singleSampleInds = singleRowInds & singleColInds

    # remove samples that only have affinity/similarity with itself
    # or only have one symmetric datapoint, for example for sample 'B' only
    # s(B,C) exist and for sample 'C' only s(C,B) exist
    # in these two cases, these samples are removed from FSAPC computation and
    # their examplers are set to themself.
    if len(singleSampleInds) > 0:
        # row indexs that left after remove single samples
        # rowLeft_old = sorted(list(set(range(n_samples)) - singleSampleInds))
        mask = np.ones(n_samples, dtype=bool)
        mask[np.array(singleSampleInds)] = False
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
        mask[singleSampleInds] = False
        idx_to_remove = np.logical_or(rows == cols, mask)
        rows = rows[idx_to_remove]
        cols = cols[idx_to_remove]
        rowBased_data = rowBased_data[idx_to_remove]

        # rows, cols, rowBased_data = sparseAP_cy.removeSingleSamples(
        #     rows, cols, rowBased_data, singleSampleInds)
        # assert set(rows) == set(rows)
        # assert set(cols) == set(cols)
        # assert set(data) == set(rowBased_data)
    else:  # no samples are removed
        rowLeftOriDict = None

    # for samples that need copy a minimal value to have at least two
    # datapoints in row/column
    # for samples that row have single data point, copy minimal value of this
    # sample's column
    singleRowInds = singleRowInds - singleSampleInds
    if len(singleRowInds) > 0:
        rows, cols, rowBased_data = copySym(
            rows, cols, rowBased_data, singleRowInds)
    # for samples that col have single data point, copy minimal value of this sample's row
    singleColInds = singleColInds - singleSampleInds
    if len(singleColInds) > 0:
        cols, rows, rowBased_data = copySym(
            cols, rows, rowBased_data, singleColInds)

    # change row, col index if there is any sample removed
    if len(singleSampleInds) > 0:
        changeIndV = np.vectorize(lambda x: rowOriLeftDict[x])
        rows = changeIndV(rows)
        cols = changeIndV(cols)

    # rearrange based on new row index and new col index
    sortedLeftOriInd = np.lexsort((cols, rows)).astype(np.int)

    rows = sparseAP_cy.npArrRearrange_int_para(
        rows, sortedLeftOriInd)
    cols = sparseAP_cy.npArrRearrange_int_para(
        cols, sortedLeftOriInd)
    rowBased_data = sparseAP_cy.npArrRearrange_float_para(
        rowBased_data, sortedLeftOriInd)
    # rows2 = rows[sortedLeftOriInd]
    # cols2 = cols[sortedLeftOriInd]
    # rowBased_data2 = rowBased_data[sortedLeftOriInd]
    # assert rowBased_data1 == rowBased_data2
    # assert cols1 == cols2
    # assert rows1 == rows2

    return rows, cols, rowBased_data, rowLeftOriDict, \
        singleSampleInds, n_samples - len(singleSampleInds)


def matrix_to_row_col_data(X):
    """Convert sparse affinity/similarity matrix to arrays.

    (row_array,col_array,data_array)
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
    """
    Input preference should be a numeric scalar,
    a string of 'min' / 'median', or a list/np 1D array(length of samples).
    Return preference list(same length as samples)
    """
    err = "Preference should be a numeric scalar, a string of "
    "'min' / 'median', or a list/np 1D array(length of samples)."
    "\nYour input preference is: {0})".format(str(preference))

    if (isinstance(preference, list) or isinstance(preference, np.ndarray)):
        if len(preference) != n_samples:
            raise ValueError()
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
    return rows, cols, data


def _sparse_row_maxindex(data, row_indptr):
    # data and row_idx must have same dimensions.
    # row_idx is ordered
    tmp = np.empty(row_indptr.shape[0]-1, dtype=int)
    for i in range(1, row_indptr.shape[0]):
        i_start = row_indptr[i - 1]
        i_end = row_indptr[i]
        # tmp[i_start:i_end] = np.sort(data[i_start:i_end])[::-1]
        tmp[i - 1] = np.argmax(data[i_start:i_end]) + i_start
    return tmp


def _sparse_row_sum_update(data, row_indptr, kk):
    # data and row_idx must have same dimensions.
    # row_idx is ordered
    tmp = np.empty(data.shape[0])
    for i in range(1, row_indptr.shape[0]):
        i_start = row_indptr[i - 1]
        i_end = row_indptr[i]
        col_sum = np.sum(data[i_start:i_end])
        np.clip(data[i_start:i_end] - col_sum, 0, np.inf, tmp[i_start:i_end])
        kk_ind = kk[i - 1]
        tmp[kk_ind] = data[kk_ind] - col_sum
    return tmp


def _updateR_maxRow(arr, row_indptr):
    """Given a numpy array(arr) and split index(sp[0,...,len(arr)]),
    Return a array that have max value of each split, except the position that have max value will have second max value
    """
    max_row = np.empty(arr.shape[0])
    for i in range(1, row_indptr.shape[0]):
        i_start = row_indptr[i - 1]
        i_end = row_indptr[i]
        smi, mi = (np.argsort(arr[i_start:i_end]) + i_start)[-2:]
        max_row[i_start:i_end] = arr[mi]
        max_row[mi] = arr[smi]
    return max_row


def sparse_ap(S, preference=None, convergence_iter=15, max_iter=200,
              damping=0.5, copy=True, verbose=False,
              return_n_iter=False, convergence_percentage=0.999999):
    """Perform Affinity Propagation Clustering of data.

    Same as affinity_propagation(), but assume S be a sparse matrix.

    Parameters
    ----------

    S : array-like, shape (n_samples, n_samples)
        Matrix of similarities between points

    preference : array-like, shape (n_samples,) or float, optional
        Preferences for each point - points with larger values of
        preferences are more likely to be chosen as exemplars. The number of
        exemplars, i.e. of clusters, is influenced by the input preferences
        value. If the preferences are not passed as arguments, they will be
        set to the median of the input similarities (resulting in a moderate
        number of clusters). For a smaller amount of clusters, this can be set
        to the minimum value of the similarities.

    convergence_iter : int, optional, default: 15
        Number of iterations with no change in the number
        of estimated clusters that stops the convergence.

    max_iter : int, optional, default: 200
        Maximum number of iterations

    damping : float, optional, default: 0.5
        Damping factor between 0.5 and 1.

    copy : boolean, optional, default: True
        If copy is False, the affinity matrix is modified inplace by the
        algorithm, for memory efficiency

    verbose : boolean, optional, default: False
        The verbosity level

    return_n_iter : bool, default False
        Whether or not to return the number of iterations.

    Returns
    -------

    cluster_centers_indices : array, shape (n_clusters,)
        index of clusters centers

    labels : array, shape (n_samples,)
        cluster labels for each point

    n_iter : int
        number of iterations run. Returned only if `return_n_iter` is
        set to True.

    Notes
    -----
    See examples/cluster/plot_affinity_propagation.py for an example.

    References
    ----------
    Brendan J. Frey and Delbert Dueck, "Clustering by Passing Messages
    Between Data Points", Science Feb. 2007
    """
    rows, cols, data = matrix_to_row_col_data(S)
    # Get parameters (nSamplesOri, preference_list, damping)
    n_samples = S.shape[0]
    preferences = parse_preference(preference, n_samples, data)
    if damping < 0.5 or damping >= 1:
        raise ValueError('damping must be >= 0.5 and < 1')
    if convergence_percentage is None:
        convergence_percentage = 1

    random_state = np.random.RandomState(0)

    # set diag of affinity/similarity matrix to preference_list
    rows, cols, data = _set_sparse_diagonal(rows, cols, data, preferences)

    # reOrder by rowbased
    sortedLeftOriInd = np.lexsort((cols, rows))
    rows = rows[sortedLeftOriInd]
    cols = cols[sortedLeftOriInd]
    data = data[sortedLeftOriInd]

    # For the FSAPC to work, specifically in computation of R and A matrix,
    # each row/column of Affinity/similarity matrix should have at least two
    # datapoints.
    # Samples do not meet this condition are removed from computation
    # (so their exemplars are themself) or copy a minimal value of
    # corresponding column/row
    rows, cols, data, rowLeftOriDict, singleSampleInds, n_samples = \
        rmSingleSamples(rows, cols, data, n_samples)

    data_len = data.shape[0]
    row_indptr = np.insert(np.where(np.diff(rows) > 0)[0] + 1, 0, 0)
    if row_indptr[-1] != data_len:
        row_indptr = np.append(row_indptr, data_len)

    row_to_col_ind_arr = np.lexsort((rows, cols))
    rows_colbased = rows[row_to_col_ind_arr]
    cols_colbased = cols[row_to_col_ind_arr]
    col_to_row_ind_arr = np.lexsort((cols_colbased, rows_colbased))

    col_indptr = np.insert(np.where(np.diff(cols_colbased) > 0)[0] + 1, 0, 0)
    if col_indptr[-1] != data_len:
        col_indptr = np.append(col_indptr, data_len)

    kk_col_index = np.where(rows_colbased == cols_colbased)[0]
    kk_row_index = np.where(rows == cols)[0]

    # Add random small value to remove degeneracies
    data_len = data.shape[0]
    data += ((np.finfo(np.double).eps * data + np.finfo(np.double).tiny * 100)
             * random_state.randn(data_len))

    # Initialize matrix A, R
    A = np.zeros(data_len, dtype=float)
    R = np.zeros(data_len, dtype=float)
    tmp = np.zeros(data_len, dtype=float)
    # Update R, A matrix until meet convergence condition or
    # reach max iteration. In FSAPC, the convergence condition is when there is
    # more than convergence_iter iteration in rows that have exact clustering
    # result or have similar clustering result that similarity is great than
    # convergence_percentage if convergence_percentage is set other than None
    # (default convergence_percentage is 0.999999, that is one datapoint in 1
    # million datapoints have different clustering.)
    # This condition is added to FSAPC is because FSAPC is designed to deal
    # with large data set.

    lastLabels = np.empty((0), dtype=np.int)
    labels = np.empty((0), dtype=np.int)
    convergeCount = 0

    for it in range(max_iter):
        lastLabels = labels

        # tmp = A + S; compute responsibilities
        np.add(A, data, tmp)
        np.subtract(data, _updateR_maxRow(tmp, row_indptr), tmp)

        # Damping
        tmp *= 1. - damping
        R *= damping
        R += tmp

        # tmp = Rp; compute availabilities
        np.maximum(R, 0, tmp)
        tmp[kk_row_index] = R[kk_row_index]

        # tmp = -Anew
        tmp = _sparse_row_sum_update(tmp[row_to_col_ind_arr], col_indptr,
                                     kk_col_index)[col_to_row_ind_arr]

        # Damping
        tmp *= 1. - damping
        A *= damping
        A -= tmp

        labels = _sparse_row_maxindex(A + R, row_indptr)

        # check convergence
        eq_perc = np.sum(lastLabels == labels) / float(labels.shape[0])
        if eq_perc >= convergence_percentage and labels.shape[0] > 0:
            convergeCount += 1
        else:
            convergeCount = 0
        if convergeCount == convergence_iter:
            break

    # Converting labels back to original sample index
    cols = cols[labels]
    if singleSampleInds is None or len(singleSampleInds) == 0:
        cluster_centers_indices = cols
    else:
        cluster_centers_indices = [rowLeftOriDict[el] for el in cols]
        for ind in sorted(singleSampleInds):
            cluster_centers_indices.insert(ind, ind)
        cluster_centers_indices = np.asarray(cluster_centers_indices)

    _centers = np.unique(cluster_centers_indices)
    lookup = {v: i for i, v in enumerate(_centers)}
    labels = np.array([lookup[x] for x in cluster_centers_indices], dtype=int)

    if return_n_iter:
        return cluster_centers_indices, labels, it + 1
    else:
        return cluster_centers_indices, labels


def affinity_propagation(S, preference=None, convergence_iter=15, max_iter=200,
                         damping=0.5, copy=True, verbose=False,
                         return_n_iter=False):
    """Perform Affinity Propagation Clustering of data

    Read more in the :ref:`User Guide <affinity_propagation>`.

    Parameters
    ----------

    S : array-like, shape (n_samples, n_samples)
        Matrix of similarities between points

    preference : array-like, shape (n_samples,) or float, optional
        Preferences for each point - points with larger values of
        preferences are more likely to be chosen as exemplars. The number of
        exemplars, i.e. of clusters, is influenced by the input preferences
        value. If the preferences are not passed as arguments, they will be
        set to the median of the input similarities (resulting in a moderate
        number of clusters). For a smaller amount of clusters, this can be set
        to the minimum value of the similarities.

    convergence_iter : int, optional, default: 15
        Number of iterations with no change in the number
        of estimated clusters that stops the convergence.

    max_iter : int, optional, default: 200
        Maximum number of iterations

    damping : float, optional, default: 0.5
        Damping factor between 0.5 and 1.

    copy : boolean, optional, default: True
        If copy is False, the affinity matrix is modified inplace by the
        algorithm, for memory efficiency

    verbose : boolean, optional, default: False
        The verbosity level

    return_n_iter : bool, default False
        Whether or not to return the number of iterations.

    Returns
    -------

    cluster_centers_indices : array, shape (n_clusters,)
        index of clusters centers

    labels : array, shape (n_samples,)
        cluster labels for each point

    n_iter : int
        number of iterations run. Returned only if `return_n_iter` is
        set to True.

    Notes
    -----
    See examples/cluster/plot_affinity_propagation.py for an example.

    References
    ----------
    Brendan J. Frey and Delbert Dueck, "Clustering by Passing Messages
    Between Data Points", Science Feb. 2007
    """
    S = as_float_array(S, copy=copy)
    n_samples = S.shape[0]

    if S.shape[0] != S.shape[1]:
        raise ValueError("S must be a square array (shape=%s)" % repr(S.shape))

    if preference is None:
        preference = np.median(S)
    if damping < 0.5 or damping >= 1:
        raise ValueError('damping must be >= 0.5 and < 1')

    random_state = np.random.RandomState(0)

    # Place preference on the diagonal of S
    S.flat[::(n_samples + 1)] = preference

    A = np.zeros((n_samples, n_samples))
    R = np.zeros((n_samples, n_samples))  # Initialize messages
    # Intermediate results
    tmp = np.zeros((n_samples, n_samples))

    # Remove degeneracies
    S += ((np.finfo(np.double).eps * S + np.finfo(np.double).tiny * 100) *
          random_state.randn(n_samples, n_samples))

    # Execute parallel affinity propagation updates
    e = np.zeros((n_samples, convergence_iter))

    ind = np.arange(n_samples)

    for it in range(max_iter):
        # tmp = A + S; compute responsibilities
        np.add(A, S, tmp)
        I = np.argmax(tmp, axis=1)
        Y = tmp[ind, I]  # np.max(A + S, axis=1)
        tmp[ind, I] = -np.inf
        Y2 = np.max(tmp, axis=1)

        # tmp = Rnew
        np.subtract(S, Y[:, None], tmp)
        tmp[ind, I] = S[ind, I] - Y2

        # Damping
        tmp *= 1 - damping
        R *= damping
        R += tmp

        # tmp = Rp; compute availabilities
        np.maximum(R, 0, tmp)
        tmp.flat[::n_samples + 1] = R.flat[::n_samples + 1]

        # tmp = -Anew
        tmp -= np.sum(tmp, axis=0)
        dA = np.diag(tmp).copy()
        tmp.clip(0, np.inf, tmp)
        tmp.flat[::n_samples + 1] = dA

        # Damping
        tmp *= 1 - damping
        A *= damping
        A -= tmp

        # Check for convergence
        E = (np.diag(A) + np.diag(R)) > 0
        e[:, it % convergence_iter] = E
        K = np.sum(E, axis=0)

        if it >= convergence_iter:
            se = np.sum(e, axis=1)
            unconverged = (np.sum((se == convergence_iter) + (se == 0))
                           != n_samples)
            if (not unconverged and (K > 0)) or (it == max_iter):
                if verbose:
                    print("Converged after %d iterations." % it)
                break
    else:
        if verbose:
            print("Did not converge")

    I = np.where(np.diag(A + R) > 0)[0]
    K = I.size  # Identify exemplars

    if K > 0:
        c = np.argmax(S[:, I], axis=1)
        c[I] = np.arange(K)  # Identify clusters
        # Refine the final set of exemplars and clusters and return results
        for k in range(K):
            ii = np.where(c == k)[0]
            j = np.argmax(np.sum(S[ii[:, np.newaxis], ii], axis=0))
            I[k] = ii[j]

        c = np.argmax(S[:, I], axis=1)
        c[I] = np.arange(K)
        labels = I[c]
        # Reduce labels to a sorted, gapless, list
        cluster_centers_indices = np.unique(labels)
        labels = np.searchsorted(cluster_centers_indices, labels)
    else:
        labels = np.empty((n_samples, 1))
        cluster_centers_indices = None
        labels.fill(np.nan)

    if return_n_iter:
        return cluster_centers_indices, labels, it + 1
    else:
        return cluster_centers_indices, labels


###############################################################################

class AffinityPropagation(BaseEstimator, ClusterMixin):
    """Perform Affinity Propagation Clustering of data.

    Read more in the :ref:`User Guide <affinity_propagation>`.

    Parameters
    ----------
    damping : float, optional, default: 0.5
        Damping factor between 0.5 and 1.

    convergence_iter : int, optional, default: 15
        Number of iterations with no change in the number
        of estimated clusters that stops the convergence.

    max_iter : int, optional, default: 200
        Maximum number of iterations.

    copy : boolean, optional, default: True
        Make a copy of input data.

    preference : array-like, shape (n_samples,) or float, optional
        Preferences for each point - points with larger values of
        preferences are more likely to be chosen as exemplars. The number
        of exemplars, ie of clusters, is influenced by the input
        preferences value. If the preferences are not passed as arguments,
        they will be set to the median of the input similarities.

    affinity : string, optional, default=``euclidean``
        Which affinity to use. At the moment ``precomputed`` and
        ``euclidean`` are supported. ``euclidean`` uses the
        negative squared euclidean distance between points.

    verbose : boolean, optional, default: False
        Whether to be verbose.


    Attributes
    ----------
    cluster_centers_indices_ : array, shape (n_clusters,)
        Indices of cluster centers

    cluster_centers_ : array, shape (n_clusters, n_features)
        Cluster centers (if affinity != ``precomputed``).

    labels_ : array, shape (n_samples,)
        Labels of each point

    affinity_matrix_ : array, shape (n_samples, n_samples)
        Stores the affinity matrix used in ``fit``.

    n_iter_ : int
        Number of iterations taken to converge.

    Notes
    -----
    See examples/cluster/plot_affinity_propagation.py for an example.

    The algorithmic complexity of affinity propagation is quadratic
    in the number of points.

    References
    ----------

    Brendan J. Frey and Delbert Dueck, "Clustering by Passing Messages
    Between Data Points", Science Feb. 2007
    """

    def __init__(self, damping=.5, max_iter=200, convergence_iter=15,
                 copy=True, preference=None, affinity='euclidean',
                 verbose=False):

        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.copy = copy
        self.verbose = verbose
        self.preference = preference
        self.affinity = affinity

    @property
    def _pairwise(self):
        return self.affinity == "precomputed"

    def fit(self, X, y=None):
        """ Create affinity matrix from negative euclidean distances, then
        apply affinity propagation clustering.

        Parameters
        ----------

        X: array-like, shape (n_samples, n_features) or (n_samples, n_samples)
            Data matrix or, if affinity is ``precomputed``, matrix of
            similarities / affinities.
        """
        X = check_array(X, accept_sparse='csr')
        if self.affinity == "precomputed":
            self.affinity_matrix_ = X
        elif self.affinity == "euclidean":
            self.affinity_matrix_ = -euclidean_distances(X, squared=True)
        else:
            raise ValueError("Affinity must be 'precomputed' or "
                             "'euclidean'. Got %s instead"
                             % str(self.affinity))

        # dense matrix or sparse?
        _ap = affinity_propagation if type(X) == np.ndarray else sparse_ap

        self.cluster_centers_indices_, self.labels_, self.n_iter_ = \
            _ap(
                self.affinity_matrix_, self.preference, max_iter=self.max_iter,
                convergence_iter=self.convergence_iter, damping=self.damping,
                copy=self.copy, verbose=self.verbose, return_n_iter=True)

        if self.affinity != "precomputed":
            self.cluster_centers_ = X[self.cluster_centers_indices_].copy()

        return self

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : array, shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self, "cluster_centers_indices_")
        if not hasattr(self, "cluster_centers_"):
            raise ValueError("Predict method is not supported when "
                             "affinity='precomputed'.")

        return pairwise_distances_argmin(X, self.cluster_centers_)
