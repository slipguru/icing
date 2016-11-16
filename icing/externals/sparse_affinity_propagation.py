"""Affinity Propagation clustering algorithm.

This module is an extension of sklearn's Affinity Propagation to take into
account sparse matrices, which are not currently supported.

Author: Federico Tomasi
Copyright (c) 2016, Federico Tomasi.
Licensed under the FreeBSD license (see LICENSE.txt).

Original authors for sklearn:

# Author: Alexandre Gramfort alexandre.gramfort@inria.fr
#        Gael Varoquaux gael.varoquaux@normalesup.org

# License: BSD 3 clause
"""
import numpy as np

from scipy.sparse import coo_matrix
from scipy.sparse import issparse
from sklearn.cluster import AffinityPropagation
from sklearn.utils import as_float_array, check_array
from sklearn.metrics import euclidean_distances

from ._sparse_affinity_propagation import (
    parse_preference, _set_sparse_diagonal,
    _sparse_row_maxindex, _sparse_row_sum_update,
    _update_r_max_row, remove_single_samples)


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
        algorithm, for memory efficiency.
        Unused, but left for sklearn affinity propagation consistency.

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
    # rows, cols, data = matrix_to_row_col_data(S)
    rows, cols, data = S.row, S.col, as_float_array(S.data, copy=copy)
    n_samples = S.shape[0]

    preferences = parse_preference(preference, n_samples, data)
    if damping < 0.5 or damping >= 1:
        raise ValueError('damping must be >= 0.5 and < 1')
    if convergence_percentage is None:
        convergence_percentage = 1

    random_state = np.random.RandomState(0)

    # Place preference on the diagonal of S
    rows, cols, data = _set_sparse_diagonal(rows, cols, data, preferences)
    rows, cols, data, left_ori_dict, idx_single_samples, n_samples = \
        remove_single_samples(rows, cols, data, n_samples)

    data_len = data.shape[0]
    row_indptr = np.append(
        np.insert(np.where(np.diff(rows) > 0)[0] + 1, 0, 0), data_len)

    row_to_col_idx = np.lexsort((rows, cols))
    rows_colbased = rows[row_to_col_idx]
    cols_colbased = cols[row_to_col_idx]
    col_to_row_idx = np.lexsort((cols_colbased, rows_colbased))

    col_indptr = np.append(
        np.insert(np.where(np.diff(cols_colbased) > 0)[0] + 1, 0, 0), data_len)

    kk_row_index = np.where(rows == cols)[0]
    kk_col_index = np.where(rows_colbased == cols_colbased)[0]

    # Initialize messages
    A = np.zeros(data_len, dtype=float)
    R = np.zeros(data_len, dtype=float)
    # Intermediate results
    tmp = np.zeros(data_len, dtype=float)

    # Remove degeneracies
    data += ((np.finfo(np.double).eps * data + np.finfo(np.double).tiny * 100)
             * random_state.randn(data_len))

    labels = np.empty(0, dtype=np.int)
    last_labels = None
    convergence_count = 0

    for it in range(max_iter):
        last_labels = labels.copy()

        # tmp = A + S; compute responsibilities
        np.add(A, data, tmp)
        np.subtract(data, _update_r_max_row(tmp, row_indptr), tmp)

        # Damping
        tmp *= 1. - damping
        R *= damping
        R += tmp

        # tmp = Rp; compute availabilities
        np.maximum(R, 0, tmp)
        tmp[kk_row_index] = R[kk_row_index]

        # tmp = -Anew
        tmp = tmp[row_to_col_idx]
        _sparse_row_sum_update(tmp, col_indptr, kk_col_index)
        tmp = tmp[col_to_row_idx]

        # Damping
        tmp *= 1. - damping
        A *= damping
        A -= tmp

        # Check for convergence
        labels = _sparse_row_maxindex(A + R, row_indptr)
        if labels.shape[0] == 0 or np.sum(last_labels == labels) / \
                float(labels.shape[0]) < convergence_percentage:
            convergence_count = 0
        else:
            convergence_count += 1
        if convergence_count == convergence_iter:
            if verbose:
                print("Converged after %d iterations." % it)
            break
    else:
        if verbose:
            print("Did not converge")

    if idx_single_samples is None or len(idx_single_samples) == 0:
        cluster_centers_indices = cols[labels]
    else:
        cluster_centers_indices = [left_ori_dict[el] for el in cols[labels]]
        for ind in sorted(idx_single_samples):
            cluster_centers_indices.insert(ind, ind)
        cluster_centers_indices = np.asarray(cluster_centers_indices)

    _centers = np.unique(cluster_centers_indices)
    lookup = {v: i for i, v in enumerate(_centers)}
    labels = np.array([lookup[x] for x in cluster_centers_indices], dtype=int)

    if return_n_iter:
        return cluster_centers_indices, labels, it + 1
    else:
        return cluster_centers_indices, labels


###############################################################################

class AffinityPropagation(AffinityPropagation):
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
                 verbose=False, convergence_percentage=0.999999):
        super(AffinityPropagation, self).__init__(
            damping=damping,
            max_iter=max_iter,
            convergence_iter=convergence_iter,
            copy=copy,
            verbose=verbose,
            preference=preference,
            affinity=affinity)
        self.convergence_percentage = convergence_percentage

    def fit(self, X, **kwargs):
        """Apply affinity propagation clustering.

        Create affinity matrix from negative euclidean distances if required.

        Parameters
        ----------
        X: array-like or sparse matrix,
                shape (n_samples, n_features) or (n_samples, n_samples)
            Data matrix or, if affinity is ``precomputed``, matrix of
            similarities / affinities.
        """
        if not issparse(X):
            return super(AffinityPropagation, self).fit(X, **kwargs)

        # Since X is sparse, this converts it in a coo_matrix if required
        X = check_array(X, accept_sparse='coo')
        if self.affinity == "precomputed":
            self.affinity_matrix_ = X
        elif self.affinity == "euclidean":
            self.affinity_matrix_ = coo_matrix(
                -euclidean_distances(X, squared=True))
        else:
            raise ValueError("Affinity must be 'precomputed' or "
                             "'euclidean'. Got %s instead"
                             % str(self.affinity))

        self.cluster_centers_indices_, self.labels_, self.n_iter_ = \
            sparse_ap(
                self.affinity_matrix_, self.preference, max_iter=self.max_iter,
                convergence_iter=self.convergence_iter, damping=self.damping,
                copy=self.copy, verbose=self.verbose, return_n_iter=True,
                convergence_percentage=self.convergence_percentage)

        if self.affinity != "precomputed":
            self.cluster_centers_ = X.data[self.cluster_centers_indices_].copy()

        return self
