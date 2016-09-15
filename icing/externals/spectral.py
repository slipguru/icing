# -*- coding: utf-8 -*-
"""Algorithms for spectral clustering"""

# Author: Gael Varoquaux gael.varoquaux@normalesup.org
#         Brian Cheung
#         Wei LI <kuantkid@gmail.com>
# License: BSD 3 clause
import warnings

import numpy as np

from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import spectral_embedding
from sklearn.cluster.k_means_ import k_means

from sklearn.cluster import SpectralClustering
from sklearn.cluster.spectral import discretize


def spectral_clustering(affinity, n_clusters=8, n_components=None,
                        eigen_solver=None, random_state=None, n_init=10,
                        eigen_tol=0.0, assign_labels='kmeans',
                        norm_laplacian=True):
    """Apply clustering to a projection to the normalized laplacian.

    In practice Spectral Clustering is very useful when the structure of
    the individual clusters is highly non-convex or more generally when
    a measure of the center and spread of the cluster is not a suitable
    description of the complete cluster. For instance when clusters are
    nested circles on the 2D plan.

    If affinity is the adjacency matrix of a graph, this method can be
    used to find normalized graph cuts.

    Read more in the :ref:`User Guide <spectral_clustering>`.

    Parameters
    -----------
    affinity : array-like or sparse matrix, shape: (n_samples, n_samples)
        The affinity matrix describing the relationship of the samples to
        embed. **Must be symmetric**.

        Possible examples:
          - adjacency matrix of a graph,
          - heat kernel of the pairwise distance matrix of the samples,
          - symmetric k-nearest neighbours connectivity matrix of the samples.

    n_clusters : integer, optional
        Number of clusters to extract.

    n_components : integer, optional, default is n_clusters
        Number of eigen vectors to use for the spectral embedding

    eigen_solver : {None, 'arpack', 'lobpcg', or 'amg'}
        The eigenvalue decomposition strategy to use. AMG requires pyamg
        to be installed. It can be faster on very large, sparse problems,
        but may also lead to instabilities

    random_state : int seed, RandomState instance, or None (default)
        A pseudo random number generator used for the initialization
        of the lobpcg eigen vectors decomposition when eigen_solver == 'amg'
        and by the K-Means initialization.

    n_init : int, optional, default: 10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    eigen_tol : float, optional, default: 0.0
        Stopping criterion for eigendecomposition of the Laplacian matrix
        when using arpack eigen_solver.

    assign_labels : {'kmeans', 'discretize'}, default: 'kmeans'
        The strategy to use to assign labels in the embedding
        space.  There are two ways to assign labels after the laplacian
        embedding.  k-means can be applied and is a popular choice. But it can
        also be sensitive to initialization. Discretization is another
        approach which is less sensitive to random initialization. See
        the 'Multiclass spectral clustering' paper referenced below for
        more details on the discretization approach.

    Returns
    -------
    labels : array of integers, shape: n_samples
        The labels of the clusters.

    References
    ----------

    - Normalized cuts and image segmentation, 2000
      Jianbo Shi, Jitendra Malik
      http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.160.2324

    - A Tutorial on Spectral Clustering, 2007
      Ulrike von Luxburg
      http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.165.9323

    - Multiclass spectral clustering, 2003
      Stella X. Yu, Jianbo Shi
      http://www1.icsi.berkeley.edu/~stellayu/publication/doc/2003kwayICCV.pdf

    Notes
    ------
    The graph should contain only one connect component, elsewhere
    the results make little sense.

    This algorithm solves the normalized cut for k=2: it is a
    normalized spectral clustering.
    """
    if assign_labels not in ('kmeans', 'discretize'):
        raise ValueError("The 'assign_labels' parameter should be "
                         "'kmeans' or 'discretize', but '%s' was given"
                         % assign_labels)

    random_state = check_random_state(random_state)
    n_components = n_clusters if n_components is None else n_components
    maps = spectral_embedding(affinity, n_components=n_components,
                              eigen_solver=eigen_solver,
                              random_state=random_state,
                              eigen_tol=eigen_tol, drop_first=False,
                              norm_laplacian=norm_laplacian)

    if assign_labels == 'kmeans':
        _, labels, _ = k_means(maps, n_clusters, random_state=random_state,
                               n_init=n_init)
    else:
        labels = discretize(maps, random_state=random_state)

    return labels


class SpectralClustering(SpectralClustering):
    """Apply clustering to a projection to the normalized laplacian.

    In practice Spectral Clustering is very useful when the structure of
    the individual clusters is highly non-convex or more generally when
    a measure of the center and spread of the cluster is not a suitable
    description of the complete cluster. For instance when clusters are
    nested circles on the 2D plan.

    If affinity is the adjacency matrix of a graph, this method can be
    used to find normalized graph cuts.

    When calling ``fit``, an affinity matrix is constructed using either
    kernel function such the Gaussian (aka RBF) kernel of the euclidean
    distanced ``d(X, X)``::

            np.exp(-gamma * d(X,X) ** 2)

    or a k-nearest neighbors connectivity matrix.

    Alternatively, using ``precomputed``, a user-provided affinity
    matrix can be used.

    Read more in the :ref:`User Guide <spectral_clustering>`.

    Parameters
    -----------
    n_clusters : integer, optional
        The dimension of the projection subspace.

    affinity : string, array-like or callable, default 'rbf'
        If a string, this may be one of 'nearest_neighbors', 'precomputed',
        'rbf' or one of the kernels supported by
        `sklearn.metrics.pairwise_kernels`.

        Only kernels that produce similarity scores (non-negative values that
        increase with similarity) should be used. This property is not checked
        by the clustering algorithm.

    gamma : float
        Scaling factor of RBF, polynomial, exponential chi^2 and
        sigmoid affinity kernel. Ignored for
        ``affinity='nearest_neighbors'``.

    degree : float, default=3
        Degree of the polynomial kernel. Ignored by other kernels.

    coef0 : float, default=1
        Zero coefficient for polynomial and sigmoid kernels.
        Ignored by other kernels.

    n_neighbors : integer
        Number of neighbors to use when constructing the affinity matrix using
        the nearest neighbors method. Ignored for ``affinity='rbf'``.

    eigen_solver : {None, 'arpack', 'lobpcg', or 'amg'}
        The eigenvalue decomposition strategy to use. AMG requires pyamg
        to be installed. It can be faster on very large, sparse problems,
        but may also lead to instabilities

    random_state : int seed, RandomState instance, or None (default)
        A pseudo random number generator used for the initialization
        of the lobpcg eigen vectors decomposition when eigen_solver == 'amg'
        and by the K-Means initialization.

    n_init : int, optional, default: 10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    eigen_tol : float, optional, default: 0.0
        Stopping criterion for eigendecomposition of the Laplacian matrix
        when using arpack eigen_solver.

    assign_labels : {'kmeans', 'discretize'}, default: 'kmeans'
        The strategy to use to assign labels in the embedding
        space. There are two ways to assign labels after the laplacian
        embedding. k-means can be applied and is a popular choice. But it can
        also be sensitive to initialization. Discretization is another approach
        which is less sensitive to random initialization.

    kernel_params : dictionary of string to any, optional
        Parameters (keyword arguments) and values for kernel passed as
        callable object. Ignored by other kernels.

    Attributes
    ----------
    affinity_matrix_ : array-like, shape (n_samples, n_samples)
        Affinity matrix used for clustering. Available only if after calling
        ``fit``.

    labels_ :
        Labels of each point

    Notes
    -----
    If you have an affinity matrix, such as a distance matrix,
    for which 0 means identical elements, and high values means
    very dissimilar elements, it can be transformed in a
    similarity matrix that is well suited for the algorithm by
    applying the Gaussian (RBF, heat) kernel::

        np.exp(- X ** 2 / (2. * delta ** 2))

    Another alternative is to take a symmetric version of the k
    nearest neighbors connectivity matrix of the points.

    If the pyamg package is installed, it is used: this greatly
    speeds up computation.

    References
    ----------

    - Normalized cuts and image segmentation, 2000
      Jianbo Shi, Jitendra Malik
      http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.160.2324

    - A Tutorial on Spectral Clustering, 2007
      Ulrike von Luxburg
      http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.165.9323

    - Multiclass spectral clustering, 2003
      Stella X. Yu, Jianbo Shi
      http://www1.icsi.berkeley.edu/~stellayu/publication/doc/2003kwayICCV.pdf
    """

    def __init__(self, n_clusters=8, eigen_solver=None, random_state=None,
                 n_init=10, gamma=1., affinity='rbf', n_neighbors=10,
                 eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1,
                 kernel_params=None, norm_laplacian=True):
        super(SpectralClustering, self).__init__(
            n_clusters=n_clusters, eigen_solver=eigen_solver,
            random_state=random_state, n_init=n_init, gamma=gamma,
            affinity=affinity, n_neighbors=n_neighbors, eigen_tol=eigen_tol,
            assign_labels=assign_labels, degree=degree, coef0=coef0,
            kernel_params=kernel_params)
        self.norm_laplacian = norm_laplacian

    def fit(self, X, y=None):
        """Creates an affinity matrix for X using the selected affinity,
        then applies spectral clustering to this affinity matrix.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            OR, if affinity==`precomputed`, a precomputed affinity
            matrix of shape (n_samples, n_samples)
        """
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'],
                        dtype=np.float64)
        if X.shape[0] == X.shape[1] and self.affinity != "precomputed":
            warnings.warn("The spectral clustering API has changed. ``fit``"
                          "now constructs an affinity matrix from data. To use"
                          " a custom affinity matrix, "
                          "set ``affinity=precomputed``.")

        if self.affinity == 'nearest_neighbors':
            connectivity = kneighbors_graph(X, n_neighbors=self.n_neighbors, include_self=True)
            self.affinity_matrix_ = 0.5 * (connectivity + connectivity.T)
        elif self.affinity == 'precomputed':
            self.affinity_matrix_ = X
        else:
            params = self.kernel_params
            if params is None:
                params = {}
            if not callable(self.affinity):
                params['gamma'] = self.gamma
                params['degree'] = self.degree
                params['coef0'] = self.coef0
            self.affinity_matrix_ = pairwise_kernels(X, metric=self.affinity,
                                                     filter_params=True,
                                                     **params)

        random_state = check_random_state(self.random_state)
        self.labels_ = spectral_clustering(self.affinity_matrix_,
                                           n_clusters=self.n_clusters,
                                           eigen_solver=self.eigen_solver,
                                           random_state=random_state,
                                           n_init=self.n_init,
                                           eigen_tol=self.eigen_tol,
                                           assign_labels=self.assign_labels,
                                           norm_laplacian=self.norm_laplacian)
        return self
