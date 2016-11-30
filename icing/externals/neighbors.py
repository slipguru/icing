"""Portion of file taken from sklearn.neighbors.base."""
import numpy as np
from scipy.sparse import csr_matrix, issparse
import multiprocessing as mp
import itertools

from sklearn.metrics import pairwise_distances
from sklearn.utils import check_array

def _func(X, Y):
    # Only calculate metric for upper triangle
    out = np.zeros((X.shape[0], Y.shape[0]), dtype='int')
    iterator = itertools.product(range(X.shape[0]), range(Y.shape[0]))
    for i, j in iterator:
        out[i, j] = int(len(set(X[i].getVGene('set')) &
                            set(Y[j].getVGene('set'))) > 0 or
                        len(set(X[i].getJGene('set')) &
                            set(Y[j].getJGene('set'))) > 0)

    return out

def radius_neighbors(X=None, radius=None, return_distance=True):
    """Finds the neighbors within a given radius of a point or points.
    Return the indices and distances of each point from the dataset
    lying in a ball with size ``radius`` around the points of the query
    array. Points lying on the boundary are included in the results.
    The result points are *not* necessarily sorted by distance to their
    query point.
    Parameters
    ----------
    X : array-like, (n_samples, n_features), optional
        The query point or points.
        If not provided, neighbors of each indexed point are returned.
        In this case, the query point is not considered its own neighbor.
    radius : float
        Limiting distance of neighbors to return.
        (default is the value passed to the constructor).
    return_distance : boolean, optional. Defaults to True.
        If False, distances will not be returned
    Returns
    -------
    dist : array, shape (n_samples,) of arrays
        Array representing the distances to each point, only present if
        return_distance=True. The distance values are computed according
        to the ``metric`` constructor parameter.
    ind : array, shape (n_samples,) of arrays
        An array of arrays of indices of the approximate nearest points
        from the population matrix that lie within a ball of size
        ``radius`` around the query points.
    Examples
    --------
    In the following example, we construct a NeighborsClassifier
    class from an array representing our data set and ask who's
    the closest point to [1, 1, 1]:
    >>> import numpy as np
    >>> samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
    >>> from sklearn.neighbors import NearestNeighbors
    >>> neigh = NearestNeighbors(radius=1.6)
    >>> neigh.fit(samples) # doctest: +ELLIPSIS
    NearestNeighbors(algorithm='auto', leaf_size=30, ...)
    >>> rng = neigh.radius_neighbors([[1., 1., 1.]])
    >>> print(np.asarray(rng[0][0])) # doctest: +ELLIPSIS
    [ 1.5  0.5]
    >>> print(np.asarray(rng[1][0])) # doctest: +ELLIPSIS
    [1 2]
    The first array returned contains the distances to all points which
    are closer than 1.6, while the second array returned contains their
    indices.  In general, multiple points can be queried at the same time.
    Notes
    -----
    Because the number of neighbors of each point is not necessarily
    equal, the results for multiple query points cannot be fit in a
    standard data array.
    For efficiency, `radius_neighbors` returns arrays of objects, where
    each object is a 1D array of indices or distances.
    """
    n_samples = X.shape[0]
    from joblib import delayed, Parallel
    from sklearn.utils import gen_even_slices
    n_jobs = max(mp.cpu_count(), n_samples)
    print(gen_even_slices(X.shape[0], n_jobs))
    fd = delayed(_func)
    dist = Parallel(n_jobs=n_jobs, verbose=0)(
        fd(X, X[s])
        for s in gen_even_slices(X.shape[0], n_jobs))

    neigh_ind_list = [np.where(d > 0)[0] for d in np.hstack(dist)]
    # print(neigh_ind_list)


    # See https://github.com/numpy/numpy/issues/5456
    # if you want to understand why this is initialized this way.
    neigh_ind = np.empty(n_samples, dtype='object')
    neigh_ind[:] = neigh_ind_list

    return neigh_ind


def radius_neighbors_graph(X=None, radius=None, mode='connectivity'):
    """Computes the (weighted) graph of Neighbors for points in X
    Neighborhoods are restricted the points at a distance lower than
    radius.
    Parameters
    ----------
    X : array-like, shape = [n_samples, n_features], optional
        The query point or points.
        If not provided, neighbors of each indexed point are returned.
        In this case, the query point is not considered its own neighbor.
    radius : float
        Radius of neighborhoods.
        (default is the value passed to the constructor).
    mode : {'connectivity', 'distance'}, optional
        Type of returned matrix: 'connectivity' will return the
        connectivity matrix with ones and zeros, in 'distance' the
        edges are Euclidean distance between points.
    Returns
    -------
    A : sparse matrix in CSR format, shape = [n_samples, n_samples]
        A[i, j] is assigned the weight of edge that connects i to j.
    Examples
    --------
    >>> X = [[0], [3], [1]]
    >>> from sklearn.neighbors import NearestNeighbors
    >>> neigh = NearestNeighbors(radius=1.5)
    >>> neigh.fit(X) # doctest: +ELLIPSIS
    NearestNeighbors(algorithm='auto', leaf_size=30, ...)
    >>> A = neigh.radius_neighbors_graph(X)
    >>> A.toarray()
    array([[ 1.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 1.,  0.,  1.]])
    See also
    --------
    kneighbors_graph
    """
    # construct CSR matrix representation of the NN graph
    if mode == 'connectivity':
        A_ind = radius_neighbors(X, radius, return_distance=False)
        A_data = None
    else:
        raise ValueError(
            'Unsupported mode, must be "connectivity" '
            'but got %s instead' % mode)

    n_samples1 = A_ind.shape[0]
    n_neighbors = np.array([len(a) for a in A_ind])
    A_ind = np.concatenate(list(A_ind))
    if A_data is None:
        A_data = np.ones(len(A_ind))
    A_indptr = np.concatenate((np.zeros(1, dtype=int),
                               np.cumsum(n_neighbors)))

    sparse = csr_matrix((A_data, A_ind, A_indptr),
                        shape=(n_samples1, n_samples1))
    import scipy
    return sparse + sparse.T + scipy.sparse.eye(sparse.shape[0])
