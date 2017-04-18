"""TODO."""
import logging
import numpy as np
import scipy
import fastcluster

from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import DBSCAN
from sklearn.utils.sparsetools import connected_components

from icing.externals import AffinityPropagation
from icing.utils import extra


def define_clusts(similarity_matrix, threshold=0.05, max_iter=200,
                  method='ap'):
    """Define clusters given the similarity matrix and the threshold."""
    n, labels = connected_components(similarity_matrix, directed=False)
    prev_max_clust = 0
    print("connected components: %d" % n)
    clusters = labels.copy()

    if method == 'dbscan':
        ap = DBSCAN(metric='precomputed', min_samples=1, eps=.2, n_jobs=-1)
    if method == 'ap':
        ap = AffinityPropagation(affinity='precomputed', max_iter=max_iter,
                                 preference='median')

    for i in range(n):
        idxs = np.where(labels == i)[0]
        if idxs.shape[0] > 1:
            sm = similarity_matrix[idxs][:, idxs]
            sm += sm.T + scipy.sparse.eye(sm.shape[0])

            # Hierarchical clustering
            if method == 'hc':
                dists = squareform(1 - sm.toarray())
                links = fastcluster.linkage(dists, method='ward')
                try:
                    clusters_ = fcluster(links, threshold, 'distance')
                except ValueError as err:
                    logging.critical(err)
                    clusters_ = np.zeros(1, dtype=int)

            # DBSCAN
            elif method == 'dbscan':
                db = ap.fit(1. - sm.toarray())
                # Number of clusters in labels, ignoring noise if present.
                clusters_ = db.labels_
                # n_clusters_ = len(set(clusters_)) - int(0 in clusters_)

            # AffinityPropagation
            # ap = AffinityPropagation(affinity='precomputed')
            elif method == 'ap':
                db = ap.fit(sm)
                clusters_ = db.labels_
            else:
                raise ValueError("clustering method %s unknown" % method)

            if np.min(clusters_) == 0:
                clusters_ += 1
            clusters_ += prev_max_clust
            clusters[idxs] = clusters_
            prev_max_clust = max(clusters_)
        else:  # connected component contains just 1 element
            prev_max_clust += 1
            clusters[idxs] = prev_max_clust
    return np.array(extra.flatten(clusters))
