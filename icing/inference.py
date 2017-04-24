"""Compatibility module for sklearn-like ICING usage."""
import os
import logging
import numpy as np
import gzip

from functools import partial
from six.moves import cPickle as pkl
from sklearn.base import BaseEstimator
from sklearn.cluster import DBSCAN, MiniBatchKMeans

from icing.core.cluster import define_clusts
from icing.core.distances import distance_dataframe, StringKernelDistance
from icing.similarity_ import compute_similarity_matrix
from icing.utils import extra


class DefineClones(BaseEstimator):
    """Clustering container for defining final clones."""

    def __init__(
        self, tag='debug', root=None, cluster='ap', igsimilarity=None,
            threshold=0.05, compute_similarity=True, clustering=None):
        """Description of params."""
        self.tag = tag
        self.root = root
        self.cluster = cluster
        self.igsimilarity = igsimilarity
        self.threshold = threshold
        self.compute_similarity = compute_similarity
        self.clustering = clustering

    @property
    def save_results(self):
        return self.root is not None

    def fit(self, records, db_name=None):
        """Run docstings."""
        if self.save_results and not os.path.exists(self.root):
            if self.root is None:
                self.root = 'results_' + self.tag + extra.get_time()
            os.makedirs(self.root)
            logging.warn("No root folder supplied, folder %s "
                         "created", os.path.abspath(self.root))

        if self.save_results:
            output_filename = self.tag
            output_folder = os.path.join(self.root, output_filename)

            # Create exp folder into the root folder
            os.makedirs(output_folder)

        if self.compute_similarity:
            similarity_matrix = compute_similarity_matrix(
                records, sparse_mode=True,
                igsimilarity=self.igsimilarity)

            if self.save_results:
                sm_filename = output_filename + '_similarity_matrix.pkl.tz'
                try:
                    pkl.dump(similarity_matrix, gzip.open(
                        os.path.join(output_folder, sm_filename), 'w+'))
                    logging.info("Dumped similarity matrix: %s",
                                 os.path.join(output_folder, sm_filename))
                except OverflowError:
                    logging.error("Cannot dump similarity matrix")

            logging.info("Start define_clusts function ...")
            labels = define_clusts(
                similarity_matrix, threshold=self.threshold,
                method=self.cluster)
        else:
            # use a method which does not require an explicit similarity_matrix
            # first, encode the IgRecords into strings
            X_string = [x.features for x in records]
            X_string = np.array(X_string, dtype=object)
            logging.info("Start clonal inference ...")
            from icing.core.distances import is_distance
            if not is_distance(self.igsimilarity):
                raise ValueError("If not computing similarity matrix, "
                                 "you need to use a distance metric. "
                                 "See icing.core.distances")
            if self.clustering is None:
                raise ValueError("If not computing similarity matrix, "
                                 "you need to pass a clustering method")

            self.clustering.metric = partial(self.clustering.metric, X_string)
            print self.clustering
            # Fit on a index array
            # see https://github.com/scikit-learn/scikit-learn/issues/3737
            # labels = self.clustering.fit_predict(X_string)
            labels = self.clustering.fit_predict(
                np.arange(X_string.shape[0]).reshape(-1, 1))

        # Number of clusters in labels, ignoring noise if present.
        n_clones = len(set(labels)) - (1 if -1 in labels else 0)
        if self.cluster.lower() == 'ap':
            # log only number of clones
            logging.critical("Number of clones: %i", n_clones)
        else:
            logging.critical(
                "Number of clones: %i, threshold %.3f", n_clones,
                self.threshold)
        if self.save_results:
            with open(os.path.join(output_folder, 'summary.txt'), 'w') as f:
                f.write("filename: %s\n" % db_name)
                f.write("clones: %i\n" % n_clones)

            cl_filename = output_filename + '_labels.pkl.tz'
            pkl.dump([labels, self.threshold], gzip.open(
                os.path.join(output_folder, cl_filename), 'w+'))
            logging.info("Dumped labels and threshold: %s",
                         os.path.join(output_folder, cl_filename))
            self.output_folder_ = output_folder

        clone_dict = {k.id: v for k, v in zip(records, labels)}
        self.clone_dict_ = clone_dict

        return self


class ICINGTwoStep(BaseEstimator):

    def __init__(self, eps=0.5, kmeans_params=None, dbscan_params=None):
        self.eps = eps
        self.dbscan_params = dbscan_params or {}
        self.kmeans_params = kmeans_params or dict(n_init=100, n_clusters=100)

    def fit(self, X, y=None, sample_weight=None):
        """X is a dataframe."""
        if not self.dbscan_params:
            self.dbscan_params = dict(
                min_samples=1, n_jobs=-1, algorithm='brute',
                metric=partial(distance_dataframe, X, **dict(
                    junction_dist=StringKernelDistance(min_kn=1, max_kn=10),
                    correct=False, tol=0)))

        self.dbscan_params['eps'] = self.eps
        # new part: group by junction and v genes
        groups = X.groupby(["v_gene_set_str", "aa_junc"]).groups
        groups_values = groups.values()  # list of lists
        idxs = np.array([elem[0] for elem in groups_values])  # take one of them
        sample_weight = np.array([len(elem) for elem in groups_values])
        X_all = idxs.reshape(-1, 1)

        if self.kmeans_params.get('n_clusters', True):
            # ensure the number of clusters is higher than points
            self.kmeans_params['n_clusters'] = min(
                self.kmeans_params['n_clusters'], X_all.shape[0])
        kmeans = MiniBatchKMeans(**self.kmeans_params)

        lengths = X['aa_junction_length'].values
        kmeans.fit(lengths[idxs].reshape(-1, 1))
        dbscan_labels = np.zeros_like(kmeans.labels_).ravel()
        for label in np.unique(kmeans.labels_):
            idx_row = np.where(kmeans.labels_ == label)[0]

            X_idx = idxs[idx_row].reshape(-1, 1)
            weights = sample_weight[idx_row]

            db_labels = DBSCAN(**self.dbscan_params).fit_predict(
                X_idx, sample_weight=weights)
            dbscan_labels[idx_row] = db_labels + np.max(dbscan_labels) + 1

        labels = dbscan_labels

        # new part: put together the labels
        labels_ext = np.zeros(X.shape[0], dtype=int)
        labels_ext[idxs] = labels
        for i, list_ in enumerate(groups_values):
            labels_ext[list_] = labels[i]
        self.labels_ = labels_ext

    def fit_predict(self, X, y=None, sample_weight=None):
        """Perform clustering on X and returns cluster labels.

        Parameters
        ----------
        X : array or sparse (CSR) matrix of shape (n_samples, n_features), or
                array of shape (n_samples, n_samples)
            A feature array, or array of distances between samples if
            ``metric='precomputed'``.
        sample_weight : array, shape (n_samples,), optional
            Weight of each sample, such that a sample with a weight of at least
            ``min_samples`` is by itself a core sample; a sample with negative
            weight may inhibit its eps-neighbor from being core.
            Note that weights are absolute, and default to 1.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            cluster labels
        """
        self.fit(X, sample_weight=sample_weight)
        return self.labels_
