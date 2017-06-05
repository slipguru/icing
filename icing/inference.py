"""Compatibility module for sklearn-like ICING usage."""
from __future__ import print_function
import os
import logging
import numpy as np
import gzip

from functools import partial
from six.moves import cPickle as pkl
from sklearn.base import BaseEstimator
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.neighbors import BallTree

from icing.core.distances import distance_dataframe, StringDistance
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
            from icing.core.cluster import define_clusts
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

    def __init__(self, eps=0.5, model='aa', kmeans_params=None,
                 dbscan_params=None, method='dbscan', hdbscan_params=None,
                 dbspark_params=None, verbose=False):
        self.eps = eps
        self.model = 'aa_' if model == 'aa' else ''
        self.dbscan_params = dbscan_params or {}
        self.kmeans_params = kmeans_params or dict(n_init=100, n_clusters=100)
        self.method = method
        self.hdbscan_params = hdbscan_params or {}
        self.dbspark_params = dbspark_params or {}
        self.verbose = verbose

    def fit(self, X, y=None, sample_weight=None):
        """X is a dataframe."""
        if self.method not in ("dbscan", "hdbscan", "spark"):
            raise ValueError("Unsupported method '%s'" % self.method)
        if not self.dbscan_params:
            self.dbscan_params = dict(
                min_samples=1, n_jobs=-1, algorithm='brute',
                metric=partial(distance_dataframe, X, **dict(
                    junction_dist=StringDistance(),
                    correct=False, tol=0)))
        if not self.hdbscan_params and self.method == 'hdbscan':
            self.hdbscan_params = dict(
                min_samples=1, n_jobs=-1,
                metric=partial(distance_dataframe, X, **dict(
                    junction_dist=StringDistance(),
                    correct=False, tol=0)))

        self.dbscan_params['eps'] = self.eps
        # new part: group by junction and v genes
        if self.method == 'hdbscan' and False:
            # no grouping; unsupported sample_weight
            groups_values = [[x] for x in np.arange(X.shape[0])]
        else:
            # list of lists
            groups_values = X.groupby(
                ["v_gene_set_str", self.model + "junc"]).groups.values()

        idxs = np.array([elem[0] for elem in groups_values])  # take one of them
        sample_weight = np.array([len(elem) for elem in groups_values])
        X_all = idxs.reshape(-1, 1)

        if self.kmeans_params.get('n_clusters', True):
            # ensure the number of clusters is higher than points
            self.kmeans_params['n_clusters'] = min(
                self.kmeans_params['n_clusters'], X_all.shape[0])
        kmeans = MiniBatchKMeans(**self.kmeans_params)

        lengths = X[self.model + 'junction_length'].values
        kmeans.fit(lengths[idxs].reshape(-1, 1))
        dbscan_labels = np.zeros_like(kmeans.labels_).ravel()

        if self.method == 'hdbscan':
            from hdbscan import HDBSCAN
            from hdbscan.prediction import all_points_membership_vectors
            dbscan_sk = HDBSCAN(**self.hdbscan_params)
        else:
            dbscan_sk = DBSCAN(**self.dbscan_params)
        if self.method == 'spark':
            from pyspark import SparkContext
            from icing.externals.pypardis import dbscan as dbpard
            sc = SparkContext.getOrCreate()
            sample_weight_map = dict(zip(idxs, sample_weight))
            # self.dbscan_params.pop('n_jobs', None)
            dbscan = dbpard.DBSCAN(
                dbscan_params=self.dbscan_params,
                **self.dbspark_params)
        # else:

        for i, label in enumerate(np.unique(kmeans.labels_)):
            idx_row = np.where(kmeans.labels_ == label)[0]

            if self.verbose:
                print("Iteration %d/%d" % (i, np.unique(kmeans.labels_).size),
                      "(%d seqs)" % idx_row.size, end='\r')

            X_idx = idxs[idx_row].reshape(-1, 1).astype('float64')
            weights = sample_weight[idx_row]

            if idx_row.size == 1:
                db_labels = np.array([0])
            elif self.method == 'spark' and idx_row.size > 5000:
                test_data = sc.parallelize(enumerate(X_idx))
                dbscan.train(test_data, sample_weight=sample_weight_map)
                db_labels = np.array(dbscan.assignments())[:, 1]
            elif self.method == 'hdbscan':
                db_labels = dbscan_sk.fit_predict(X_idx)  # unsupported weights
                # avoid noise samples
                soft_clusters = all_points_membership_vectors(dbscan_sk)
                db_labels = np.array([np.argmax(x) for x in soft_clusters])
            else:
                db_labels = dbscan_sk.fit_predict(
                    X_idx, sample_weight=weights)

            if len(dbscan_sk.core_sample_indices_) < 1:
                db_labels[:] = 0
            if -1 in db_labels:
                balltree = BallTree(
                    X_idx[dbscan_sk.core_sample_indices_],
                    metric=dbscan_sk.metric)
                noise_labels = balltree.query(
                    X_idx[db_labels == -1], k=1, return_distance=False).ravel()
                # get labels for core points, then assign to noise points based
                # on balltree
                dbscan_noise_labels = db_labels[
                    dbscan_sk.core_sample_indices_][noise_labels]
                db_labels[db_labels == -1] = dbscan_noise_labels

            # hopefully, there are no noisy samples at this time
            db_labels[db_labels > -1] = db_labels[db_labels > -1] + np.max(dbscan_labels) + 1
            dbscan_labels[idx_row] = db_labels  # + np.max(dbscan_labels) + 1

        if self.method == 'spark':
            sc.stop()
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
