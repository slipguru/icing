"""Compatibility module for sklearn-like ICING usage."""

import os
import imp
import shutil
import argparse
import logging
import time
import numpy as np
import gzip
import scipy
import fastcluster


from collections import defaultdict
from functools import partial
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import DBSCAN
# from sklearn.cluster import AffinityPropagation
from sklearn.utils.sparsetools import connected_components
from scipy import sparse
from six.moves import cPickle as pkl
from sklearn.base import BaseEstimator

import icing
from icing import __version__
from icing.core.cloning import define_clones
from icing.core.learning_function import generate_correction_function
from icing.utils import extra
from icing.utils import io

from icing.core.distances import string_distance
from icing.core.similarity_scores import similarity_score_tripartite as mwi
from icing.core.parallel_distance import sm_sparse
from icing.externals import AffinityPropagation
from icing.kernel import sum_string_kernel
from icing.models.model import model_matrix
from icing.validation import scores
from icing.utils import extra

from icing.core.cloning import define_clusts


class StringKernel(BaseEstimator):
    """Utility class for string kernel"""

    def __init__(self, min_kn=1, max_kn=2, lamda=.5,
                 check_min_length=0, hard_matching=0):
        self.min_kn = min_kn
        self.max_kn = max_kn
        self.lamda = lamda
        self.check_min_length = check_min_length
        self.hard_matching = hard_matching

    def pairwise(self, x1, x2):
        return sum_string_kernel(
            [x1, x2], verbose=False, normalize=1, return_float=1,
            min_kn=self.min_kn, max_kn=self.max_kn, lamda=self.lamda,
            check_min_length=self.check_min_length,
            hard_matching=self.hard_matching)


class StringSimilarity(BaseEstimator):
    """Utility class for string distance."""

    def __init__(self, model='ham', dist_mat=None, tol=3):
        self.model = model
        self.dist_mat = dist_mat
        self.tol = tol

        if self.dist_mat is None:
            self.dist_mat = model_matrix(model)
        self.dist_mat_max = np.max(dist_mat)

    def pairwise(self, x1, x2):
        return 1 - string_distance(
            x1, x2, len(x1), len(x2), dist_mat=self.dist_mat,
            dist_mat_max=self.dist_mat_max,
            tol=self.tol)


class IgSimilarity(BaseEstimator):

    # def __init__(self, method='jaccard', model='ham', dist_mat=None, dist_mat_max=1,
    #     tol=3, rm_duplicates=False,
    #     v_weight=1., j_weight=1., vj_weight=.5, sk_weight=.5,
    #     correction_function=(lambda _: 1), correct=True,
    #     sim_score_params=None, ssk_params=None):
    #     pass
    def __init__(
        self, junction_sim, tol=3, rm_duplicates=False,
            correct=True, correct_by=None):
        """Calculate a similarity between two input immunoglobulins.

        Parameters
        ----------
        tol : int, optional, default: 3
            Tolerance in the length of the sequences. Default is 3 (3 nucleotides
            form an amminoacid. If seq1 and seq2 represent amminoacidic sequences,
            use tol = 1).

        Returns
        -------
        similarity : float
            A normalised similarity score between ig1 and ig2. Values are in [0,1].
            0: ig1 is completely different from ig2.
            1: ig1 and ig2 are the same.

        """
        self.junction_sim = junction_sim
        self.rm_duplicates = rm_duplicates
        self.correct = correct
        self.correct_by = correct_by
        self.tol = tol

    def pairwise(self, x1, x2):
        """Compute pairwise similarity.

        Parameters
        ----------
        ig1, ig2 : externals.DbCore.IgRecord
        Instances of two immunoglobulins.
        """
        if self.rm_duplicates and x1.junc == x2.junc:
            return 0.

        if abs(x1.junction_length - x2.junction_length) > self.tol or \
                len(x1.setV & x2.setV) < 1:
            return 0.

        similarity = self.junction_sim.pairwise(x1.junc, x2.junc)
        if similarity > 0 and self.correct:
            correction = self.correct_by(np.mean((x1.mut, x2.mut)))
            similarity *= np.clip(correction, 0, 1)
        return max(similarity, 0)


class DefineClones(BaseEstimator):
    """todo."""

    def __init__(
        self, tag='debug', root=None, cluster='ap', igsimilarity=None,
            threshold=0.05):
        """Description of params."""
        self.tag = tag
        self.root = root
        self.cluster = cluster
        self.igsimilarity = igsimilarity
        self.threshold = threshold

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

        similarity_matrix = compute_similarity_matrix(
            records, sparse_mode=True,
            igsimilarity=self.igsimilarity)

        if self.save_results:
            output_filename = self.tag
            output_folder = os.path.join(self.root, output_filename)

            # Create exp folder into the root folder
            os.makedirs(output_folder)

            sm_filename = output_filename + '_similarity_matrix.pkl.tz'
            try:
                pkl.dump(similarity_matrix, gzip.open(
                    os.path.join(output_folder, sm_filename), 'w+'))
                logging.info("Dumped similarity matrix: %s",
                             os.path.join(output_folder, sm_filename))
            except OverflowError:
                logging.error("Cannot dump similarity matrix")

        logging.info("Start define_clusts function ...")
        clusters = define_clusts(
            similarity_matrix, threshold=self.threshold, method=self.cluster)
        n_clones = np.max(clusters) - np.min(clusters) + 1
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

            cl_filename = output_filename + '_clusters.pkl.tz'
            pkl.dump([clusters, self.threshold], gzip.open(
                os.path.join(output_folder, cl_filename), 'w+'))
            logging.info("Dumped clusters and threshold: %s",
                         os.path.join(output_folder, cl_filename))
            self.output_folder_ = output_folder

        clone_dict = {k.id: v for k, v in zip(records, clusters)}
        self.clone_dict_ = clone_dict

        return self


def compute_similarity_matrix(db_iter, sparse_mode=True, igsimilarity=None):
    """Compute the similarity matrix from a database iterator.

    Parameters
    ----------
    db_iter : generator
        Records loaded by the script `ici_run.py`.
    sparse_mode : bool, optional, default `True`
        Return a sparse similarity matrix.
    sim_func_args : dict, optional
        Optional parameters for the similarity function.

    Returns
    -------
    similarity_matrix : scipy.sparse array or numpy.ndarray
        Similarity matrix between records in db_iter.

    Notes
    -----
    This function uses sparse matrices and an inverse index to efficiently
    compute the similarity matrix between a potentially long list of records.
    In ancient times, this function was simply a call to
    ``parallel_distance.distance_matrix_parallel(db_iter, d_func, sparse_mode)``
    However, this method is really inefficient, so the matrix is computed
    between chosen couples of values.
    """
    igs = list(db_iter)
    n = len(igs)

    # set_defaults_sim_func(sim_func_args, igs)
    # logging.info("Similarity function parameters: %s", sim_func_args)
    # similarity_function = partial(sim_function, **sim_func_args)

    # logging.info("Start similar_elements function ...")
    # rows, cols = similar_elements(dd, igs, n, similarity_function)

    logging.info("Start parallel_sim_matrix function ...")
    data, rows, cols = sm_sparse(
        np.array(igs), igsimilarity.pairwise, igsimilarity.tol)

    sparse_mat = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    similarity_matrix = sparse_mat  # connected components works well
    if not sparse_mode:
        similarity_matrix = similarity_matrix.toarray()

    return similarity_matrix


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
