"""Module for utilities for computing similarities."""
import logging
import numpy as np

from scipy import sparse
from sklearn.base import BaseEstimator

from icing.core.distances import StringDistance
from icing.core.parallel_distance import sm_sparse
from icing.kernel import stringkernel
from icing.models.model import model_matrix


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


class Similarity(BaseEstimator):
    _estimator_type = "similarity"

class StringKernel(Similarity):
    """Utility class for string kernel."""

    def __init__(self, min_kn=1, max_kn=2, lamda=.5,
                 check_min_length=0, hard_matching=0):
        self.min_kn = min_kn
        self.max_kn = max_kn
        self.lamda = lamda
        self.check_min_length = check_min_length
        self.hard_matching = hard_matching

    def pairwise(self, x1, x2):
        return stringkernel(
            [x1, x2], verbose=False, normalize=1, return_float=1,
            min_kn=self.min_kn, max_kn=self.max_kn, lamda=self.lamda,
            check_min_length=self.check_min_length,
            hard_matching=self.hard_matching)


class StringSimilarity(StringDistance, Similarity):
    """Utility class for string distance."""

    def __init__(self, model='ham', dist_mat=None, tol=3):
        self.model = model
        self.dist_mat = dist_mat
        self.tol = tol

        if self.dist_mat is None:
            self.dist_mat = model_matrix(model)
        self.dist_mat_max = np.max(np.max(self.dist_mat))

    def pairwise(self, x1, x2):
        return 1 - super(StringSimilarity, self).pairwise(x1, x2)


class IgSimilarity(Similarity):
    """Container for computing distance between IgRecords."""

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


def is_similarity(estimator):
    """Returns True if the given estimator encode a distance."""
    return getattr(estimator, "_estimator_type", None) == "similarity"
