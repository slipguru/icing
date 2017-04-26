import joblib as jl
import numpy as np
from sklearn.base import BaseEstimator


class StringKernel(BaseEstimator):
    """Utility class for string kernel."""

    def __init__(self, kn=1, lamda=.5,
                 check_min_length=0, hard_matching=1, normalize=True):
        self.kn = kn
        self.lamda = lamda
        self.check_min_length = check_min_length
        self.hard_matching = hard_matching
        self.normalize = normalize

    def pairwise(self, x, y):
        len_x, len_y = len(x), len(y)
        if len_x < self.kn or len_y < self.kn:
            # do not compute kernel
            return x == y

        x_dim = len_x + 1
        y_dim = len_y + 1
        # Allocate and initialise Kd
        Kd = [np.ones(x_dim * y_dim), np.zeros(x_dim * y_dim)]
        # Kd now contains two matrices, that are n+1 x m+1 (empty string included)
        # Kd[0] is composed by 1s (follows the definition of K_0)
        # Kd[1] is composed by 0s -> it starts to be filled

        #  start with i = kn = 1, 2, 3 ...
        _lambda = self.lamda
        for i in range(1, self.kn):
            #   Set the Kd to zero for those lengths of s and t
            #   where s (or t) has exactly length i-1 and t (or s)
            #   has length >= i-1. L-shaped upside down matrix

            for j in range(i - 1, len_x):
                Kd[i % 2][j * y_dim + i - 1] = 0

            for j in range(i - 1, len_y):
                Kd[i % 2][(i - 1) * y_dim + j] = 0

            for j in range(i, len_x):
                # Kdd maintains the contribution of the left and diagonal terms
                # that is, ONLY the contribution of the left (not influenced by the
                # upper terms) and the eventual contibution of lambda^2 in case the
                # chars are the same
                Kdd = 0

                for k in range(i, len_y):
                    if x[j - 1] != y[k - 1]:
                        # ((.))-1 is because indices start with 0 (not with 1)
                        Kdd *= _lambda
                    else:
                        Kdd = _lambda * (Kdd + (_lambda * Kd[(i + 1) % 2][(j - 1) * y_dim + k - 1]))
                    Kd[i % 2][j*y_dim+k] = _lambda * Kd[i % 2][(j - 1) * y_dim + k] + Kdd

        # Calculate K
        sum_ = 0
        for i in range(self.kn - 1, len_x):
            for j in range(self.kn - 1, len_y):
                # hard matching
                if self.hard_matching:
                    if x[i] == y[j]:
                        sum_ += _lambda * _lambda * Kd[(self.kn - 1) % 2][i*y_dim + j]
                else:
                    # soft matching, regulated from models.h, amminoacidic model
                    sum_ += _lambda * _lambda * \
                          self.aa_model[(ord(x[i])-65)*26 + ord(y[j])-65] * \
                          Kd[(self.kn - 1) % 2][i*y_dim + j];
        return sum_


    def compute_norms(self, records):
        self.norms = [self.pairwise(x, x) for x in records]

    def fit(self, strings):
        """String kernel of a single subsequence length."""
        # Get values for normalization, it is computed for elements in diagonal
        n_samples = len(strings)
        if self.normalize:
            self.compute_norms(strings)

        # Compute kernel using dynamic programming
        _kernel = np.empty(n_samples * n_samples)
        for i in range(n_samples):
            offset = int(bool(self.normalize))
            if self.normalize:
                _kernel[i * n_samples + i] = 1

            for j in range(i + offset, n_samples):
                _kernel[i*n_samples+j] = self.pairwise(strings[i], strings[j])
                if self.normalize:
                    _kernel[i*n_samples+j] /= np.sqrt(self.norms[i] * self.norms[j]);
                _kernel[j*n_samples+i] = _kernel[i*n_samples+j];

        self.kernel_ = _kernel.reshape(n_samples, n_samples)

        return self


def _worker_string_kernel(estimator, strings, kn):
    single_kernel = StringKernel(
        kn=kn, lamda=estimator.lamda,
        check_min_length=estimator.check_min_length,
        hard_matching=estimator.hard_matching,
        normalize=False).fit(strings).kernel_
    return single_kernel


class SumStringKernel(BaseEstimator):
    """Utility class for string kernel."""

    def __init__(self, min_kn=1, max_kn=2, lamda=.5, n_jobs=-1,
                 check_min_length=0, hard_matching=True, normalize=True):
        self.min_kn = min_kn
        self.max_kn = max_kn
        self.lamda = lamda
        self.check_min_length = check_min_length
        self.hard_matching = hard_matching
        self.normalize = normalize
        self.n_jobs = n_jobs

    def pairwise(self, x1, x2):
        self.fit((x1, x2))
        return self.kernel_[0, 1]

    def fit(self, strings):
        """Kernel is built as the sum of string kernels of different length."""
        # Get values for normalization, it is computed for elements in diagonal
        n_samples = len(strings)

        # special case
        if self.n_jobs == 1:
            kernel = np.zeros((n_samples, n_samples))
            for kn in range(self.min_kn, self.max_kn + 1):
                kernel += StringKernel(
                    kn=kn, lamda=self.lamda,
                    check_min_length=self.check_min_length,
                    hard_matching=self.hard_matching,
                    normalize=False).fit(strings).kernel_
        else:
            kernel = jl.Parallel(n_jobs=self.n_jobs)(jl.delayed(_worker_string_kernel)(
                self, strings, kn) for kn in range(self.min_kn, self.max_kn + 1))
            kernel = reduce(lambda x, y: sum((x, y)), kernel)

        if self.normalize:
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    kernel[i, j] /= np.sqrt(kernel[i, i] * kernel[j, j])
                    kernel[j, i] = kernel[i, j]
            kernel.flat[::n_samples + 1] = 1

        self.kernel_ = kernel

        return self
