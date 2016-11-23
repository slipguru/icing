#!/usr/bin/env python
"""Utility functions for icing.

Author: Federico Tomasi
Copyright (c) 2016, Federico Tomasi.
Licensed under the FreeBSD license (see LICENSE.txt).
"""

import fcntl
import numpy as np
import os
import re
import struct
import sys
import termios
import time

from datetime import datetime

# class Counter(object):
#     """Counter which contains the lock. Atomic update"""
#
#     def __init__(self, initval=0):
#         self.val = mp.RawValue('i', initval)
#         self.lock = mp.Lock()
#
#     def increment(self, n=1):
#         with self.lock:
#             self.val.value += n
#
#     def value(self):
#         with self.lock:
#             return self.val.value
# def chunks(l, n):
#     """Yield successive n-sized chunks from l."""
#     for i in range(0, len(l), n):
#         yield l[i:i+n]


def negative_exponential(x, a, c, d):
    """Return the value of a negative exponential function."""
    return a * np.exp(-c * x) + d


def ensure_symmetry(X):
    """Ensure matrix symmetry.

    Parameters
    -----------
    X : numpy.ndarray
        Input matrix of precomputed pairwise distances.

    Returns
    -----------
    new_X : numpy.ndarray
        Symmetric distance matrix. Values are averaged.
    """
    import numpy as np
    return (X.T + X) / 2. if not np.array_equal(X, X.T) else X


def distance_to_affinity_matrix(X, delta=.2, minimum_value=0):
    """Convert the distance matrix into an affinity matrix.

    Distances are converted into Gaussian affinities.

    Parameters
    ----------
    X : array
        Distance matrix.
    delta : float, optional, default .2
        Gaussian parameter.
    minimum_value : float, optional, default 0
        Positive number. Substitute this to 0 values.

    Returns
    -------
    affinity : numpy.ndarray
        Gaussian affinity matrix.
    """
    affinity = np.exp(- X ** 2 / (2. * delta ** 2))
    if minimum_value > 0:
        affinity[affinity == 0] += minimum_value
    return affinity


def affinity_to_laplacian_matrix(A, normalised=False, tol=None,
                                 get_eigvals=False, rw=False):
    """Convert an affinity matrix into a Laplacian of the correspondent graph.

    Distances are converted into Gaussian affinities.

    Parameters
    ----------
    A : array
        Affinity matrix.
    normalised : bool, optional, default `False`
        Compute the normalised version of the Laplacian matrix.
    tol : None or float, optional
        If specified, set to 0 all values in the Laplacian less than `tol`.
    get_eigvals : bool, optional, default `False`
        Return also the eigenvalues of the Laplacian.

    Returns
    -------
    L : numpy.ndarray
        Laplacian matrix.
    """
    W = A - np.diag(np.diag(A))
    Deg = np.diag([np.sum(x) for x in W])
    L = Deg - W

    if normalised and not rw:
        # aux = np.linalg.inv(np.diag([np.sqrt(np.sum(x)) for x in W]))
        aux = np.diag(1. / np.array([np.sqrt(np.sum(x)) for x in W]))
        L = np.eye(L.shape[0]) - (np.dot(np.dot(aux, W), aux))
    elif rw:
        # normalised is ignored; it is implicit
        L = np.eye(L.shape[0]) - \
            np.dot(np.diag(1. / np.array([np.sum(x) for x in W])), W)

    if tol is not None:
        L[np.abs(L) < tol] = 0.

    if get_eigvals:
        w = np.linalg.eigvals(L)
        return L, w

    return L


def split_list(l, n):
    """Split list in n chunks."""
    for i in xrange(n):
        yield l[i * n:(i + 1) * n]


def junction_re(x, n='N', filt='[\.-/]'):
    """Convert filt char in n in x."""
    return re.sub(filt, n, str(x))


def flatten(x):
    """Flatten a list."""
    return [y for l in x for y in flatten(l)] \
        if type(x) in (list, np.ndarray) else [x]


def term_processes(ps, e=''):
    """Terminate processes in ps and exit the program."""
    sys.stderr.write(e + '\nTerminating processes ...')
    for p in ps:
        p.terminate()
        p.join()
    sys.stderr.write('... done.\n')
    sys.exit()


def _terminate(ps, e=''):
    """Terminate processes in ps and exit the program."""
    sys.stderr.write(e)
    sys.stderr.write('Terminating processes ...')
    for p in ps:
        p.terminate()
        p.join()
    sys.stderr.write('... done.\n')
    sys.exit()


def get_time_from_seconds(seconds):
    """Transform seconds into formatted time string."""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)


def get_time(ms=False):
    """Get current seconds and return them as a formatted string."""
    fmt = '%Y-%m-%d_%H.%M.%S'
    if ms:
        fmt += ',%f'
    return datetime.fromtimestamp(time.time()).strftime(fmt)


def mkpath(path):
    """If not exists, make the specified path."""
    if not os.path.exists(path):
        os.makedirs(path)


def items_iterator(dictionary):
    """Add support for python2 or 3 dictionary iterators."""
    try:
        gen = dictionary.iteritems()  # python 2
    except:
        gen = dictionary.items()  # python 3
    return gen


def combine_dicts(a, b):
    """Combine dictionaries a and b."""
    return dict(a.items() + b.items() +
                [(k, list(set(a[k] + b[k]))) for k in set(b) & set(a)])


def set_module_defaults(module, dictionary):
    """Set default variables of a module, given a dictionary.

    Used after the loading of the configuration file to set some defaults.
    """
    for k, v in items_iterator(dictionary):
        try:
            getattr(module, k)
        except AttributeError:
            setattr(module, k, v)


# progress bar
try:
    TERMINAL_COLS = struct.unpack('hh', fcntl.ioctl(
        sys.stdout, termios.TIOCGWINSZ, '1234'))[1]
except:
    TERMINAL_COLS = 50


def _bold(msg):
    return u'\033[1m{}\033[0m'.format(msg)


def _progress(current, total):
    prefix = '%d / %d' % (current, total)
    bar_start = ' ['
    bar_end = '] '

    bar_size = TERMINAL_COLS - len(prefix + bar_start + bar_end)
    try:
        amount = int(current / (total / float(bar_size)))
    except (ZeroDivisionError):
        amount = 0
    remain = bar_size - amount

    bar = '=' * amount + ' ' * remain
    return _bold(prefix) + bar_start + bar + bar_end


def progressbar(i, max_i):
    """Print a progressbar at time i, where i goes from 0 to max_i."""
    sys.stdout.flush()
    sys.stdout.write('\r' + _progress(i, max_i))
    if i >= max_i:
        sys.stdout.write('\n')
    sys.stdout.flush()
