#!/usr/bin/env python

import numpy as np
import scipy
import matplotlib.pyplot as plt
import multiprocessing as mp
import sys
import pandas as pd
import seaborn  #TODO remove dependence from seaborn

from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_samples, silhouette_score

from entropy.plotting import Tango, print_utils

from ..utils import utils
from ..utils.utils import _terminate

# fn = '/home/fede/src/adenine/adenine/examples/TM_matrix.csv'
# X = pd.io.parsers.read_csv(fn, header=0, index_col=0).as_matrix() # TM distance between Igs
# if not (X.T == X).all():
#     X = (X.T + X) / 2.
#
# # convert distance into an affinity matrix
# delta = .2
# A = np.exp(- X ** 2 / (2. * delta ** 2))
# A[A==0] += 1e-16
# l = []
# cluster_list = range(8,700,3)
# for i in cluster_list:
#     sys.stdout.write("Begin SP with {} clusters. ".format(i))
#     sys.stdout.flush()
#     # W = A - np.diag(np.diag(A))
#     # Deg = np.diag([np.sum(x) for x in W])
#     # L = Deg - W
#     # # aux = np.linalg.inv(np.diag([np.sqrt(np.sum(x)) for x in W]))
#     # # L = np.eye(L.shape[0]) - (np.dot(np.dot(aux,W),aux)) # normalised L
#     # print L
#     # L[np.abs(L) < 1e-17] = 0.
#     # print L
#     # w, v = np.linalg.eig(L)
#     # w = np.array(sorted(np.abs(w)))
#     # print w[:15]
#     # print("Num clusters: {}".format(w[w < 1e-8].shape[0]))
#     # break
#
#     # D /= np.max(D)
#     # print X.shape
#     sp = SpectralClustering(n_clusters=i, affinity='precomputed')
#     sp.fit(A)
#     silh_list = silhouette_samples(X, sp.labels_, metric='precomputed')
#     # print sp.affinity_matrix_
#     # print sp.labels_
#     # f, ax = plt.subplots()
#     # heatmap = ax.pcolor(sp.affinity_matrix_, cmap=plt.cm.Blues, alpha=0.8)
#     # plt.show()
#     # silh_avg = silhouette_score(X, sp.labels_)
#     silh_avg = np.mean(silh_list)
#     sys.stdout.write("Avg silhouette: {:.4f}\n".format(silh_avg))
#     sys.stdout.flush()
#     l.append(silh_avg)


def single_spectral(lock, cluster_list, affinity_matrix, dist_matrix, arr_length, global_counter, queue_y):
    while global_counter.value < arr_length:
        with lock:
            if not global_counter.value < arr_length: break
            idx = global_counter.value
            global_counter.value += 1
        i = cluster_list[idx]
        # print("SP with {} clusters".format(i))
        sp = SpectralClustering(n_clusters=i, affinity='precomputed')
        sp.fit(affinity_matrix)

        silhouette_list = silhouette_samples(dist_matrix, sp.labels_, metric="precomputed")
        silhouette_avg = np.mean(silhouette_list)
        queue_y[idx] = silhouette_avg


def multi_spectral(cluster_list, affinity_matrix, dist_matrix):
    n = len(cluster_list)
    nproc = min(mp.cpu_count(), n)
    count = mp.Value('i', 0)
    ps = []
    queue_y = mp.Array('d', [0.]*n)
    lock = mp.Lock()
    try:
        for _ in range(nproc):
            p = mp.Process(target=single_spectral,
                           args=(lock, cluster_list, affinity_matrix, dist_matrix, n, count, queue_y))
            p.start()
            ps.append(p)

        for p in ps:
            p.join()
    except (KeyboardInterrupt, SystemExit): _terminate(ps,'Exit signal received\n')
    except Exception as e: _terminate(ps,'ERROR: %s\n' % e)
    except: _terminate(ps,'ERROR: Exiting with unknown exception\n')
    return queue_y


def plot_avg_silhouette(X=None, fn=None, verbose=True, interactive_mode=False):
    """...

    Parameters
    ----------
    X : array-like, optional
        Symmetric distance matrix.
    fn : str, optional
        Alternatively, the filename for the distance matrix can be specified.

    Returns
    -------
    None
    """
    if X is None:
        try:
            X = pd.io.parsers.read_csv(fn, header=0, index_col=0).as_matrix()
        except:
            sys.stderr.write("ERROR: {} cannot be read\n".format(dm_fn))
            sys.exit()

    X = X.toarray()
    # check symmetry
    if not (X.T == X).all():
        X = (X.T + X) / 2.

    # convert distances into (Gaussian) affinities
    delta = .2
    A = np.exp(- X ** 2 / (2. * delta ** 2))
    # A[A==0] += 1e-16

    fig, ax = (plt.gcf(), plt.gca())
    fig.suptitle("Average silhouette for each number of clusters")
    # print_utils.ax_set_defaults(ax)

    # convert distance matrix into a condensed one
    # dist_arr = scipy.spatial.distance.squareform(dist_matrix)

    # cluster_list = range(10,X.shape[0],10)
    cluster_list = map(int, np.linspace(10, X.shape[0], 30))
    y = multi_spectral(cluster_list, A, X)
    ax.plot(cluster_list, y, Tango.next(), marker='o', linestyle='-', label='')

    # leg = ax.legend(loc='lower right')
    # leg.get_frame().set_linewidth(0.0)
    ax.set_xlabel("Clusters")
    ax.set_ylabel("Silhouette")
    fig.tight_layout()
    if interactive_mode:
        plt.show()
    path = "results/"
    utils.mkpath(path)
    plt.savefig(path+"res_silh_SC_{}.png".format(utils.get_time()))

if __name__ == '__main__':
    plot_avg_silhouette('/home/fede/Dropbox/projects/plotting/examples/TM_matrix.csv')
