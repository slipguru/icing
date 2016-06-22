#!/usr/bin/env python
"""Documentation for silhouette_hierarchical.py."""
import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
import multiprocessing as mp
import sys; sys.setrecursionlimit(10000)
import seaborn

from sklearn.metrics import silhouette_samples  # , silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster

from ..externals import Tango
from ..utils import utils
from ..utils.utils import _terminate


def compute_x_y(dist_matrix, Z, threshold, mode='clusters'):
    """Compute the average silhouette at a given threshold.

    Parameters
    ----------
    dist_matrix : array-like
        Precomputed distance matrix between points.
    Z : array-like
        Linkage matrix, results of scipy.cluster.hierarchy.linkage.
    threshold : float
        Specifies where to cut the dendrogram.
    mode : ('clusters', 'thresholds'), optional
        Choose what to visualise on the x-axis.

    Returns
    -------
    x : float
        Based on mode, it can contains the number of clusters or threshold.
    silhouette_avg : float
        The average silhouette.
    """
    cluster_labels = fcluster(Z, threshold, 'distance')
    silhouette_list = silhouette_samples(dist_matrix, cluster_labels,
                                         metric="precomputed")
    silhouette_avg = np.mean(silhouette_list)
    x = max(cluster_labels) if mode == 'clusters' else threshold
    return x, silhouette_avg


def multi_cut_dendrogram(dist_matrix, Z, threshold_arr, n, mode='clusters'):
    """Cut a dendrogram at some heights.

    Parameters
    ----------
    dist_matrix : array-like
        Precomputed distance matrix between points.
    Z : array-like
        Linkage matrix, results of scipy.cluster.hierarchy.linkage.
    threshold_arr : array-like
        One-dimensional array which contains the thresholds where to cut the
        dendrogram.
    n : int
        Length of threshold_arr
    mode : ('clusters', 'thresholds'), optional
        Choose what to visualise on the x-axis.

    Returns
    -------
    queue_{x, y} : array-like
        The results to be visualised on a plot.

    """
    def _internal(dist_matrix, Z, threshold_arr, idx, nprocs, arr_length,
                  queue_x, queue_y, mode='clusters'):
        for i in range(idx, arr_length, nprocs):
            queue_x[i], queue_y[i] = compute_x_y(dist_matrix, Z,
                                                 threshold_arr[i], mode)

    nprocs = min(mp.cpu_count(), n)
    queue_x, queue_y = mp.Array('d', [0.]*n), mp.Array('d', [0.]*n)
    ps = []
    try:
        for idx in range(nprocs):
            p = mp.Process(target=_internal,
                           args=(dist_matrix, Z, threshold_arr, idx, nprocs, n,
                                 queue_x, queue_y, mode))
            p.start()
            ps.append(p)

        for p in ps:
            p.join()
    except (KeyboardInterrupt, SystemExit):
        _terminate(ps, 'Exit signal received\n')
    except Exception as e:
        _terminate(ps, 'ERROR: %s\n' % e)
    except:
        _terminate(ps, 'ERROR: Exiting with unknown exception\n')
    return queue_x, queue_y


def plot_avg_silhouette(X=None, config_file='config.py', method_list=None,
                        mode='clusters', n=20, verbose=True,
                        interactive_mode=False):
    """Plot average silhouette for each tree cutting.

    A linkage matrix for each method in method_list is used.

    Parameters
    ----------
    X : array-like, dimensions = 2
        Distance matrix.
    config_file : str, optional
        If X is None, use config_file to compute X with
            `ignet.core.cloning.distance_matrix`
    method_list : array-like, optional
        String array which contains a list of methods for computing the
        linkage matrix. If None, all the avalable methods will be used.
    mode : ('clusters', 'threshold')
        Choose what to visualise on x-axis.
    n : int, optional
        Choose at how many heights dendrogram must be cut.
    verbose : boolean, optional
        How many output messages visualise.
    interactive_mode : boolean, optional
        True: final plot will be visualised and saved.
        False: final plot will be only saved.

    Returns
    -------
    filename : str
        The output filename.
    """
    if X is None:
        from ignet.core.cloning import distance_matrix
        X = 1. - distance_matrix(config_file).toarray()
    if method_list is None:
        method_list = ('single', 'complete', 'average', 'weighted',
                       'centroid', 'median', 'ward')

    threshold_arr = np.linspace(0.02, 0.8, n)

    plt.close()
    fig, ax = (plt.gcf(), plt.gca())  # if plt.get_fignums() else plt.subplots()
    fig.suptitle("Average silhouette for each tree cutting")
    # print_utils.ax_set_defaults(ax)

    # convert distance matrix into a condensed one
    dist_arr = scipy.spatial.distance.squareform(X)
    for method in method_list:
        if verbose:
            print("Compute linkage with method = {}...".format(method))
        Z = linkage(dist_arr, method=method, metric='euclidean')
        # dendrogram(Z)
        # plt.close()
        if verbose:
            print("Start cutting ...")
        max_i = max(Z[:, 2])
        # if use_joblib:
        #     x, y = zip(*jl.Parallel(n_jobs=mp.cpu_count())
        #                (jl.delayed(compute_x_y)(X, Z, i, mode)
        #                for i in threshold_arr*max_i))
        # else:
        x, y = multi_cut_dendrogram(X, Z, threshold_arr*max_i, n, mode)
        ax.plot(x, y, Tango.nextDark(), marker='o', linestyle='-', label=method)

    leg = ax.legend(loc='lower right')
    leg.get_frame().set_linewidth(0.0)
    ax.set_xlabel(mode[0].upper()+mode[1:])
    ax.set_ylabel("Silhouette")
    fig.tight_layout()
    if interactive_mode:
        plt.show()
    path = "results"
    utils.mkpath(path)
    filename = os.path.join(path, "res_silh_HC_{}.png".format(utils.get_time()))
    plt.savefig(filename)
    return filename


if __name__ == '__main__':
    plot_avg_silhouette()
