#!/usr/bin/env python

import numpy as np
import scipy
import matplotlib.pyplot as plt
import multiprocessing as mp
import sys
sys.setrecursionlimit(10000)
import seaborn  #TODO remove dependence from seaborn

from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

from entropy.plotting import Tango, print_utils

from ..utils import utils
from ..utils.utils import _terminate

def compute_x_y(dist_matrix, Z, threshold, mode='clusters'):
    cluster_labels = fcluster(Z, threshold, 'distance')
    silhouette_list = silhouette_samples(dist_matrix, cluster_labels, metric="precomputed")
    silhouette_avg = np.mean(silhouette_list)
    x = max(cluster_labels) if mode == 'clusters' else threshold
    return x, silhouette_avg

def single_fcluster(lock, dist_matrix, Z, threshold_arr, arr_length, global_counter, queue_x, queue_y, mode='clusters'):
    while global_counter.value < arr_length:
        with lock:
            if not global_counter.value < arr_length: break
            idx_threshold = global_counter.value
            global_counter.value += 1

        queue_x[idx_threshold], queue_y[idx_threshold] = compute_x_y(dist_matrix, Z, threshold_arr[idx_threshold], mode)


def multi_cut_dendrogram(dist_matrix, Z, threshold_arr, n=None, mode='clusters'):
    if not n:
        n = len(threshold_arr)
    nproc = min(mp.cpu_count(), n)
    global_counter = mp.Value('i', 0)
    proc_list = []
    queue_x, queue_y = mp.Array('d', [0.]*n), mp.Array('d', [0.]*n)
    lock = mp.Lock()
    try:
        for _ in range(nproc):
            p = mp.Process(target=single_fcluster,
                           args=(lock, dist_matrix, Z, threshold_arr, n, global_counter, queue_x, queue_y, mode))
            p.start()
            proc_list.append(p)

        for p in proc_list:
            p.join()
    except (KeyboardInterrupt, SystemExit): _terminate(proc_list,'Exit signal received\n')
    except Exception as e: _terminate(proc_list,'ERROR: %s\n' % e)
    except: _terminate(proc_list,'ERROR: Exiting with unknown exception\n')
    return queue_x, queue_y


def plot_avg_silhouette(X=None, config_file='config.py', method_list=None,
                        mode='clusters', n=20, verbose=True,
                        interactive_mode=False):
    """Plot average silhouette for each tree cutting using a linkage matrix
    created according to each method in method_list.

    Parameters
    ----------
    X : array-like, dimensions = 2
        Distance matrix.
    ...
    mode : str, optional, values in ['clusters', 'threshold']
        Choose what to visualise on x-axis.

    Returns
    -------
    None
    """
    if X is None:
        from .. import define_clones_network
        X = define_clones_network.main(config_file)
    if method_list is None:
        # method_list = ['single','complete','average','weighted','centroid','median','ward']
        method_list = ['complete','average','weighted','centroid','median','ward']

    threshold_arr = np.linspace(0.02, 0.8, n)

    plt.close()
    fig, ax = (plt.gcf(), plt.gca())  # if plt.get_fignums() else plt.subplots()
    fig.suptitle("Average silhouette for each tree cutting")
    # print_utils.ax_set_defaults(ax)

    # convert distance matrix into a condensed one
    dist_arr = scipy.spatial.distance.squareform(X)
    for method in method_list:
        if verbose: print("Compute linkage with method = {}...".format(method))
        Z = linkage(dist_arr, method=method, metric='euclidean')
        dendrogram(Z)
        plt.savefig('ciao')
        plt.close()
        if verbose: print("Start cutting ...")
        max_i = max(Z[:, 2])
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
    fn = "{}/res_silh_HC_{}.png".format(path, utils.get_time())
    plt.savefig(fn)
    return fn


def plot_avg_silhouette_joblib(X=None, config_file='config.py', method_list=None, mode='clusters', n=20, verbose=True, interactive_mode=False):
    """Plot average silhouette for each tree cutting using a linkage matrix
    created according to each method in method_list.

    Parameters
    ----------
    ...
    mode : str, optional, values in ['clusters', 'threshold']
        Choose what to visualise on x-axis.
    n : int, optional
        Choose at how many heights dendrogram must be cut

    Returns
    -------
    None
    """
    import joblib as jl

    if X is None:
        from .. import define_clones_network
        X = define_clones_network.main(config_file)
    if method_list is None:
        method_list = ['single','complete','average','weighted','centroid','median','ward']

    threshold_arr = np.linspace(0.02, 0.8, n)

    plt.close()
    fig, ax = (plt.gcf(), plt.gca())
    fig.suptitle("Average silhouette for each tree cutting")
    # print_utils.ax_set_defaults(ax)

    # convert distance matrix into a condensed one
    dist_arr = scipy.spatial.distance.squareform(X)
    for method in method_list:
        if verbose: print("Compute linkage with method = {}...".format(method))
        Z = linkage(dist_arr, method=method, metric='euclidean')

        if verbose: print("Start cutting ...")
        max_i = max(Z[:,2])
        x, y = zip(*jl.Parallel(n_jobs=mp.cpu_count())
                   (jl.delayed(compute_x_y)(X, Z, i, mode)
                   for i in threshold_arr*max_i))
        ax.plot(x, y, Tango.next(), marker='o', linestyle='-', label=method)

    leg = ax.legend(loc='lower right')
    leg.get_frame().set_linewidth(0.0)
    ax.set_xlabel(mode[0].upper()+mode[1:])
    ax.set_ylabel("Silhouette")
    fig.tight_layout()
    if interactive_mode: plt.show()
    path = "results/"
    utils.mkpath(path)
    plt.savefig(path+"res_silh_HC_jl_{}.png".format(utils.get_time()))

if __name__ == '__main__':
    plot_avg_silhouette()
