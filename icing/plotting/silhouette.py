#!/usr/bin/env python
"""Plotting functions for silhouette analysis.

Author: Federico Tomasi
Copyright (c) 2016, Federico Tomasi.
Licensed under the FreeBSD license (see LICENSE.txt).
"""
from __future__ import print_function, division

import logging
import matplotlib; matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import scipy
import seaborn as sns; sns.set_context('notebook')
import sys; sys.setrecursionlimit(10000)

from scipy.cluster.hierarchy import linkage, fcluster
# from sklearn.cluster import SpectralClustering
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_samples  # , silhouette_score

from icing.externals import SpectralClustering
from icing.externals import Tango
from icing.utils import extra


def plot_clusters_silhouette(X, cluster_labels, n_clusters, root='',
                             file_format='pdf'):
    """Plot the silhouette score for each cluster, given the distance matrix X.

    Parameters
    ----------
    X : array_like, shape [n_samples_a, n_samples_a]
        Distance matrix.
    cluster_labels : array_like
        List of integers which represents the cluster of the corresponding
        point in X. The size must be the same has a dimension of X.
    n_clusters : int
        The number of clusters.
    root : str, optional
        The root path for the output creation
    file_format : ('pdf', 'png')
        Choose the extension for output images.
    """
    # Create a subplot with 1 row and 2 columns
    fig, (ax1) = plt.subplots(1, 1)
    fig.set_size_inches(20, 15)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    # ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels,
                                                  metric="precomputed")
    silhouette_avg = np.mean(sample_silhouette_values)
    logging.info("Average silhouette_score: %.4f", silhouette_avg)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        # ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("silhouette coefficient values")
    ax1.set_ylabel("cluster label")

    # The vertical line for average silhoutte score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.suptitle(("Silhouette analysis (n_clusters {}, avg score {:.4f}, "
                  "tot Igs {}".format(n_clusters, silhouette_avg, X.shape[0])),
                 fontsize=14, fontweight='bold')
    filename = os.path.join(root, 'silhouette_analysis_{}.{}'
                                  .format(extra.get_time(), file_format))
    fig.savefig(filename)
    logging.info('Figured saved %s', filename)


def best_intersection(id_list, cluster_dict):
    """Compute jaccard index between id_list and each list in dict."""
    set1 = set(id_list)
    best_score = (0., 0., 0.)  # res, numerator, denominator
    best_set = ()
    best_key = -1
    for k in cluster_dict:
        set2 = set(cluster_dict[k])
        numerator = len(set1 & set2)
        denominator = len(set1 | set2)
        score = numerator / denominator
        if score > best_score[0] or best_key == -1:
            best_score = (score, numerator, denominator)
            best_set = set2.copy()
            best_key = k
    # print(set1, "and best", best_set, best_score)
    if best_key != -1:
        del cluster_dict[best_key]
    return best_score, cluster_dict, best_set


def save_results_clusters(filename, sample_names, cluster_labels):
    with open(filename, 'w') as f:
        for a, b in zip(sample_names, cluster_labels):
            f.write("{}, {}\n".format(a, b))


def single_silhouette_dendrogram(dist_matrix, Z, threshold, mode='clusters',
                                 method='single', sample_names=None):
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
    nclusts = np.unique(cluster_labels).shape[0]

    save_results_clusters("res_{}_{:03d}_clust.csv".format(method, nclusts),
                          sample_names, cluster_labels)

    try:
        silhouette_list = silhouette_samples(dist_matrix, cluster_labels,
                                             metric="precomputed")
        silhouette_avg = np.mean(silhouette_list)
        x = max(cluster_labels) if mode == 'clusters' else threshold
    except ValueError as e:
        if max(cluster_labels) == 1:
            x = 1 if mode == 'clusters' else threshold
            silhouette_avg = 0
        else:
            raise(e)

    return x, silhouette_avg


def multi_cut_dendrogram(dist_matrix, Z, threshold_arr, n, mode='clusters',
                         method='single', n_jobs=-1, sample_names=None):
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
    def _internal(dist_matrix, Z, threshold_arr, idx, n_jobs, arr_length,
                  queue_x, queue_y, mode='clusters', method='single',
                  sample_names=None):
        for i in range(idx, arr_length, n_jobs):
            queue_x[i], queue_y[i] = single_silhouette_dendrogram(
                dist_matrix, Z, threshold_arr[i], mode, method, sample_names)

    if n_jobs == -1:
        n_jobs = min(mp.cpu_count(), n)
    queue_x, queue_y = mp.Array('d', [0.] * n), mp.Array('d', [0.] * n)
    ps = []
    try:
        for idx in range(n_jobs):
            p = mp.Process(target=_internal,
                           args=(dist_matrix, Z, threshold_arr, idx, n_jobs, n,
                                 queue_x, queue_y, mode, method, sample_names))
            p.start()
            ps.append(p)

        for p in ps:
            p.join()
    except (KeyboardInterrupt, SystemExit):
        extra.term_processes(ps, 'Exit signal received\n')
    except BaseException as e:
        extra.term_processes(ps, 'ERROR: %s\n' % e)
    return queue_x, queue_y


def plot_average_silhouette_dendrogram(
        X, method_list=None, mode='clusters', n=20, min_threshold=0.02,
        max_threshold=0.8, verbose=True, interactive_mode=False,
        file_format='pdf', xticks=None, xlim=None, figsize=None, n_jobs=-1,
        sample_names=None):
    """Plot average silhouette for each tree cutting.

    A linkage matrix for each method in method_list is used.

    Parameters
    ----------
    X : array-like
        Symmetric 2-dimensional distance matrix.
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
    file_format : ('pdf', 'png')
        Choose the extension for output images.

    Returns
    -------
    filename : str
        The output filename.
    """
    if method_list is None:
        method_list = ('single', 'complete', 'average', 'weighted',
                       'centroid', 'median', 'ward')

    plt.close()
    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    fig, ax = (plt.gcf(), plt.gca())  # if plt.get_fignums() else plt.subplots()
    fig.suptitle("Average silhouette for each tree cutting")
    # print_utils.ax_set_defaults(ax)

    # convert distance matrix into a condensed one
    dist_arr = scipy.spatial.distance.squareform(X)
    for method in method_list:
        if verbose:
            print("Compute linkage with method = {}...".format(method))
        Z = linkage(dist_arr, method=method, metric='euclidean')
        if method == 'ward':
            threshold_arr = np.linspace(np.percentile(Z[:, 2], 70), max(Z[:, 2]), n)
        else:
            threshold_arr = np.linspace(min_threshold, max_threshold, n)
            max_i = max(Z[:, 2]) if method != 'ward' else np.percentile(Z[:, 2], 99.5)
            threshold_arr *= max_i

        x, y = multi_cut_dendrogram(X, Z, threshold_arr, n, mode, method, n_jobs,
                                    sample_names=sample_names)
        ax.plot(x, y, Tango.nextDark(), marker='o', ms=3, ls='-', label=method)

    # fig.tight_layout()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    leg = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # leg = ax.legend(loc='lower right')
    leg.get_frame().set_linewidth(0.0)
    ax.set_xlabel(mode[0].upper() + mode[1:])
    ax.set_ylabel("Silhouette")
    if xticks is not None:
        plt.xticks(xticks)
    if xlim is not None:
        plt.xlim(xlim)
    plt.margins(.2)
    plt.subplots_adjust(bottom=0.15)
    if interactive_mode:
        plt.show()

    path = "results"
    extra.mkpath(path)
    filename = os.path.join(path, "result_silhouette_hierarchical_{}.{}"
                                  .format(extra.get_time(), file_format))
    fig.savefig(filename, bbox_extra_artists=(leg,), bbox_inches='tight')
    # fig.savefig(filename, dpi=300, format='png')
    return filename


def multi_cut_spectral(cluster_list, affinity_matrix, dist_matrix, n_jobs=-1,
                       sample_names=None):
    """Perform a spectral clustering with variable cluster sizes.

    Parameters
    ----------
    cluster_list : array-like
        Contains the list of the number of clusters to use at each step.
    affinity_matrix : array-like
        Precomputed affinity matrix.
    dist_matrix : array-like
        Precomputed distance matrix between points.

    Returns
    -------
    queue_y : array-like
        Array to be visualised on the y-axis. Contains the list of average
        silhouette for each number of clusters present in cluster_list.

    """
    def _internal(cluster_list, affinity_matrix, dist_matrix,
                  idx, n_jobs, n, queue_y):
        for i in range(idx, n, n_jobs):
            sp = SpectralClustering(n_clusters=cluster_list[i],
                                    affinity='precomputed',
                                    norm_laplacian=True,
                                    n_init=1000)
            sp.fit(affinity_matrix)

            save_results_clusters("res_spectral_{:03d}_clust.csv"
                                  .format(cluster_list[i]),
                                  sample_names, sp.labels_)

            silhouette_list = silhouette_samples(dist_matrix, sp.labels_,
                                                 metric="precomputed")
            queue_y[i] = np.mean(silhouette_list)

    n = len(cluster_list)
    if n_jobs == -1:
        n_jobs = min(mp.cpu_count(), n)
    queue_y = mp.Array('d', [0.] * n)
    ps = []
    try:
        for idx in range(n_jobs):
            p = mp.Process(target=_internal,
                           args=(cluster_list, affinity_matrix, dist_matrix,
                                 idx, n_jobs, n, queue_y))
            p.start()
            ps.append(p)

        for p in ps:
            p.join()
    except (KeyboardInterrupt, SystemExit):
        extra.term_processes(ps, 'Exit signal received\n')
    except BaseException as e:
        extra.term_processes(ps, 'ERROR: %s\n' % e)
    return queue_y


def plot_average_silhouette_spectral(
        X, n=30, min_clust=10, max_clust=None, verbose=True,
        interactive_mode=False, file_format='pdf', n_jobs=-1,
        sample_names=None, affinity_delta=.2, is_affinity=False):
    """Plot average silhouette for some clusters, using an affinity matrix.

    Parameters
    ----------
    X : array-like
        Symmetric 2-dimensional distance matrix.
    verbose : boolean, optional
        How many output messages visualise.
    interactive_mode : boolean, optional
        True: final plot will be visualised and saved.
        False: final plot will be only saved.
    file_format : ('pdf', 'png')
        Choose the extension for output images.

    Returns
    -------
    filename : str
        The output filename.
    """
    X = extra.ensure_symmetry(X)
    A = extra.distance_to_affinity_matrix(X, delta=affinity_delta) if not is_affinity else X

    plt.close()
    fig, ax = (plt.gcf(), plt.gca())
    fig.suptitle("Average silhouette for each number of clusters")

    if max_clust is None:
        max_clust = X.shape[0]
    cluster_list = np.unique(map(int, np.linspace(min_clust, max_clust, n)))
    y = multi_cut_spectral(cluster_list, A, X, n_jobs=n_jobs,
                           sample_names=sample_names)
    ax.plot(cluster_list, y, Tango.next(), marker='o', linestyle='-', label='')

    # leg = ax.legend(loc='lower right')
    # leg.get_frame().set_linewidth(0.0)
    ax.set_xlabel("Clusters")
    ax.set_ylabel("Silhouette")
    fig.tight_layout()
    if interactive_mode:
        plt.show()
    path = "results"
    extra.mkpath(path)
    filename = os.path.join(path, "result_silhouette_spectral_{}.{}"
                                  .format(extra.get_time(), file_format))
    plt.savefig(filename)
    plt.close()

    # plot eigenvalues
    from adenine.core import plotting
    plotting.eigs(
        '', A,
        filename=os.path.join(path, "eigs_spectral_{}.{}"
                                    .format(extra.get_time(), file_format)),
        n_components=50,
        normalised=True,
        rw=True
    )

    return filename

def multi_cut_ap(preferences, affinity_matrix, dist_matrix, n_jobs=-1,
                 sample_names=None):
    """Perform a AP clustering with variable cluster sizes.

    Parameters
    ----------
    cluster_list : array-like
        Contains the list of the number of clusters to use at each step.
    affinity_matrix : array-like
        Precomputed affinity matrix.
    dist_matrix : array-like
        Precomputed distance matrix between points.

    Returns
    -------
    queue_y : array-like
        Array to be visualised on the y-axis. Contains the list of average
        silhouette for each number of clusters present in cluster_list.

    """
    def _internal(preferences, affinity_matrix, dist_matrix,
                  idx, n_jobs, n, queue_y):
        for i in range(idx, n, n_jobs):
            ap = AffinityPropagation(preference=preferences[i],
                                     affinity='precomputed',
                                     max_iter=500)
            ap.fit(affinity_matrix)

            cluster_labels = ap.labels_.copy()
            nclusts = np.unique(cluster_labels).shape[0]
            save_results_clusters("res_ap_{:03d}_clust.csv"
                                  .format(nclusts),
                                  sample_names, ap.labels_)

            if nclusts > 1:
                try:
                    silhouette_list = silhouette_samples(dist_matrix, ap.labels_,
                                                     metric="precomputed")
                    queue_y[i] = np.mean(silhouette_list)
                except BaseException:
                    print(dist_matrix.shape, ap.labels_.shape)

    n = len(preferences)
    if n_jobs == -1:
        n_jobs = min(mp.cpu_count(), n)
    queue_y = mp.Array('d', [0.] * n)
    ps = []
    try:
        for idx in range(n_jobs):
            p = mp.Process(target=_internal,
                           args=(preferences, affinity_matrix, dist_matrix,
                                 idx, n_jobs, n, queue_y))
            p.start()
            ps.append(p)

        for p in ps:
            p.join()
    except (KeyboardInterrupt, SystemExit):
        extra.term_processes(ps, 'Exit signal received\n')
    except BaseException as e:
        extra.term_processes(ps, 'ERROR: %s\n' % e)
    return queue_y


def plot_average_silhouette_ap(
        X, n=30, verbose=True,
        interactive_mode=False, file_format='pdf', n_jobs=-1,
        sample_names=None, affinity_delta=.2, is_affinity=False):
    """Plot average silhouette for some clusters, using an affinity matrix.

    Parameters
    ----------
    X : array-like
        Symmetric 2-dimensional distance matrix.
    verbose : boolean, optional
        How many output messages visualise.
    interactive_mode : boolean, optional
        True: final plot will be visualised and saved.
        False: final plot will be only saved.
    file_format : ('pdf', 'png')
        Choose the extension for output images.

    Returns
    -------
    filename : str
        The output filename.
    """
    X = extra.ensure_symmetry(X)
    A = extra.distance_to_affinity_matrix(X, delta=affinity_delta) if not is_affinity else X

    plt.close()
    fig, ax = (plt.gcf(), plt.gca())
    fig.suptitle("Average silhouette for each number of clusters")

    # dampings = np.linspace(.5, 1, n+1)[:-1]
    # from sklearn.metrics.pairwise import pairwise_distances
    # S = -pairwise_distances(X, metric='precomputed', squared=True)
    preferences = np.append(np.linspace(np.min(A), np.median(A), n - 1),
                            np.median(A))
    y = multi_cut_ap(preferences, A, X, n_jobs=n_jobs,
                     sample_names=sample_names)
    ax.plot(preferences, y, Tango.next(), marker='o', linestyle='-', label='')

    # leg = ax.legend(loc='lower right')
    # leg.get_frame().set_linewidth(0.0)
    ax.set_xlabel("Clusters")
    ax.set_ylabel("Silhouette")
    fig.tight_layout()
    if interactive_mode:
        plt.show()
    plt.close()

    return None


if __name__ == '__main__':
    """Example of usage.
    silhouette.plot_average_silhouette_dendrogram(
        X, min_threshold=.7, max_threshold=1.1, n=200, xticks=range(0, 50, 4),
        xlim=[0, 50], figsize=(20, 8),
        method_list=('median', 'ward', 'complete'))
    silhouette.plot_average_silhouette_spectral(X, min_clust=2, max_clust=10, n=10)
    """
