#!/usr/bin/env python
"""Plotting functions for silhouette analysis.

Author: Federico Tomasi
Copyright (c) 2016, Federico Tomasi.
Licensed under the FreeBSD license (see LICENSE.txt).
"""
from __future__ import print_function

import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import multiprocessing as mp
import sys; sys.setrecursionlimit(10000)
import seaborn as sns; sns.set_context('notebook')
import logging

from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_samples  # , silhouette_score

from ..externals import Tango
from ..utils import extra


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
    logging.info("Average silhouette_score: {:.4f}".format(silhouette_avg))

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
    logging.info('Figured saved {}'.format(filename))


def single_silhouette_dendrogram(dist_matrix, Z, threshold, mode='clusters'):
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
            queue_x[i], queue_y[i] = single_silhouette_dendrogram(
                                        dist_matrix, Z, threshold_arr[i], mode)

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
        extra._terminate(ps, 'Exit signal received\n')
    except Exception as e:
        extra._terminate(ps, 'ERROR: %s\n' % e)
    except:
        extra._terminate(ps, 'ERROR: Exiting with unknown exception\n')
    return queue_x, queue_y


def plot_average_silhouette_dendrogram(X, method_list=None,
                                       mode='clusters', n=20,
                                       min_threshold=0.02,
                                       max_threshold=0.8,
                                       verbose=True,
                                       interactive_mode=False,
                                       file_format='pdf',
                                       xticks=None,
                                       xlim=None):
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


        # print(threshold_arr)
        # print(Z[:, 2])
        # if use_joblib:
        #     x, y = zip(*jl.Parallel(n_jobs=mp.cpu_count())
        #                (jl.delayed(compute_x_y)(X, Z, i, mode)
        #                for i in threshold_arr*max_i))
        x, y = multi_cut_dendrogram(X, Z, threshold_arr, n, mode)
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


def multi_cut_spectral(cluster_list, affinity_matrix, dist_matrix):
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
                  idx, nprocs, n, queue_y):
        for i in range(idx, n, nprocs):
            sp = SpectralClustering(n_clusters=cluster_list[i],
                                    affinity='precomputed')
            sp.fit(affinity_matrix)
            silhouette_list = silhouette_samples(dist_matrix, sp.labels_,
                                                 metric="precomputed")
            queue_y[i] = np.mean(silhouette_list)

    n = len(cluster_list)
    nprocs = min(mp.cpu_count(), n)
    queue_y = mp.Array('d', [0.] * n)
    ps = []
    try:
        for idx in range(nprocs):
            p = mp.Process(target=_internal,
                           args=(cluster_list, affinity_matrix, dist_matrix,
                                 idx, nprocs, n, queue_y))
            p.start()
            ps.append(p)

        for p in ps:
            p.join()
    except (KeyboardInterrupt, SystemExit):
        extra._terminate(ps, 'Exit signal received\n')
    except Exception as e:
        extra._terminate(ps, 'ERROR: %s\n' % e)
    except:
        extra._terminate(ps, 'ERROR: Exiting with unknown exception\n')
    return queue_y


def plot_average_silhouette_spectral(X, n=30,
                                     min_clust=10,
                                     max_clust=None,
                                     verbose=True,
                                     interactive_mode=False,
                                     file_format='pdf'):
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
    A = extra.distance_to_affinity_matrix(X)

    plt.close()
    fig, ax = (plt.gcf(), plt.gca())
    fig.suptitle("Average silhouette for each number of clusters")

    if max_clust is None: max_clust = X.shape[0]
    cluster_list = np.unique(map(int, np.linspace(min_clust, max_clust, n)))
    y = multi_cut_spectral(cluster_list, A, X)
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
    return filename


if __name__ == '__main__':
    import pandas as pd
    from ignet.utils.extra import ensure_symmetry

    df = pd.read_csv('/home/fede/Dropbox/projects/Franco_Fabio_Marcat/'
                     'TM_matrix.csv', index_col=0)
    X = df.as_matrix()
    X = ensure_symmetry(X)

    plot_average_silhouette_dendrogram(X)
    plot_average_silhouette_spectral(X)
