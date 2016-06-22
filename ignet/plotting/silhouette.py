#!/usr/bin/env python
"""Plotting functions for silhouette analysis."""
from __future__ import print_function

import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import multiprocessing as mp
import sys; sys.setrecursionlimit(10000)
import seaborn

from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_samples  # , silhouette_score

from ..externals import Tango
from ..utils import utils
from ..utils.utils import _terminate


__author__ = 'Federico Tomasi'


def plot_clusters_silhouette(X, cluster_labels, n_clusters):
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
    print("average silhouette_score: ", silhouette_avg)

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
    filename = 'silhouette_analysis_{}.png'.format(utils.get_time())
    fig.savefig(filename)
    print("Figure saved in {}".format(os.path.abspath(filename)))


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
        _terminate(ps, 'Exit signal received\n')
    except Exception as e:
        _terminate(ps, 'ERROR: %s\n' % e)
    except:
        _terminate(ps, 'ERROR: Exiting with unknown exception\n')
    return queue_x, queue_y


def plot_average_silhouette_dendrogram(
                            X=None, config_file='config.py', method_list=None,
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
                  idx, nprocs, arr_length, queue_y):
        for i in range(idx, arr_length, nprocs):
            sp = SpectralClustering(n_clusters=cluster_list[i],
                                    affinity='precomputed')
            sp.fit(affinity_matrix)
            silhouette_list = silhouette_samples(dist_matrix, sp.labels_,
                                                 metric="precomputed")
            queue_y[i] = np.mean(silhouette_list)

    n = len(cluster_list)
    nprocs = min(mp.cpu_count(), n)
    queue_y = mp.Array('d', [0.]*n)
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
        _terminate(ps, 'Exit signal received\n')
    except Exception as e:
        _terminate(ps, 'ERROR: %s\n' % e)
    except:
        _terminate(ps, 'ERROR: Exiting with unknown exception\n')
    return queue_y


def plot_average_silhouette_spectral(
                X=None, fn=None, verbose=True, interactive_mode=False):
    """Plot average silhouette for some clusters, using an affinity matrix.

    Parameters
    ----------
    X : array-like
        2-dimensional distance matrix in sparse format.
    fn : str, optional
        If X is None, load X from fn as a standard csv file.
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
        try:
            import pandas as pd
            X = pd.io.parsers.read_csv(fn, header=0, index_col=0).as_matrix()
        except:
            sys.stderr.write("ERROR: {} cannot be read\n".format(fn))
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
    utils.mkpath(path)
    filename = os.path.join(path, "res_silh_HC_{}.png".format(utils.get_time()))
    plt.savefig(filename)
    return filename
