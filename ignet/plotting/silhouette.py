#!/usr/bin/env python
"""Plotting functions for silhouette analysis."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.metrics import silhouette_samples  # , silhouette_score

from ..utils import utils

__author__ = 'Federico Tomasi'


def compute_silhouette_score(X, cluster_labels, n_clusters):
    """Plot the silhoeutte score for each cluster, given the distance matrix X.

    Parameters
    ----------
    X : array_like
        Distance matrix. Must be in the form returned by
        scipy.spatial.distance.squareform,
        e.g. squareform(1. - similarity_matrix.toarray())
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
    print "average silhouette_score: ", silhouette_avg

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
    fig.savefig('silhouette_analysis_{}.png'.format(utils.get_time()))
