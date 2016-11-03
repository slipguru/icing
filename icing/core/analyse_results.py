#!/usr/bin/env python
"""Perform the analysis on the results of `ici_run.py`.

Author: Federico Tomasi
Copyright (c) 2016, Federico Tomasi.
Licensed under the FreeBSD license (see LICENSE.txt).
"""
import logging
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster import hierarchy

from icing.plotting import silhouette
from icing.utils import extra


def analyse(sm, labels, root='', plotting_context=None, file_format='pdf',
            force_silhouette=False, threshold=None):
    """Perform analysis.

    Parameters
    ----------
    sm : array, shape = [n_samples, n_samples]
        Precomputed similarity matrix.
    labels : array, shape = [n_samples]
        Association of each sample to a clsuter.
    root : string
        The root path for the output creation.
    plotting_context : dict, None, or one of {paper, notebook, talk, poster}
        See seaborn.set_context().
    file_format : ('pdf', 'png')
        Choose the extension for output images.
    """
    sns.set_context(plotting_context)

    if force_silhouette or sm.shape[0] < 8000:
        silhouette.plot_clusters_silhouette(1. - sm.toarray(), labels,
                                            max(labels), root=root,
                                            file_format=file_format)
    else:
        logging.warn(
            "Silhouette analysis is not performed due to the "
            "matrix dimensions. With a matrix %ix%i, you would need to "
            "allocate %.2fMB in memory. If you know what you are doing, "
            "specify 'force_silhouette = True' in the config file in %s, "
            "then re-execute the analysis.\n", sm.shape[0], sm.shape[0],
            sm.shape[0]**2 * 8 / (2.**20), root)

    # Generate dendrogram
    import scipy.spatial.distance as ssd
    Z = hierarchy.linkage(ssd.squareform(1. - sm.toarray()), method='complete',
                          metric='euclidean')

    plt.close()
    fig, (ax) = plt.subplots(1, 1)
    fig.set_size_inches(20, 15)
    hierarchy.dendrogram(Z, ax=ax)
    ax.axhline(threshold, color="red", linestyle="--")
    plt.show()
    filename = os.path.join(root, 'dendrogram_{}.{}'
                                  .format(extra.get_time(), file_format))
    fig.savefig(filename)
    logging.info('Figured saved %s', filename)

    plt.close()
    fig, (ax) = plt.subplots(1, 1)
    fig.set_size_inches(20, 15)
    plt.hist(1. - sm.toarray(), bins=50, normed=False)
    plt.ylim([0, 10])
    fig.savefig(filename + "_histogram_distances.pdf")
