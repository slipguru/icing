#!/usr/bin/env python
"""TODO doc."""
import logging
import seaborn as sns

from ..plotting import silhouette

__author__ = 'Federico Tomasi'


def analyse(X, labels, root='', plotting_context=None, file_format='pdf',
            force_silhouette=False):
    """Perform analysis.

    Parameters
    ----------
    X : array, shape = [n_samples, n_samples]
        Precomputed distance matrix.
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

    if force_silhouette or X.shape[0] < 8000:
        silhouette.plot_clusters_silhouette(X, labels, max(labels), root=root,
                                            file_format=file_format)
    else:
        logging.warn("Silhouette analysis is not performed due to the "
                     "matrix dimensions. With a matrix {0}x{0}, you would "
                     "need to allocate {1:.2f}MB in memory. If you know what "
                     "you are doing, specify 'force_silhouette = True' in the "
                     "config file in {}.\n"
                     .format(X.shape[0], X.shape[0]**2*8/(2.**20), root))
