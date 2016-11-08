#!/usr/bin/env python
"""Graph-based similarity scores.

Author: Federico Tomasi
Copyright (c) 2016, Federico Tomasi.
Licensed under the FreeBSD license (see LICENSE.txt).
"""
from __future__ import division, print_function

from math import sqrt


def jaccard_index(nodes_a, nodes_b):
    """Jaccard index of a bipartite graph."""
    return len(nodes_a & nodes_b) / len(nodes_a | nodes_b)


def simpson_index(nodes_a, nodes_b):
    """Simpson index of a bipartite graph."""
    return len(nodes_a & nodes_b) / min(len(nodes_a), len(nodes_b))


def geometric_index(nodes_a, nodes_b):
    """Geometric index of a bipartite graph."""
    common_nodes = len(nodes_a & nodes_b)
    return common_nodes * common_nodes / (len(nodes_a) * len(nodes_b))


def cosine_index(nodes_a, nodes_b):
    """Cosine index of a bipartite graph."""
    return len(nodes_a & nodes_b) / sqrt(len(nodes_a) * len(nodes_b))


def pcc_index(nodes_a, nodes_b, ny):
    """Pearson correlation coeafficient.

    The result has been shifted to ensure a codomain of [0,1] instead
    of [-1,1].
    """
    len_a = len(nodes_a)
    len_b = len(nodes_b)
    return abs((len(nodes_a & nodes_b) * ny - len_a * len_b) /
               sqrt(len_a * len_b * (ny - len_a) * (ny - len_b)))


def hypergeometric_index(nodes_a, nodes_b):
    """Hypergeometric index of a bipartite graph."""
    raise NotImplementedError("hypergeometric index not implemented")


def connection_specificity_index(nodes_a, nodes_b):
    """Connection specificity index of a bipartite graph."""
    raise NotImplementedError("CSI index not implemented")


def similarity_score_bipartite(nodes_connected_to_A, nodes_connected_to_B,
                               method='jaccard'):
    """Similarity score for bipartite graphs.

    Parameters
    ----------
    nodes_connected_to_{A, B} : list
        List of nodes associated to {A, B}.
    method : 'jaccard', 'simpson', 'geometric', 'cosine'
        Method to use to calculate similarity score.

    Returns
    -------
    similarity_score : float
        The computed similarity score between A and B.
        Values are in range [0,1].
    """
    if not nodes_connected_to_A or not nodes_connected_to_B:
        return 0.
    nodes_a, nodes_b = set(nodes_connected_to_A), set(nodes_connected_to_B)
    if method.lower() == 'jaccard':
        return jaccard_index(nodes_a, nodes_b)
    elif method.lower() == 'simpson':
        return simpson_index(nodes_a, nodes_b)
    elif method.lower() == 'geometric':
        return geometric_index(nodes_a, nodes_b)
    elif method.lower() == 'cosine':
        return cosine_index(nodes_a, nodes_b)
    else:
        raise ValueError("Method %s not supported" % method)


def similarity_score_tripartite(V_genes_A, V_genes_B, J_genes_A, J_genes_B,
                                r1=1., r2=1., method='jaccard',
                                sim_score_params=None):
    """Similarity score for tripartite graphs.

    Parameters
    ----------
    V_genes_{A, B} : list
        List of genes of type V associated to node {A, B}.
    J_genes_{A, B} : list
        List of genes of type J associated to node {A, B}.
    r1 : float, optional, default = 1.
        Weight coefficient for V genes. w1 / w2 = r1 / r2
    r2 : float, optional, default = 1.
        Weight coefficient for J genes. w1 / w2 = r1 / r2
    method : string, optional, default = 'jaccard'
        Method to use to calculate similarity score.

    Returns
    -------
    similarity_score : float
        The computed similarity score between A and B.
        Values are in range [0,1].
    """
    if not V_genes_A or not V_genes_B or (r1 == 0 and r2 != 0):
        # sys.stderr.write("V genes unspecified or zero weight."
        #                  "Computing similarity between J genes ..."
        #                  "V1 = {}, V2 = {}\n".format(V_genes_A, V_genes_B))
        return similarity_score_bipartite(J_genes_A, J_genes_B, method)
    if r1 < 0 or r2 < 0 or (r1 == 0 and r2 == 0):
        raise ValueError("Weights cannot be negative")
    if not J_genes_A or not J_genes_B or (r2 == 0 and r1 != 0):
        # sys.stderr.write("J genes unspecified or zero weight."
        #                  "Computing similarity between V genes ..."
        #                  "J1 = {}, J2 = {}\n".format(J_genes_A, J_genes_B))
        return similarity_score_bipartite(V_genes_A, V_genes_B, method)

    # enforce sets
    V_genes_A = set(V_genes_A)
    V_genes_B = set(V_genes_B)
    J_genes_A = set(J_genes_A)
    J_genes_B = set(J_genes_B)
    w1 = w2 = 1.

    if method.lower() == 'jaccard':
        tot_V = len(V_genes_A | V_genes_B)
        tot_J = len(J_genes_A | J_genes_B)
        if r1 != 1. or r2 != 1.:
            # sys.stdout.write("Recalculating w1 and w2, based on "
            #                  "r1 = {.2f} and r2 = {.2f}\n".format(r1, r2))
            w1 = r1 * (tot_V + tot_J) / (r1 * tot_V + r2 * tot_J)
            # w2 = (tot_V + tot_J) / tot_J - w1 * tot_V / tot_J
            w2 = 1. + tot_V / tot_J * (1. - w1)
        return (w1 * len(V_genes_A & V_genes_B) +
                w2 * len(J_genes_A & J_genes_B)) / (tot_V + tot_J)

    elif method.lower() == 'simpson':
        min_V = min(len(V_genes_A), len(V_genes_B))
        min_J = min(len(J_genes_A), len(J_genes_B))
        if r1 != 1. or r2 != 1.:
            w1 = r1 * (min_V + min_J) / (r1 * min_V + r2 * min_J)
            # w2 = (min_V + min_J) / min_J - w1 * min_V / min_J
            w2 = 1. + min_V / min_J * (1. - w1)
        return (w1 * len(V_genes_A & V_genes_B) +
                w2 * len(J_genes_A & J_genes_B)) / (min_V + min_J)

    elif method.lower() == 'geometric':
        tot_V = len(V_genes_A) * len(V_genes_B)
        tot_J = len(J_genes_A) * len(J_genes_B)
        if r1 != 1. or r2 != 1.:
            w1 = r1 * (tot_V + tot_J) / (r1 * tot_V + r2 * tot_J)
            # w2 = (min_V + min_J) / min_J - w1 * min_V / min_J
            w2 = 1. + tot_V / tot_J * (1. - w1)

        common_V = len(V_genes_A & V_genes_B)
        common_J = len(J_genes_A & J_genes_B)
        return (w1 * common_V * common_V +
                w2 * common_J * common_J) / (tot_V + tot_J)

    elif method.lower() == 'jaccard_new':
        w1 = r1 / r1 + r2
        w2 = 1 - w1
        return (w1 * jaccard_index(V_genes_A, V_genes_B) +
                w2 * jaccard_index(J_genes_A, J_genes_B))

    elif method.lower() == 'simpson_new':
        w1 = r1 / r1 + r2
        w2 = 1 - w1
        return (w1 * simpson_index(V_genes_A, V_genes_B) +
                w2 * simpson_index(J_genes_A, J_genes_B))

    elif method.lower() == 'geometric_new':
        w1 = r1 / r1 + r2
        w2 = 1 - w1
        return (w1 * geometric_index(V_genes_A, V_genes_B) +
                w2 * geometric_index(J_genes_A, J_genes_B))

    elif method.lower() == 'cosine_new':
        w1 = r1 / r1 + r2
        w2 = 1 - w1
        return (w1 * cosine_index(V_genes_A, V_genes_B) +
                w2 * cosine_index(J_genes_A, J_genes_B))

    elif method.lower() == 'pcc':
        w1 = r1 / r1 + r2
        w2 = 1 - w1
        nv = sim_score_params.get('nV')
        nj = sim_score_params.get('nJ')
        return (w1 * pcc_index(V_genes_A, V_genes_B, nv) +
                w2 * pcc_index(J_genes_A, J_genes_B, nj))
    else:
        raise NotImplementedError("Method {} not supported\n".format(method))
