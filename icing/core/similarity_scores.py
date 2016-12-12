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


def connection_specificity_index(nodes_a, nodes_b, pcc_ab, tot_nodes,
        nodes_connected_to_a, nodes_connected_to_b):
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


def _balance_contribution(common_V, common_J, tot_V, tot_J, r1, r2):
    w1 = w2 = 1.
    if r1 != 1. or r2 != 1.:
        w1 = r1 * (tot_V + tot_J) / (r1 * tot_V + r2 * tot_J)
        # w2 = 1. + tot_V / tot_J * (1. - w1)
        w2 = r2 / r1 * w1
    try:
        result = (w1 * common_V + w2 * common_J) / (tot_V + tot_J)
    except ZeroDivisionError:
        result = 0
    return result


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
    if r1 < 0 or r2 < 0 or (r1 == 0 and r2 == 0):
        raise ValueError("Weights cannot be negative")
    if not V_genes_A or not V_genes_B or (r1 == 0 and r2 != 0):
        return similarity_score_bipartite(J_genes_A, J_genes_B, method)
    if not J_genes_A or not J_genes_B or (r2 == 0 and r1 != 0):
        return similarity_score_bipartite(V_genes_A, V_genes_B, method)

    # enforce sets
    V_genes_A = set(V_genes_A)
    V_genes_B = set(V_genes_B)
    J_genes_A = set(J_genes_A)
    J_genes_B = set(J_genes_B)

    if method == 'jaccard':
        common_V = len(V_genes_A & V_genes_B)
        common_J = len(J_genes_A & J_genes_B)
        tot_V = len(V_genes_A | V_genes_B)
        tot_J = len(J_genes_A | J_genes_B)
        return _balance_contribution(common_V, common_J, tot_V, tot_J, r1, r2)

    elif method == 'simpson':
        common_V = len(V_genes_A & V_genes_B)
        common_J = len(J_genes_A & J_genes_B)
        tot_V = min(len(V_genes_A), len(V_genes_B))
        tot_J = min(len(J_genes_A), len(J_genes_B))
        return _balance_contribution(common_V, common_J, tot_V, tot_J, r1, r2)

    elif method == 'geometric':
        common_V = len(V_genes_A & V_genes_B)
        common_V *= common_V
        common_J = len(J_genes_A & J_genes_B)
        common_J *= common_J
        tot_V = len(V_genes_A) * len(V_genes_B)
        tot_J = len(J_genes_A) * len(J_genes_B)
        return _balance_contribution(common_V, common_J, tot_V, tot_J, r1, r2)

    elif method == 'cosine':
        common_V = len(V_genes_A & V_genes_B)
        common_J = len(J_genes_A & J_genes_B)
        tot_V = sqrt(len(V_genes_A) * len(V_genes_B))
        tot_J = sqrt(len(J_genes_A) * len(J_genes_B))
        return _balance_contribution(common_V, common_J, tot_V, tot_J, r1, r2)

    elif method == 'firstkul':
        # First Kulczynski coefficient
        # Note that the codomain is [0, inf)
        common_V = len(V_genes_A & V_genes_B)
        common_J = len(J_genes_A & J_genes_B)
        tot_V = len(V_genes_A) + len(V_genes_B) - 2 * common_V
        tot_J = len(J_genes_A) + len(J_genes_B) - 2 * common_J
        return _balance_contribution(common_V, common_J, tot_V, tot_J, r1, r2)

    elif method == 'dice':
        common_V = 2 * len(V_genes_A & V_genes_B)
        common_J = 2 * len(J_genes_A & J_genes_B)
        tot_V = len(V_genes_A) + len(V_genes_B)
        tot_J = len(J_genes_A) + len(J_genes_B)
        return _balance_contribution(common_V, common_J, tot_V, tot_J, r1, r2)

    elif method == 'russelrao':
        common_V = len(V_genes_A & V_genes_B)
        common_J = len(J_genes_A & J_genes_B)
        tot_V = sim_score_params.get('nV')
        tot_J = sim_score_params.get('nJ')
        return _balance_contribution(common_V, common_J, tot_V, tot_J, r1, r2)

    elif method == 'pcc':
        nv = sim_score_params.get('nV')
        len_va = len(V_genes_A)
        len_vb = len(V_genes_B)
        common_V = abs((len(V_genes_A & V_genes_B) * nv - len_va * len_vb))
        tot_V = sqrt(len_va * len_vb * (nv - len_va) * (nv - len_vb))

        nj = sim_score_params.get('nJ')
        len_ja = len(J_genes_A)
        len_jb = len(J_genes_B)
        common_J = abs((len(J_genes_A & J_genes_B) * nj - len_ja * len_jb))
        tot_J = sqrt(len_ja * len_jb * (nj - len_ja) * (nj - len_jb))
        return _balance_contribution(common_V, common_J, tot_V, tot_J, r1, r2)

    elif method == 'jaccard_new':
        w1 = r1 / r1 + r2
        w2 = 1 - w1
        return (w1 * jaccard_index(V_genes_A, V_genes_B) +
                w2 * jaccard_index(J_genes_A, J_genes_B))

    elif method == 'simpson_new':
        w1 = r1 / r1 + r2
        w2 = 1 - w1
        return (w1 * simpson_index(V_genes_A, V_genes_B) +
                w2 * simpson_index(J_genes_A, J_genes_B))

    elif method == 'geometric_new':
        w1 = r1 / r1 + r2
        w2 = 1 - w1
        return (w1 * geometric_index(V_genes_A, V_genes_B) +
                w2 * geometric_index(J_genes_A, J_genes_B))

    elif method == 'cosine_new':
        w1 = r1 / r1 + r2
        w2 = 1 - w1
        return (w1 * cosine_index(V_genes_A, V_genes_B) +
                w2 * cosine_index(J_genes_A, J_genes_B))

    elif method == 'pcc_new':
        w1 = r1 / r1 + r2
        w2 = 1 - w1
        nv = sim_score_params.get('nV')
        nj = sim_score_params.get('nJ')
        return (w1 * pcc_index(V_genes_A, V_genes_B, nv) +
                w2 * pcc_index(J_genes_A, J_genes_B, nj))
    else:
        raise NotImplementedError("Method {} not supported\n".format(method))
