from __future__ import division, print_function
import sys

def jaccard_index(nodes_A, nodes_B):
    """Jaccard index of a bipartite graph."""
    return len(nodes_A & nodes_B) / len(nodes_A | nodes_B)

def simpson_index(nodes_A, nodes_B):
    """Simpson index of a bipartite graph."""
    return len(nodes_A & nodes_B) / min(len(nodes_A), len(nodes_B))

def geometric_index(nodes_A, nodes_B):
    """Geometric index of a bipartite graph."""
    return len(nodes_A & nodes_B) ** 2 / (len(nodes_A) * len(nodes_B))

def cosine_index(nodes_A, nodes_B):
    """Cosine index of a bipartite graph."""
    from math import sqrt
    return len(nodes_A & nodes_B) / sqrt(len(nodes_A) * len(nodes_B))

def pcc_index(nodes_A, nodes_B):
    """Pearson correlation coeafficient"""
    raise NotImplementedError("PCC index not implemented")

def hypergeometric_index(nodes_A, nodes_B):
    raise NotImplementedError("hypergeometric index not implemented")

def connection_specificity_index(nodes_A, nodes_B):
    raise NotImplementedError("CSI index not implemented")

def similarity_score_bipartite(nodes_connected_to_A, nodes_connected_to_B, method='jaccard'):
    if not nodes_connected_to_A or not nodes_connected_to_B:
        return 0.
    nodes_A, nodes_B = set(nodes_connected_to_A), set(nodes_connected_to_B)
    if method.lower() == 'jaccard':
        return jaccard_index(nodes_A, nodes_B)
    elif method.lower() == 'simpson':
        return simpson_index(nodes_A, nodes_B)
    elif method.lower() == 'geometric':
        return geometric_index(nodes_A, nodes_B)
    elif method.lower() == 'cosine':
        return cosine_index(nodes_A, nodes_B)
    else:
        print("Method {} not supported".format(method))
        return -1

def similarity_score_tripartite(V_genes_A, V_genes_B, J_genes_A, J_genes_B,
                                r1=1., r2=1., method='jaccard'):
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
        The computed similarity score between A and B. Values are in range [0,1].
    """
    if not V_genes_A or not V_genes_B:
        # print("V genes unspecified or empty.")
        return 0.
    if r1 < 0 or r2 < 0:
        raise ValueError("Weights cannot be negative")
    if not J_genes_A or not J_genes_B:
        # print("J genes unspecified or zero weight. Computing similarity between V genes ...")
        sys.stderr.write("J genes unspecified or zero weight. Computing similarity between V genes ..."
                         "J1 = {}, J2 = {}\n".format(J_genes_A, J_genes_B))
        return similarity_score_bipartite(V_genes_A, V_genes_B, method)
    V_genes_A, V_genes_B, J_genes_A, J_genes_B = map(set,(V_genes_A, V_genes_B, J_genes_A, J_genes_B))
    w1, w2 = 1., 1.

    if method.lower() == 'jaccard':
        tot_V = len(V_genes_A | V_genes_B)
        tot_J = len(J_genes_A | J_genes_B)
        if r1 != 1. or r2 != 1.:
            sys.stdout.write("Recalculating w1 and w2, based on r1 = {.2f} and r2 = {.2f}\n".format(r1,r2))
            w1 = r1 * (tot_V + tot_J) / (r1 * tot_V + r2 * tot_J)
            # w2 = (tot_V + tot_J) / tot_J - w1 * tot_V / tot_J
            w2 = 1. + tot_V / tot_J * (1. - w1)
        return (w1 * len(V_genes_A & V_genes_B) + w2 * len(J_genes_A & J_genes_B)) / (tot_V + tot_J)

    elif method.lower() == 'simpson':
        min_V = min(len(V_genes_A), len(V_genes_B))
        min_J = min(len(J_genes_A), len(J_genes_B))
        if r1 != 1. or r2 != 1.:
            sys.stdout.write("Recalculating w1 and w2, based on r1 = {.2f} and r2 = {.2f}\n".format(r1,r2))
            w1 = r1 * (min_V + min_J) / (r1 * min_V + r2 * min_J)
            # w2 = (min_V + min_J) / min_J - w1 * min_V / min_J
            w2 = 1. + min_V / min_J * (1. - w1)
        return (w1 * len(V_genes_A & V_genes_B) + w2 * len(J_genes_A & J_genes_B)) / (min_V + min_J)
    else:
        sys.stderr.write("Method {} not supported\n".format(method))
        return -1
