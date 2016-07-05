# Imports
import os, sys, re
from itertools import chain
from pandas import DataFrame
from pandas.io.parsers import read_csv
from time import time
from Bio.Seq import translate

# IgCore imports
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from IgCore import printProgress
from DbCore import getDistMat
from DbCore import calcDistances

from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

# Defaults
default_translate = False
default_distance = 0.0
default_bygroup_model = 'hs1f'
default_hclust_model = 'chen2010'
default_norm = 'len'
default_sym = 'avg'
default_linkage = 'single'


# TODO:  moved models into core, can wait until package conversion
# Path to model files
model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')

# Amino acid Hamming distance
aa_model = getDistMat(n_score=1, gap_score=0, alphabet='aa')

# DNA Hamming distance
ham_model = getDistMat(n_score=0, gap_score=0, alphabet='dna')

# Human 1-mer model
hs1f = DataFrame([[0.00, 2.08, 1.00, 1.75],
                  [2.08, 0.00, 1.75, 1.00],
                  [1.00, 1.75, 0.00, 2.08],
                  [1.75, 1.00, 2.08, 0.00]],
                 index=['A', 'C', 'G', 'T'],
                 columns=['A', 'C', 'G', 'T'],
                 dtype=float)
hs1f_model = getDistMat(hs1f)

# Mouse 1-mer model
smith96 = DataFrame([[0.00, 2.86, 1.00, 2.14],
                     [2.86, 0.00, 2.14, 1.00],
                     [1.00, 2.14, 0.00, 2.86],
                     [2.14, 1.00, 2.86, 0.00]],
                    index=['A', 'C', 'G', 'T'],
                    columns=['A', 'C', 'G', 'T'],
                    dtype=float)
m1n_model = getDistMat(smith96)

# Human 5-mer DNA model
hs5f_file = os.path.join(model_path, 'HS5F_Distance.tab')
hs5f_model = read_csv(hs5f_file, sep='\t', index_col=0)


def getModelMatrix(model):
    """
    Simple wrapper to get distance matrix from model name

    Arguments:
    model = model name

    Return:
    a pandas.DataFrame containing the character distance matrix
    """
    if model == 'aa':
        return(aa_model)
    elif model == 'ham':
        return(ham_model)
    elif model == 'm1n':
        return(m1n_model)
    elif model == 'hs1f':
        return(hs1f_model)
    elif model == 'hs5f':
        return(hs5f_model)
    else:
        sys.stderr.write('Unrecognized distance model: %s.\n' % model)


def indexJunctions(db_iter, fields=None, mode='gene', action='first'):
    """
    Identifies preclonal groups by V, J and junction length

    Arguments:
    db_iter = an iterator of IgRecords defined by readDbFile
    fields = additional annotation fields to use to group preclones;
             if None use only V, J and junction length
    mode = specificity of alignment call to use for assigning preclones;
           one of ('allele', 'gene')
    action = how to handle multiple value fields when assigning preclones;
             one of ('first', 'set')

    Returns:
    a dictionary of {(V, J, junction length):[IgRecords]}
    """
    # Define functions for grouping keys
    if mode == 'allele' and fields is None:
        def _get_key(rec, act):
            return (rec.getVAllele(act), rec.getJAllele(act), len(rec.junction))
    # THIS --by toma
    elif mode == 'gene' and fields is None:
        def _get_key(rec, act):
            return (rec.getVGene(act), rec.getJGene(act), len(rec.junction))
    elif mode == 'allele' and fields is not None:
        def _get_key(rec, act):
            vdj = [rec.getVAllele(act), rec.getJAllele(act), len(rec.junction)]
            ann = [rec.toDict().get(k, None) for k in fields]
            return tuple(chain(vdj, ann))
    elif mode == 'gene' and fields is not None:
        def _get_key(rec, act):
            vdj = [rec.getVGene(act), rec.getJGene(act), len(rec.junction)]
            ann = [rec.toDict().get(k, None) for k in fields]
            return tuple(chain(vdj, ann))

    start_time = time()
    clone_index = {}
    rec_count = 0
    for rec in db_iter:
        key = _get_key(rec, action)

        # Print progress
        if rec_count == 0:
            print 'PROGRESS> Grouping sequences'

        printProgress(rec_count, step=1000, start_time=start_time)
        rec_count += 1

        # Assigned passed preclone records to key and failed to index None
        if all([k is not None for k in key]):
            # TODO:  Has much slow. Should have less slow.
            if action == 'set':

                f_range = range(2, 3 + (len(fields) if fields else 0) )
                vdj_range = range(2)

                # Check for any keys that have matching columns and junction length and overlapping genes/alleles
                to_remove = []
                if len(clone_index) > 0 and not key in clone_index:
                    key = list(key)
                    for k in clone_index:
                        if all([key[i] == k[i] for i in f_range]):
                            if all([not set(key[i]).isdisjoint(set(k[i])) for i in vdj_range]):
                                for i in vdj_range:  key[i] = tuple(set(key[i]).union(set(k[i])))
                                to_remove.append(k)

                # Remove original keys, replace with union of all genes/alleles and append values to new key
                val = [rec]
                val += list(chain(*(clone_index.pop(k) for k in to_remove)))
                clone_index[tuple(key)] = clone_index.get(tuple(key),[]) + val

            elif action == 'first':
                clone_index.setdefault(key, []).append(rec)
        else:
            # TODO: weird return object for missing data case
            clone_index.setdefault((0,0,0), []).append(rec)

    printProgress(rec_count, step=1000, start_time=start_time, end=True)

    return clone_index


def formClusters(dists, link, distance):
    """Form clusters based on hierarchical clustering of input distance matrix
    with linkage type and cutoff distance
    :param dists: numpy matrix of distances
    :param link: linkage type for hierarchical clustering
    :param distance: distance at which to cut into clusters
    :return: list of cluster assignments
    """
    # Make distance matrix square
    dists = squareform(dists)
    # Compute linkage
    links = linkage(dists, link)

    # import matplotlib.pyplot as plt
    # from scipy.cluster import hierarchy
    # plt.figure(figsize=(15,5))
    # p = hierarchy.dendrogram(links)

    # Break into clusters based on cutoff
    clusters = fcluster(links, distance, criterion='distance')
    return clusters


def distanceClones(records, model=default_bygroup_model, distance=default_distance,
                   dist_mat=None, norm=default_norm, sym=default_sym,
                   linkage=default_linkage):
    """
    Separates a set of IgRecords into clones

    Arguments:
    records = an iterator of IgRecords
    model = substitution model used to calculate distance
    distance = the distance threshold to assign clonal groups
    dist_mat = pandas DataFrame of pairwise nucleotide or amino acid distances
    norm = normalization method
    sym = symmetry method
    linkage = type of linkage

    Returns:
    a dictionary of lists defining {clone number: [IgRecords clonal group]}
    """
    # Get distance matrix if not provided
    if dist_mat is None:  dist_mat = getModelMatrix(model)

    # Determine length of n-mers
    if model in ['hs1f', 'm1n', 'aa', 'ham']:
        nmer_len = 1
    elif model in ['hs5f']:
        nmer_len = 5
    else:
        sys.stderr.write('Unrecognized distance model: %s.\n' % model)

    # Define unique junction mapping
    junc_map = {}
    for ig in records:
        # Check if junction length is 0
        if ig.junction_length == 0:
            return None

        junc = re.sub('[\.-]','N', str(ig.junction))
        if model == 'aa':  junc = translate(junc)

        junc_map.setdefault(junc, []).append(ig)

    # Process records
    if len(junc_map) == 1:
        return {1:records}

    # Define junction sequences
    junctions = list(junc_map.keys())

    # Calculate pairwise distance matrix based on junctions
    dists = calcDistances(junctions, nmer_len, dist_mat, norm, sym)

    #TODO add function to estimate where are the mutations on the two sequences
    # and assign a weight of (non-)similarity when computing the distance between them
    # w = computeWeight(seq1, seq2)
    #print len(list(records))
    return dists

    # Perform hierarchical clustering
    clusters = formClusters(dists, linkage, distance)

    # Turn clusters into clone dictionary
    clone_dict = {}
    for i, c in enumerate(clusters):
        clone_dict.setdefault(c, []).extend(junc_map[junctions[i]])

    return clone_dict
