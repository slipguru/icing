#!/usr/bin/env python
"""Plotting functions for silhouette analysis.

Author: Federico Tomasi
Copyright (c) 2016, Federico Tomasi.
Licensed under the FreeBSD license (see LICENSE.txt).
"""
from __future__ import print_function, division

import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import multiprocessing as mp
import sys; sys.setrecursionlimit(10000)
import seaborn as sns; sns.set_context('notebook')
import logging
import pandas as pd

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

# Prior sets to calculate how good are the clusters
sset1 = ('CLL011_1_K', 'CLL026_1_K', 'CLL154_1_K', 'CLL266_1_K',
         'CLL270_1_K', 'CLL336_1_K', 'CLL360_1_K', 'CLLG063_1_K',
         'G063_1_K', 'G086_1_K', 'PA0375_1_K', 'SI186_1_K', 'SI242_1_K',
         'SI5_1_K', 'TS53_1_K')
sset2 = ('CD0310_2_L', 'CLL175_2_L', 'CLL282_2_L', 'CLL412_2_L',
         'CLL668_2_L', 'CLL785_2_L', 'SR0112_2_L')
sset4 = ('G_183_4_K', 'G_342_4_K', 'G_733_4_K', 'G_G031_4_K', 'SI110_4_K')
sset6 = ('CLL068_6_K', 'CLL258_6_K', 'CLL861_6_K', 'CLL900_6_K', 'G107_6_K',
         'RC87_6_K', 'SI153_6_K')
sset7 = ('SI15_7_L', 'SI21_7_L')
sset8 = ('G_039_8_K', 'G_057_8_K', 'G_114_8_K', 'G_657_8_K', 'G_MS0115_8_K',
         'G_NI099_8_K', 'G_RC25_8_K', 'G_SI89_8_K')
sset9 = ('AD0221_9_L', 'CLL051_9_L')
set_subset = sorted([sset1, sset2, sset4, sset6, sset7, sset8,
                     sset9], key=lambda x: len(x), reverse=True)

sset_K = (
    'AC0120__K',
    'AE119_no_K',
    'AM0220_96_K',
    'AP0406__K',
    'AR0090__K',
    'AS0407__K',
    'A_FB0103__K',
    'A_OMO201_no_K',
    'A_TC230_no_K',
    'BC0047__K',
    'BL0188__K',
    'BL0445__K',
    'BV0176__K',
    'CA0251_61_K',
    'CA0331__K',
    'CC0244_N13_K',
    'CI0005__K',
    'CLL002__K',
    'CLL003__K',
    'CLL007__K',
    'CLL008__K',
    'CLL011_1_K',
    'CLL017_59_K',
    'CLL018__K',
    'CLL019__K',
    'CLL021__K',
    'CLL023__K',
    'CLL026_1_K',
    'CLL035__K',
    'CLL038__K',
    'CLL042__K',
    'CLL047__K',
    'CLL048__K',
    'CLL056__K',
    'CLL059__K',
    'CLL064__K',
    'CLL066__K',
    'CLL067__K',
    'CLL068_6_K',
    'CLL071__K',
    'CLL079__K',
    'CLL081__K',
    'CLL083__K',
    'CLL085__K',
    'CLL087__K',
    'CLL088__K',
    'CLL093__K',
    'CLL108__K',
    'CLL110__K',
    'CLL1109__K',
    'CLL112__K',
    'CLL118__K',
    'CLL119__K',
    'CLL121__K',
    'CLL122__K',
    'CLL125__K',
    'CLL127__K',
    'CLL135__K',
    'CLL138__K',
    'CLL141__K',
    'CLL153__K',
    'CLL154_1_K',
    'CLL156__K',
    'CLL165__K',
    'CLL178__K',
    'CLL215__K',
    'CLL216__K',
    'CLL249__K',
    'CLL258_6_K',
    'CLL266_1_K',
    'CLL270_1_K',
    'CLL288__K',
    'CLL302__K',
    'CLL309__K',
    'CLL321__K',
    'CLL331__K',
    'CLL334__K',
    'CLL336_1_K',
    'CLL343__K',
    'CLL354__K',
    'CLL358__K',
    'CLL360_1_K',
    'CLL361__K',
    'CLL364__K',
    'CLL377__K',
    'CLL402_59?_K',
    'CLL403__K',
    'CLL408__K',
    'CLL415__K',
    'CLL483__K',
    'CLL495__K',
    'CLL569__K',
    'CLL585__K',
    'CLL606__K',
    'CLL861_6_K',
    'CLL900_6_K',
    'CLLG006__K',
    'CLLG008__K',
    'CLLG009__K',
    'CLLG011__K',
    'CLLG012__K',
    'CLLG013__K',
    'CLLG014__K',
    'CLLG015__K',
    'CLLG019__K',
    'CLLG020__K',
    'CLLG021__K',
    'CLLG022__K',
    'CLLG025__K',
    'CLLG027__K',
    'CLLG033__K',
    'CLLG034__K',
    'CLLG036__K',
    'CLLG040__K',
    'CLLG041__K',
    'CLLG045__K',
    'CLLG050__K',
    'CLLG054__K',
    'CLLG057__K',
    'CLLG062__K',
    'CLLG063_1_K',
    'CLLG065__K',
    'CLLG067__K',
    'CLLG069__K',
    'CLLG070__K',
    'CLLG071__K',
    'CLLG072__K',
    'CLLG073__K',
    'CLLG074__K',
    'CLLGE083__K',
    'CLLGE128__K',
    'CLLGE137__K',
    'CLLGE151__K',
    'CLLGE154__K',
    'CLLGE156__K',
    'CLLGE260__K',
    'CLLGN02__K',
    'CLLGN03__K',
    'CLLGN04__K',
    'CLLGN06__K',
    'CLLGN07__K',
    'CLLGN08__K',
    'CLLGN13__K',
    'CLLGN14__K',
    'CLLGN18__K',
    'CLLGN19__K',
    'CLLGN22__K',
    'CLLGN27__K',
    'CLLGN28__K',
    'CLLGN29__K',
    'CLLGN30__K',
    'CLLMF09__K',
    'CLLRC012__K',
    'CLLRC014__K',
    'CP0104__K',
    'CR0203__K',
    'CR0297__K',
    'CS13__K',
    'CS16__K',
    'CS19__K',
    'CS28__K',
    'CS3__K',
    'CS30__K',
    'CS32__K',
    'CS34__K',
    'CS35__K',
    'CS37__K',
    'CS38__K',
    'CS4__K',
    'CS43__K',
    'CS44__K',
    'CS45__K',
    'CS55__K',
    'CS58__K',
    'CS6__K',
    'CS61__K',
    'CS70__K',
    'CV0287__K',
    'CZ10__K',
    'CZ101__K',
    'CZ104__K',
    'CZ105__K',
    'CZ112__K',
    'CZ12__K',
    'CZ23__K',
    'CZ31__K',
    'CZ32__K',
    'CZ38__K',
    'CZ4__K',
    'CZ47__K',
    'CZ5__K',
    'CZ58__K',
    'CZ59__K',
    'CZ72__K',
    'CZ79__K',
    'CZ85__K',
    'CZ89__K',
    'CZ99__K',
    'DA0346__K',
    'DA0445__K',
    'DC0414__K',
    'FA0020__K',
    'FC0388__K',
    'FT0451__K',
    'G025__K',
    'G027__K',
    'G033__K',
    'G034__K',
    'G036__K',
    'G062__K',
    'G063_1_K',
    'G065__K',
    'G067__K',
    'G082__K',
    'G085__K',
    'G086_1_K',
    'G091__K',
    'G102__K',
    'G106__K',
    'G107_6_K',
    'G108__K',
    'G109__K',
    'G112__K',
    'G113__K',
    'GA0398__K',
    'GD0158__K',
    'GE135__K',
    'GE226__K',
    'GL0368__K',
    'GR0033__K',
    'GV0149_19 NO_K',
    'GV0377__K',
    'G_030__K',
    'G_033_no 16?_K',
    'G_039_8_K',
    'G_040__K',
    'G_057_8_K',
    'G_087__K',
    'G_111__K',
    'G_114_8_K',
    'G_128_no_K',
    'G_158__K',
    'G_183_4_K',
    'G_185__K',
    'G_186_no_K',
    'G_196__K',
    'G_240_4?_K',
    'G_342_4_K',
    'G_368__K',
    'G_416__K',
    'G_657_8_K',
    'G_733_4_K',
    'G_936__K',
    'G_BB238_no_K',
    'G_CA0345__K',
    'G_CZ128__K',
    'G_G016_no_K',
    'G_G031_4_K',
    'G_G061__K',
    'G_G066__K',
    'G_G087__K',
    'G_G111__K',
    'G_G115__K',
    'G_GE388__K',
    'G_GE401_ _K',
    'G_GE456_no_K',
    'G_GE519__K',
    'G_GN01__K',
    'G_LN06__K',
    'G_MS0115_8_K',
    'G_NI099_8_K',
    'G_NY202__K',
    'G_RC25_8_K',
    'G_SI102_no_K',
    'G_SI107_no_K',
    'G_SI196_8???_K',
    'G_SI291_8 NO_K',
    'G_SI89_8_K',
    'G_SI92__K',
    'G_TF001_no_K',
    'G_VD0026_no_K',
    'HG0135__K',
    'IG0181__K',
    'II0440__K',
    'KN1202__K',
    'KN1214__K',
    'KN1405__K',
    'KN1475__K',
    'KN1482__K',
    'KN1490__K',
    'KN1498__K',
    'KN1513__K',
    'KN1542__K',
    'KN1589__K',
    'KN1605__K',
    'KN1622__K',
    'KN1659__K',
    'KN1713__K',
    'KN1737__K',
    'KN1810__K',
    'KN1847__K',
    'KN1867__K',
    'KN1868__K',
    'KN2231__K',
    'KN2238__K',
    'KN2325__K',
    'KN2344__K',
    'KN2448__K',
    'KN2527__K',
    'KN541__K',
    'KN950__K',
    'KP1055__K',
    'KP1060__K',
    'KP1157__K',
    'KP1165__K',
    'KP1173__K',
    'KP1178__K',
    'KP1182__K',
    'KP1201__K',
    'KP1229__K',
    'KP1289__K',
    'KP130__K',
    'KP1306__K',
    'KP131__K',
    'KP14__K',
    'KP1403__K',
    'KP1504__K',
    'KP1540__K',
    'KP1610__K',
    'KP1627__K',
    'KP1697__K',
    'KP191__K',
    'KP1971__K',
    'KP1986__K',
    'KP2033__K',
    'KP213__K',
    'KP2143__K',
    'KP2184__K',
    'KP220__K',
    'KP2251__K',
    'KP2255__K',
    'KP297__K',
    'KP324__K',
    'KP325__K',
    'KP341__K',
    'KP343__K',
    'KP380__K',
    'KP409__K',
    'KP416__K',
    'KP438__K',
    'KP439__K',
    'KP452__K',
    'KP512__K',
    'KP585__K',
    'KP590__K',
    'KP608__K',
    'KP656__K',
    'KP668__K',
    'KP690__K',
    'KP711__K',
    'KP717__K',
    'KP79__K',
    'KP795__K',
    'KP835__K',
    'KP84__K',
    'KP860__K',
    'KP886__K',
    'KP896__K',
    'KP975__K',
    'KPCLL4__K',
    'LA0393_2 NO_K',
    'LG0038_no_K',
    'LP0255_no_K',
    'LS0378_7? Probab NO_K',
    'MA0432__K',
    'MC0081_no_K',
    'ML0298_G8_K',
    'NB0033_no_K',
    'OI0282_no_K',
    'PA0375_1_K',
    'PF0177_no_K',
    'RC10__K',
    'RC12_16_K',
    'RC15__K',
    'RC17__K',
    'RC35__K',
    'RC4__K',
    'RC87_6_K',
    'RC9__K',
    'RG0443__K',
    'SG0402_no_K',
    'SI101__K',
    'SI104__K',
    'SI105__K',
    'SI110_4_K',
    'SI112__K',
    'SI114__K',
    'SI117_28_K',
    'SI120__K',
    'SI125__K',
    'SI126__K',
    'SI128__K',
    'SI133__K',
    'SI139__K',
    'SI141__K',
    'SI149__K',
    'SI152__K',
    'SI153_6_K',
    'SI162__K',
    'SI164__K',
    'SI168__K',
    'SI17__K',
    'SI174__K',
    'SI183__K',
    'SI186_1_K',
    'SI197__K',
    'SI199_N13_K',
    'SI2__K',
    'SI200__K',
    'SI211__K',
    'SI22_3??_K',
    'SI223__K',
    'SI228__K',
    'SI23__K',
    'SI230__K',
    'SI234__K',
    'SI237__K',
    'SI241__K',
    'SI242_1_K',
    'SI243__K',
    'SI278__K',
    'SI29__K',
    'SI3__K',
    'SI34__K',
    'SI36__K',
    'SI40__K',
    'SI43__K',
    'SI44__K',
    'SI5_1_K',
    'SI50__K',
    'SI51__K',
    'SI53__K',
    'SI54__K',
    'SI55__K',
    'SI58__K',
    'SI60__K',
    'SI61__K',
    'SI63__K',
    'SI66__K',
    'SI68__K',
    'SI69__K',
    'SI70__K',
    'SI71__K',
    'SI73__K',
    'SI74__K',
    'SI75__K',
    'SI76__K',
    'SI8__K',
    'SI82__K',
    'SI86__K',
    'SI90__K',
    'SI91__K',
    'SI94__K',
    'SI95__K',
    'SI99__K',
    'TS27__K',
    'TS32__K',
    'TS34__K',
    'TS41__K',
    'TS5__K',
    'TS53_1_K',
    'TS71__K',
    'TS9__K',
    'VG0442__K')
sset_L = (
    'AD0221_9_L',
    'AG0403_no_L',
    'AM106__L',
    'A_FB0332__L',
    'A_PF07__L',
    'BA0421__L',
    'BL0277__L',
    'BP0326__L',
    'CA0160__L',
    'CD0310_2_L',
    'CLL014__L',
    'CLL020__L',
    'CLL027__L',
    'CLL041__L',
    'CLL051_9_L',
    'CLL058__L',
    'CLL099__L',
    'CLL105__L',
    'CLL115__L',
    'CLL123__L',
    'CLL126__L',
    'CLL129__L',
    'CLL136__L',
    'CLL139__L',
    'CLL147__L',
    'CLL152__L',
    'CLL169_  _L',
    'CLL172__L',
    'CLL175_2_L',
    'CLL242__L',
    'CLL255__L',
    'CLL282_2_L',
    'CLL374__L',
    'CLL400__L',
    'CLL412_2_L',
    'CLL417__L',
    'CLL561__L',
    'CLL562__L',
    'CLL668_2_L',
    'CLL785_2_L',
    'CLL799__L',
    'CLL984__L',
    'CLLG001__L',
    'CLLG002__L',
    'CLLG003__L',
    'CLLG005__L',
    'CLLG007__L',
    'CLLG010__L',
    'CLLG023__L',
    'CLLG024__L',
    'CLLG026__L',
    'CLLG029__L',
    'CLLG030__L',
    'CLLG035__L',
    'CLLG037__L',
    'CLLG039__L',
    'CLLG044__L',
    'CLLG046__L',
    'CLLG049__L',
    'CLLG052__L',
    'CLLG053__L',
    'CLLG055__L',
    'CLLG068__L',
    'CLLG075__L',
    'CLLGE080__L',
    'CLLGE136__L',
    'CLLGE146__L',
    'CLLGE211__L',
    'CLLGE265__L',
    'CLLGN09__L',
    'CLLGN11__L',
    'CLLGN12__L',
    'CLLGN17__L',
    'CLLGN20__L',
    'CLLGN24__L',
    'CLLGN25__L',
    'CLLRC088__L',
    'CLLRF22__L',
    'CR0159__L',
    'CS0361__L',
    'CS10__L',
    'CS15__L',
    'CS20__L',
    'CS33__L',
    'CS5__L',
    'CS50__L',
    'CS62__L',
    'CT0356__L',
    'CZ108__L',
    'CZ109__L',
    'CZ113__L',
    'CZ36__L',
    'CZ37__L',
    'CZ52__L',
    'CZ56__L',
    'CZ68__L',
    'CZ8__L',
    'CZ80__L',
    'CZ86__L',
    'CZ9__L',
    'DA0059__L',
    'DE0459__L',
    'DF0319__L',
    'DL0367__L',
    'DT0300__L',
    'FG0187__L',
    'G001__L',
    'G002__L',
    'G010__L',
    'G015__L',
    'G024__L',
    'G026_35_L',
    'G030__L',
    'G035__L',
    'G037__L',
    'G039__L',
    'G042__L',
    'G081__L',
    'G103__L',
    'G104__L',
    'G105__L',
    'G110__L',
    'G114__L',
    'GA0141__L',
    'GE129__L',
    'G_001_no_L',
    'G_005__L',
    'G_075__L',
    'G_078__L',
    'G_089__L',
    'G_1064__L',
    'G_417__L',
    'G_ARG009__L',
    'G_BM0223__L',
    'G_CS109__L',
    'G_G089_no_L',
    'G_G093__L',
    'G_GN015__L',
    'IA0079__L',
    'LP0076_no_L',
    'LR0418__L',
    'MA0088_56_L',
    'MF0379_no_L',
    'MS0321_no_L',
    'MV0334__L',
    'N1319__L',
    'N1408__L',
    'N1445__L',
    'N1492__L',
    'N1533__L',
    'N1553__L',
    'N1568__L',
    'N1590__L',
    'N1613__L',
    'N1638__L',
    'N1691__L',
    'N1703__L',
    'N1707__L',
    'N1816__L',
    'N1838__L',
    'N1887__L',
    'N2277__L',
    'N2307__L',
    'N2386__L',
    'N2396__L',
    'N2542__L',
    'P1000__L',
    'P1097__L',
    'P1132__L',
    'P1321__L',
    'P1430__L',
    'P1529__L',
    'P1531__L',
    'P1597__L',
    'P160__L',
    'P161__L',
    'P1618__L',
    'P1659__L',
    'P173__L',
    'P1837__L',
    'P2422__L',
    'P280__L',
    'P326__L',
    'P451__L',
    'P458__L',
    'P460__L',
    'P611__L',
    'P630__L',
    'P66__L',
    'P675__L',
    'P684__L',
    'P689__L',
    'P775__L',
    'P788__L',
    'P906__L',
    'PG0122_no_L',
    'PG0355_no_L',
    'PS0420_26_L',
    'RA0023__L',
    'RC107__L',
    'RC3__L',
    'RC68__L',
    'RF0489__L',
    'SA0488__L',
    'SI103__L',
    'SI108__L',
    'SI116__L',
    'SI127__L',
    'SI135__L',
    'SI138__L',
    'SI14__L',
    'SI15_7_L',
    'SI161__L',
    'SI18__L',
    'SI181__L',
    'SI188__L',
    'SI21_7_L',
    'SI212__L',
    'SI213__L',
    'SI235__L',
    'SI238__L',
    'SI244__L',
    'SI256__L',
    'SI26__L',
    'SI30__L',
    'SI33__L',
    'SI35__L',
    'SI37__L',
    'SI47__L',
    'SI56__L',
    'SI62__L',
    'SI64__L',
    'SI77__L',
    'SI79__L',
    'SI98__L',
    'SR0112_2_L',
    'TS29__L',
    'TS30__L',
    'TS74__L')
set_light_chain = sorted([sset_K, sset_L], key=lambda x: len(x), reverse=True)


def best_intersection(id_list, cluster_dict):
    """Compute score between id_list and each list in dict, take the best."""
    set1 = set(id_list)
    best_score = 0.
    best_set = ()
    best_key = -1
    for k in cluster_dict:
        set2 = set(cluster_dict[k])
        score = len(set1 & set2) / len(set1)
        if score > best_score or best_key == -1:
            best_score = score
            best_set = set2
            best_key = k
    # print(set1, "and best", best_set, best_score)
    if best_key != -1:
        del cluster_dict[best_key]
    return best_score, cluster_dict, best_set


def calc_stability(clusts, other_clusts):
    stability = 0.
    nclusts = len(clusts)
    # orig_clusts = clusts.copy()
    orig_clusts = []
    for _ in other_clusts:
        res, clusts, best_set = best_intersection(_, clusts)
        n_unknown = len([xx for xx in best_set if xx.endswith("_")])
        print("{1:.2f}"  # (K {2:.2f}, L {3:.2f})"
              .format(_, res,
                      (len(best_set) in (0, n_unknown) and -1) or (len([xx for xx in best_set if xx.endswith("_K") or xx.endswith("_")])-n_unknown) * 100. / (len(best_set)-n_unknown),
                      (len(best_set) in (0, n_unknown) and -1) or (len([xx for xx in best_set if xx.endswith("_L") or xx.endswith("_")])-n_unknown) * 100. / (len(best_set)-n_unknown)
                      ), end=' ')
        stability += res
        orig_clusts.append(best_set)

    km = []
    lm = []
    for c in orig_clusts:
        if len(c) > 0:
            n_unknown = len([xx for xx in c if xx.endswith("_")])
            if n_unknown < len(c):
                k_res = (len([xx for xx in c if xx.endswith("_K") or
                         xx.endswith("_")])-n_unknown) * 100. / (len(c)-n_unknown)

                l_res = (len([xx for xx in c if xx.endswith("_L") or
                         xx.endswith("_")])-n_unknown) * 100. / (len(c)-n_unknown)
                if k_res > l_res:
                    km.append(k_res)
                elif k_res < l_res:
                    lm.append(l_res)
                else:
                    km.append(k_res)
                    lm.append(l_res)

    # print("Km {:.2f}, Lm {:.2f}".format(np.mean(km), np.mean(lm)), end=' ')
    print("\nstability: {:.3f}, {} clusts (light), {} nclusts -- [{:.2f}%] k[{:.2f}] l[{:.2f}]"
          .format(stability, len(other_clusts), nclusts,
                  stability * 100. / len(other_clusts), np.mean(km), np.mean(lm)))
    # km = []
    # lm = []
    # for i in orig_clusts:
    #     c = orig_clusts[i]
    #     if len(c) > 0:
    #         n_unknown = len([xx for xx in c if xx.endswith("_")])
    #         if n_unknown < len(c):
    #             k_res = (len([xx for xx in c if xx.endswith("_K") or
    #                      xx.endswith("_")])-n_unknown) * 100. / (len(c)-n_unknown)
    #
    #             l_res = (len([xx for xx in c if xx.endswith("_L") or
    #                      xx.endswith("_")])-n_unknown) * 100. / (len(c)-n_unknown)
    #             if k_res > l_res:
    #                 km.append(k_res)
    #             elif k_res < l_res:
    #                 lm.append(l_res)
    #             else:
    #                 km.append(k_res)
    #                 lm.append(l_res)
    # print("Km {:.2f}, Lm {:.2f}".format(np.mean(km), np.mean(lm)))


def single_silhouette_dendrogram(dist_matrix, Z, threshold, mode='clusters',
                                 method='single'):
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
    cols = list(pd.read_csv('/home/fede/Dropbox/projects/Franco_Fabio_Marcat/'
                            'TM_matrix_ID_SUBSET_light_noduplicates.csv',
                            index_col=0).columns)
    with open("res_hierarchical_{:03d}_clust.csv".format(nclusts), 'w') as f:
        for a, b in zip(cols, cluster_labels):
            f.write("{}, {}\n".format(a, b))

    # Go, stability!
    # List of ids
    ids = np.array(cols)

    # Create original clusters
    clusters = {}
    for i in np.unique(cluster_labels):
        clusters[i] = ids[np.where(cluster_labels == i)]

    calc_stability(clusters, set_subset)

    # # Shuffle samples
    # from sklearn.utils import shuffle
    # idxs = list(range(0, dist_matrix.shape[0]))
    # idxs, ids = shuffle(idxs, ids)
    #
    # # Remove some random samples from dist_matrix
    # nsamples_to_remove = 20
    # idxs = idxs[:-nsamples_to_remove]
    # ids = ids[:-nsamples_to_remove]
    #
    # dm = dist_matrix[idxs][:, idxs]
    # links_sampling = linkage(scipy.spatial.distance.squareform(dm),
    #                          method=method, metric='euclidean')
    #
    # cluster_labels_sampling = fcluster(links_sampling, threshold, 'distance')
    #
    # # Create sampled clusters
    # clusters_sampling = {}
    # stability = 0.
    # for i in np.unique(cluster_labels_sampling):
    #     clusters_sampling[i] = ids[np.where(cluster_labels_sampling == i)]
    #     res, clusters = best_intersection(clusters_sampling[i], clusters)
    #     # print("Stability for {}: {:.3f}".format(i, res))
    #     stability += res
    # nclusts_sampling = np.unique(cluster_labels_sampling).shape[0]
    #
    # print("stability: {:.3f} with {} clusts, {:.3f}%".format(stability, nclusts_sampling, stability * 100. / nclusts_sampling))
    # with open("res_hierarchical_{:.2f}_clust_{:.2f}_stability_sampling.csv"
    #           .format(nclusts_sampling, stability), 'w') as f:
    #     for a, b in zip(cols, cluster_labels_sampling):
    #         f.write("{}, {}\n".format(a, b))
    # from scipy.cluster.hierarchy import dendrogram
    # plt.close()
    # f = plt.gcf()
    # dendrogram(Z)
    # f.savefig("dendrogram_{:.2f}_clust_{:.2f}_stability_sampling-tr.png"
    #           .format(nclusts_sampling, stability))
    # plt.close()
    # f = plt.gcf()
    # dendrogram(links_sampling)
    # f.savefig("dendrogram_{:.2f}_clust_{:.2f}_stability_sampling-sa.png"
    #           .format(nclusts_sampling, stability))
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
                         method='single', n_jobs=-1):
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
                  queue_x, queue_y, mode='clusters', method='single'):
        for i in range(idx, arr_length, n_jobs):
            queue_x[i], queue_y[i] = single_silhouette_dendrogram(
                dist_matrix, Z, threshold_arr[i], mode, method)

    if n_jobs == -1:
        n_jobs = min(mp.cpu_count(), n)
    queue_x, queue_y = mp.Array('d', [0.] * n), mp.Array('d', [0.] * n)
    ps = []
    try:
        for idx in range(n_jobs):
            p = mp.Process(target=_internal,
                           args=(dist_matrix, Z, threshold_arr, idx, n_jobs, n,
                                 queue_x, queue_y, mode, method))
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
                                       xlim=None,
                                       figsize=None,
                                       n_jobs=-1):
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

        x, y = multi_cut_dendrogram(X, Z, threshold_arr, n, mode, method, n_jobs)
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


def multi_cut_spectral(cluster_list, affinity_matrix, dist_matrix, n_jobs=-1):
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
                                    norm_laplacian=True)
            sp.fit(affinity_matrix)
            cols = list(pd.read_csv('/home/fede/Dropbox/projects/Franco_Fabio_Marcat/'
                                    'TM_matrix_ID_SUBSET_light_noduplicates.csv', index_col=0).columns)
            with open("res_spectral_{:03d}_clust.csv".format(cluster_list[i]), 'w') as f:
                for a, b in zip(cols, sp.labels_):
                    f.write("{}, {}\n".format(a, b))

            cluster_labels = sp.labels_.copy()
            # nclusts = np.unique(cluster_labels).shape[0]

            # Go, stability!
            # List of ids
            ids = np.array(cols)

            # Create original clusters
            clusters = {}
            for _ in np.unique(cluster_labels):
                clusters[_] = ids[np.where(cluster_labels == _)]

            # clust2 = clusters.copy()
            calc_stability(clusters, set_subset)
            # calc_stability(clust2, set_light_chain)

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
                                     file_format='pdf',
                                     n_jobs=-1):
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
    A = extra.distance_to_affinity_matrix(X, delta=.175)

    plt.close()
    fig, ax = (plt.gcf(), plt.gca())
    fig.suptitle("Average silhouette for each number of clusters")

    if max_clust is None:
        max_clust = X.shape[0]
    cluster_list = np.unique(map(int, np.linspace(min_clust, max_clust, n)))
    y = multi_cut_spectral(cluster_list, A, X, n_jobs=n_jobs)
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
    # from adenine.core import plotting
    # plotting.eigs(
    #     '', A,
    #     filename=os.path.join(path, "eigs_spectral_{}.{}"
    #                                 .format(extra.get_time(), file_format)),
    #     n_components=50,
    #     normalised=True,
    #     rw=True
    #     )
    #
    # return filename


if __name__ == '__main__':
    import pandas as pd
    from icing.utils.extra import ensure_symmetry
    from icing.plotting import silhouette

    df = pd.read_csv('/home/fede/Dropbox/projects/Franco_Fabio_Marcat/'
                     'TM_matrix_ID_SUBSET_light_noduplicates.csv', index_col=0)
    X = df.as_matrix()
    X = ensure_symmetry(X)

    silhouette.plot_average_silhouette_dendrogram(
        X, min_threshold=.7, max_threshold=1.1, n=200, xticks=range(0, 50, 4),
        xlim=[0, 50], figsize=(20, 8),
        method_list=('median', 'ward', 'complete'))
    silhouette.plot_average_silhouette_spectral(X, min_clust=2, max_clust=10, n=10)
