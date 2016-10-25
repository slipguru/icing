#!/usr/bin/env python
"""Learning function module for the mutation level correction.

Author: Federico Tomasi
Copyright (c) 2016, Federico Tomasi.
Licensed under the FreeBSD license (see LICENSE.txt).
"""
from __future__ import division

import logging
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
# import re
import seaborn as sns; sns.set_context('poster')
# import joblib as jl

from scipy.optimize import curve_fit
from sklearn import mixture

from icing import parallel_distance
from icing.core import cloning
# from icing.models import model
from icing.utils import io, extra

from string_kernel.core.src.sum_string_kernel import sum_string_kernel


def remove_duplicate_junctions(igs_list):
    igs, juncs = [], []
    for ig in igs_list:
        junc = extra.junction_re(ig.junction)
        if junc not in juncs:
            igs.append(ig)
            juncs.append(junc)
    return igs, juncs


def calcDist(el1, el2):
    # j1 = extra.junction_re(el1.junction)
    # j2 = extra.junction_re(el2.junction)
    #
    # return 1. - sum_string_kernel(
    #     [j1, j2], min_kn=1, max_kn=5, lamda=.75,
    #     verbose=False, normalize=1)[0, 1]
    return cloning.sim_function(el1, el2, correct=False)


def make_hist(juncs1, juncs2, fn, lim_mut1, lim_mut2, type_ig='Mem',
              donor1='B4', donor2=None, bins=100, max_seqs=1000, min_seqs=100,
              ig1=None, ig2=None, is_intra=True):
    if os.path.exists(fn + '.npy'):
        logging.info(fn + '.npy esists.')
        return fn
    # sample if length is exceeded (for computational costs)
    if len(juncs1) < min_seqs or len(juncs2) < min_seqs:
        return ''

    from sklearn.utils import shuffle
    if len(juncs1) > max_seqs:
        ig1, juncs1 = shuffle(ig1, juncs1)
        ig1 = ig1[:max_seqs]
        juncs1 = juncs1[:max_seqs]
    if len(juncs2) > max_seqs:
        ig2, juncs2 = shuffle(ig2, juncs2)
        ig2 = ig2[:max_seqs]
        juncs2 = juncs2[:max_seqs]

    logging.info("Computing {}".format(fn))
    if is_intra:
        # dist2nearest = parallel_distance.dnearest_intra_padding(ig1, calcDist)
        # temp TODO XXX
        dist2nearest = parallel_distance.dnearest_inter_padding(ig1, ig1, calcDist)
    else:
        dist2nearest = parallel_distance.dnearest_inter_padding(ig1, ig2, calcDist)
    if not os.path.exists(fn.split('/')[0]):
        os.makedirs(fn.split('/')[0])
    np.save(fn, dist2nearest)
    # dist2nearest = np.array([np.min(r[r>0]) for r in X])
    plt.figure(figsize=(20, 10))
    plt.hist(dist2nearest, bins=bins, normed=True)
    try:
        if not donor2:
            plt.title("Distances between {} {}-{}% and {}-{}%"
                      .format(type_ig, lim_mut1[0], lim_mut1[1],
                              lim_mut2[0], lim_mut2[1]))
        else:
            plt.title("Distances between {}-{} {} {}-{}% and {}-{}%"
                      .format(donor1, donor2, type_ig, lim_mut1[0],
                              lim_mut1[1], lim_mut2[0], lim_mut2[1]))
    except Exception:
        pass
    plt.ylabel('Count')
    plt.xticks(np.linspace(0, .5, 21))
    plt.xlabel('Ham distance (normalised)')
    plt.savefig(fn + ".png")
    plt.close()
    del dist2nearest
    return fn


def intra_donor_distance(f='', lim_mut1=(0, 0), lim_mut2=(0, 0), type_ig='Mem',
                         quantity=.15, donor='B4', bins=100, max_seqs=1000):
    """Nearest distances intra donor.

    Subsets of Igs can be selected choosing two ranges of mutations.
    """
    fn = "{0}/dist2nearest_{0}_{2}-{3}_vs_{4}-{5}_{6}bins_norm_{7}maxseqs" \
         .format(donor, type_ig.lower(), lim_mut1[0], lim_mut1[1], lim_mut2[0],
                 lim_mut2[1], bins, max_seqs)
    if os.path.exists(fn + '.npy'):
        logging.info("File {} exists.".format(fn + '.npy'))
        return fn, max(lim_mut1[1], lim_mut2[1])

    n_tot = io.get_num_records(f)
    if max(lim_mut1[1], lim_mut2[1]) == 0:
        igs = io.read_db(f, filt=(lambda x: x.mut == 0), max_records=quantity*n_tot)
        igs1, juncs1 = remove_duplicate_junctions(igs)
        juncs2 = juncs1
        igs2 = igs1
    else:
        igs = io.read_db(f, filt=(lambda x: lim_mut1[0] < x.mut <= lim_mut1[1]), max_records=quantity*n_tot)
        igs1, juncs1 = remove_duplicate_junctions(igs)
        igs = io.read_db(f, filt=(lambda x: lim_mut2[0] < x.mut <= lim_mut2[1]), max_records=quantity*n_tot)
        igs2, juncs2 = remove_duplicate_junctions(igs)

    return make_hist(juncs1, juncs2, fn, lim_mut1, lim_mut2, type_ig, donor,
                     None, bins, max_seqs, ig1=igs1, ig2=igs2), max(lim_mut1[1], lim_mut2[1])


def inter_donor_distance(f1='', f2='', lim_mut1=(0, 0), lim_mut2=(0, 0),
                         type_ig='Mem', donor1='B4', donor2='B5', bins=100,
                         max_seqs=1000, quantity=.15):
    """Nearest distances inter donors.

    Igs involved can be selected by choosing two possibly different ranges
    of mutations.
    """
    fn = "{0}/dist2nearest_{0}vs{1}-{3}_vs_{4}-{5}_{6}bins_norm_{7}maxseqs" \
         .format(donor1, donor2, type_ig.lower(), lim_mut1[0], lim_mut1[1],
                 lim_mut2[0], lim_mut2[1], bins, max_seqs)
    if os.path.exists(fn + '.npy'):
        return fn, max(lim_mut1[1], lim_mut2[1])

    if max(lim_mut1[1], lim_mut2[1]) == 0:
        igs = io.read_db(f1, filt=(lambda x: x.mut == 0))
        _, juncs1 = remove_duplicate_junctions(igs)
        igs = io.read_db(f2, filt=(lambda x: x.mut == 0))
        _, juncs2 = remove_duplicate_junctions(igs)
    elif max(lim_mut1[1], lim_mut2[1]) < 0:
        # not specified: get at random
        igs = io.read_db(f1)
        _, juncs1 = remove_duplicate_junctions(igs)
        igs = io.read_db(f2)
        _, juncs2 = remove_duplicate_junctions(igs)
    else:
        igs = io.read_db(f1, filt=(lambda x: lim_mut1[0] < x.mut <= lim_mut1[1]))
        _, juncs1 = remove_duplicate_junctions(igs)
        igs = io.read_db(f2, filt=(lambda x: lim_mut2[0] < x.mut <= lim_mut2[1]))
        _, juncs2 = remove_duplicate_junctions(igs)

    juncs1 = juncs1[:int(quantity*len(juncs1))]
    juncs2 = juncs2[:int(quantity*len(juncs2))]
    return make_hist(juncs1, juncs2, fn, lim_mut1, lim_mut2, type_ig, donor1,
                     donor2, bins, max_seqs), max(lim_mut1[1], lim_mut2[1])


def all_intra_mut(db, quantity=0.15, bins=50, max_seqs=4000):
    """Create histograms and relative mutation levels."""
    logging.info("Analysing {} ...".format(db))
    out_fles, mut_lvls = [], []
    try:
        max_mut = int(io.get_max_mut(db))
        sets = [(0, 0)] + [(i - 1, i) for i in range(1, max_mut + 1)]
        for i, j in list(zip(sets, sets)):
            o, mut = intra_donor_distance(db, i, j, quantity=quantity,
                                          donor=db.split('/')[-1],
                                          bins=bins, max_seqs=max_seqs)
            out_fles.append(o)
            mut_lvls.append(mut)
    except Exception as e:
        logging.critical(e)

    d = dict()
    for i, f in enumerate(out_fles):
        d.setdefault(mut_lvls[i], []).append(f)
    return out_fles, mut_lvls, d


def create_alpha_plot(out_files, mut_levels, __my_dict__):
    sigmas = []
    muts = []
    mut_means = dict()
    mut_sigmas = dict()
    d_dict = dict()
    for k, v in __my_dict__.iteritems():
        for o in v:
            if not o:
                continue
            dist2nearest = np.array(np.load("{}.npy".format(o))).reshape(-1, 1)
            if dist2nearest.shape[0] < 2:
                logging.error("Cannot fit a Gaussian with two distances.")
                continue

            dist2nearest_2 = -(np.array(sorted(dist2nearest)).reshape(-1, 1))
            dist2nearest = np.array(list(dist2nearest_2) +
                                    list(dist2nearest)).reshape(-1, 1)

            try:
                # new sklearn GMM
                gmm = mixture.GaussianMixture(n_components=3,
                                              covariance_type='diag')
                gmm.fit(dist2nearest)
                mean = np.max(gmm.means_)
                sigma = gmm.covariances_[np.argmax(gmm.means_)]
            except AttributeError:
                # use old sklearn method
                gmm = mixture.GMM(n_components=3)
                gmm.fit(dist2nearest)

                mean = np.max(gmm.means_)
                sigma = gmm.covars_[np.argmax(gmm.means_)]

            mut_means.setdefault(k, []).append(mean)
            mut_sigmas.setdefault(k, []).append(sigma)

            # mut = mut_levels[i]
            if '0-0' in o:
                # considering naive. Extract optimal threshold
                plt.hist(dist2nearest, bins=50, normed=True)  # debug, print
                linspace = np.linspace(-.5, .5, 1000)[:, np.newaxis]
                #plt.plot(linspace, np.exp(gmm.score_samples(linspace)[0]), 'r')

                lin = np.linspace(0, .5, 10000)[:, np.newaxis]
                pred = gmm.predict(linspace)
                argmax = np.argmax(gmm.means_)
                idx = np.min(np.where(pred == argmax)[0])
                plt.axvline(x=lin[idx], linestyle='--', color='r')
                plt.gcf().savefig("threshold_naive.pdf")
                threshold_naive = lin[idx]  # threshold
                np.save("threshold_naive", threshold_naive)

            # mut_level_mem.append(mut)
            sigmas.append(sigma)
            muts.append(k)
            d_dict.setdefault(o.split('/')[0], dict()).setdefault(k, mean)

    norm_dict = dict()
    for k, v in d_dict.iteritems():
        mu_mut_0 = float(v.get(0, -1))
        if mu_mut_0 > -1:
            for l, z in v.iteritems():
                z = mu_mut_0 / z
                norm_dict.setdefault(l, []).append(z)

    x, y, e = [], [], []
    for k, v in norm_dict.iteritems():
        x.append(k)
        y.append(np.mean(v))
        e.append(np.var(v))

    if x[0] != 0:
        logging.warn("{} should be 0. Normalising factor is not given by "
                     "mu naive".format(x[0]))
    x, y = np.array(x), np.array(y)
    popt, pcov = curve_fit(extra.negative_exponential, x, y, p0=(1, 1e-1, 1))

    xp = np.linspace(0, 50, 1000)
    # plt.plot(x, y, linestyle=' ', marker='o', label='data')
    with sns.axes_style('whitegrid'):
        plt.figure()
        plt.errorbar(x, y, e, label='data')
        plt.plot(xp, extra.negative_exponential(xp, *popt), '-',
                 label=r'$\alpha$ (neg exp)', lw=2.5)
        plt.ylabel(r'$\mu$ Naive / $\mu$ Mem')
        plt.xlabel(r'Igs mutation level')
        plt.ylim([0., 1.])
        plt.xlim([0, 50])
        plt.legend(loc='lower left')
        plt.savefig("alpha_mut_plot_poster_notitle_dpi.pdf", transparent=True)
        plt.close()

    return popt, threshold_naive


def generate_correction_function(db, quantity):
    db_no_ext = ".".join(db.split(".")[:-1])
    filename = db_no_ext + "_correction_function"

    # case 1: file exists
    if os.path.exists(filename + ".npy") and os.path.exists("threshold_naive.npy"):
        logging.info("Best parameters exists. Loading them ...")
        popt = np.load(filename + ".npy")
        threshold_naive = np.load("threshold_naive.npy")

    # case 2: file not exists
    else:
        out_files, mut_levels, __my_dict__ = all_intra_mut(db, quantity=quantity)
        popt, threshold_naive = create_alpha_plot(out_files, mut_levels, __my_dict__)

        # save for later, in case of analysis on the same db
        np.save(filename, popt)

    from functools import partial
    return (
        partial(extra.negative_exponential, a=popt[0], c=popt[1], d=popt[2]),
        threshold_naive)
