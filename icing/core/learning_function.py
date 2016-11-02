#!/usr/bin/env python
"""Learning function module for the mutation level correction.

Author: Federico Tomasi
Copyright (c) 2016, Federico Tomasi.
Licensed under the FreeBSD license (see LICENSE.txt).
"""
from __future__ import division, print_function

import logging
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns; sns.set_context('poster')

from functools import partial
from scipy.optimize import curve_fit
from sklearn import mixture

from icing.core import cloning
from icing.core import parallel_distance
from icing.utils import io, extra
from string_kernel.core.src.sum_string_kernel import sum_string_kernel


def _remove_duplicate_junctions(igs_list):
    igs, juncs = [], []
    for ig in igs_list:
        junc = extra.junction_re(ig.junction)
        if junc not in juncs:
            igs.append(ig)
            juncs.append(junc)
    return igs, juncs


def _distance(el1, el2):
    # j1 = extra.junction_re(el1.junction)
    # j2 = extra.junction_re(el2.junction)
    #
    # return 1. - sum_string_kernel(
    #     [j1, j2], min_kn=1, max_kn=5, lamda=.75,
    #     verbose=False, normalize=1)[0, 1]
    return cloning.sim_function(el1, el2, correct=False, tol=1000)


def make_hist(juncs1, juncs2, fn, lim_mut1, lim_mut2, type_ig='Mem',
              donor1='B4', donor2=None, bins=100, max_seqs=1000, min_seqs=100,
              ig1=None, ig2=None, is_intra=True, sim_func_args=None):
    if os.path.exists(fn + '.npy'):
        logging.info(fn + '.npy esists.')
        return fn
    if len(juncs1) < min_seqs or len(juncs2) < min_seqs:
        # print("only {} few seqs for {} mut"
        #       .format(len(juncs1), max(lim_mut1[1], lim_mut2[1])))
        return ''

    # sample if length is exceeded (for computational costs)
    from sklearn.utils import shuffle
    if len(juncs1) > max_seqs:
        ig1, juncs1 = shuffle(ig1, juncs1)
        ig1 = ig1[:max_seqs]
        juncs1 = juncs1[:max_seqs]
    if len(juncs2) > max_seqs:
        ig2, juncs2 = shuffle(ig2, juncs2)
        ig2 = ig2[:max_seqs]
        juncs2 = juncs2[:max_seqs]

    sim_func_args['correct'] = False
    sim_func_args['tol'] = 1000
    df = partial(cloning.sim_function, **sim_func_args)
    logging.info("Computing %s", fn)
    if is_intra:
        # dist2nearest = parallel_distance.dnearest_intra_padding(ig1, df)
        # temp TODO XXX
        dist2nearest = parallel_distance.dnearest_inter_padding(ig1, ig1, df)
    else:
        dist2nearest = parallel_distance.dnearest_inter_padding(ig1, ig2, df)
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
    plt.xlim([0, .5])
    plt.xticks(np.linspace(0, .5, 11))
    plt.xlabel('Ham distance (normalised)')
    plt.savefig(fn + ".png")
    plt.close()
    del dist2nearest
    return fn


def intra_donor_distance(f='', lim_mut1=(0, 0), lim_mut2=(0, 0), type_ig='Mem',
                         quantity=.15, donor='B4', bins=100, max_seqs=1000,
                         min_seqs=100, sim_func_args=None):
    """Nearest distances intra donor.

    Subsets of Igs can be selected choosing two ranges of mutations.
    """
    fn = "{0}/dist2nearest_{0}_{1}-{2}_vs_{3}-{4}_{5}bins_norm_{6}maxseqs" \
         .format(donor, lim_mut1[0], lim_mut1[1], lim_mut2[0],
                 lim_mut2[1], bins, max_seqs)
    if os.path.exists(fn + '.npy'):
        logging.info("File %s exists.", fn + '.npy')
        return fn, max(lim_mut1[1], lim_mut2[1])

    n_tot = io.get_num_records(f)
    if max(lim_mut1[1], lim_mut2[1]) == 0:
        igs = io.read_db(f, filt=(lambda x: x.mut == 0),
                         max_records=quantity * n_tot)
        igs1, juncs1 = _remove_duplicate_junctions(igs)
        juncs2 = juncs1
        igs2 = igs1
    else:
        igs = io.read_db(f,
                         filt=(lambda x: lim_mut1[0] < x.mut <= lim_mut1[1]),
                         max_records=quantity * n_tot)
        igs1, juncs1 = _remove_duplicate_junctions(igs)
        igs = io.read_db(f,
                         filt=(lambda x: lim_mut2[0] < x.mut <= lim_mut2[1]),
                         max_records=quantity * n_tot)
        igs2, juncs2 = _remove_duplicate_junctions(igs)

    return make_hist(juncs1, juncs2, fn, lim_mut1, lim_mut2, type_ig, donor,
                     None, bins, max_seqs, min_seqs, ig1=igs1, ig2=igs2,
                     sim_func_args=sim_func_args), \
        max(lim_mut1[1], lim_mut2[1])


def inter_donor_distance(f1='', f2='', lim_mut1=(0, 0), lim_mut2=(0, 0),
                         type_ig='Mem', donor1='B4', donor2='B5', bins=100,
                         max_seqs=1000, quantity=.15, sim_func_args=None):
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
        _, juncs1 = _remove_duplicate_junctions(igs)
        igs = io.read_db(f2, filt=(lambda x: x.mut == 0))
        _, juncs2 = _remove_duplicate_junctions(igs)
    elif max(lim_mut1[1], lim_mut2[1]) < 0:
        # not specified: get at random
        igs = io.read_db(f1)
        _, juncs1 = _remove_duplicate_junctions(igs)
        igs = io.read_db(f2)
        _, juncs2 = _remove_duplicate_junctions(igs)
    else:
        igs = io.read_db(f1, filt=(lambda x: lim_mut1[0] < x.mut <= lim_mut1[1]))
        _, juncs1 = _remove_duplicate_junctions(igs)
        igs = io.read_db(f2, filt=(lambda x: lim_mut2[0] < x.mut <= lim_mut2[1]))
        _, juncs2 = _remove_duplicate_junctions(igs)

    juncs1 = juncs1[:int(quantity * len(juncs1))]
    juncs2 = juncs2[:int(quantity * len(juncs2))]
    return make_hist(juncs1, juncs2, fn, lim_mut1, lim_mut2, type_ig, donor1,
                     donor2, bins, max_seqs, sim_func_args=sim_func_args), \
        max(lim_mut1[1], lim_mut2[1])


def all_intra_mut(db, quantity=0.15, bins=50, max_seqs=4000, min_seqs=100,
                  sim_func_args=None):
    """Create histograms and relative mutation levels."""
    logging.info("Analysing %s ...", db)
    out_fles, mut_lvls = [], []
    try:
        max_mut = int(io.get_max_mut(db))
        sets = [(0, 0)] + [(i - 1, i) for i in range(1, max_mut + 1)]
        # sets = [(0, 0)] + zip(np.arange(0, max_mut, step),
        #                       np.arange(step, max_mut + step, step))
        # lin = np.linspace(0, max_mut, 11)
        # sets = [(0, 0)] + zip(lin[:-1], lin[1:])

        for i, j in list(zip(sets, sets)):
            o, mut = intra_donor_distance(db, i, j, quantity=quantity,
                                          donor=db.split('/')[-1],
                                          bins=bins, max_seqs=max_seqs,
                                          min_seqs=min_seqs,
                                          sim_func_args=sim_func_args)
            out_fles.append(o)
            mut_lvls.append(mut)
    except Exception as e:
        logging.critical(e)

    d = dict()
    for i, f in enumerate(out_fles):
        d.setdefault(mut_lvls[i], []).append(f)
    return out_fles, mut_lvls, d


def create_alpha_plot(files, mut_levels, my_dict):
    d_dict = dict()
    print("mydict:", my_dict)

    means, samples, mutations, thresholds, medians = [], [], [], [], []
    for k, v in my_dict.iteritems():
        for o in v:
            if not o:
                continue
            dist2nearest = np.array(np.load("{}.npy".format(o))).reshape(-1, 1)
            medians.append(np.median(dist2nearest))
            mean = np.median(dist2nearest)

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
                # gmmmean = np.max(gmm.means_)
                # gmmsigma = gmm.covariances_[np.argmax(gmm.means_)]
            except AttributeError:
                # use old sklearn method
                gmm = mixture.GMM(n_components=3)
                gmm.fit(dist2nearest)
                # gmmmean = np.max(gmm.means_)
                # gmmsigma = gmm.covars_[np.argmax(gmm.means_)]

            # Extract optimal threshold
            plt.hist(dist2nearest, bins=50, normed=True)  # debug, print

            lin = np.linspace(0, 1, 10000)[:, np.newaxis]
            # plt.plot(lin, np.exp(gmm.score_samples(lin)[0]), 'r')
            pred = gmm.predict(lin)
            try:
                idx = np.min(np.where(pred == np.argmax(gmm.means_))[0])
            except ValueError:
                print("Error", np.unique(pred))

            plt.axvline(x=lin[idx], linestyle='--', color='r')
            plt.gcf().savefig("threshold_naive{}.png".format(k))
            plt.close()
            threshold = lin[idx][0]  # threshold
            # np.save("threshold_naive", threshold)

            means.append(mean)
            samples.append(dist2nearest.shape[0])
            mutations.append(k)
            thresholds.append(threshold)
            d_dict.setdefault(o.split('/')[0], dict()).setdefault(k, mean)

    print(d_dict)
    for k, v in d_dict.iteritems():  # there is only one
        keys = np.array(sorted([x for x in v]))
        print(type(keys[0]))
        mean_values = np.array([np.mean(v[x]) for x in keys])
        errors = np.array([np.var(v[x]) for x in keys])

    means, samples, mutations, thresholds = \
        map(np.array, (means, samples, mutations, thresholds))

    idxs = np.array(samples).argsort()[-3:][::-1]
    idx = idxs[0]
    if thresholds[idx] == 0:
        idx = idxs[1]
    # x, y = keys, means[idx] - mean_values
    # x, y = keys, mean_values - np.min(mean_values) + 1.
    x, y = keys, np.max(mean_values) / mean_values
    # popt, _ = curve_fit(extra.negative_exponential, x, y, p0=(1, 1e-1, 1))

    xp = np.linspace(np.min(x), np.max(x), 1000)

    order = 3
    p3 = np.poly1d(np.polyfit(x, y, order))
    p2 = np.poly1d(np.polyfit(x, y, 2))
    with sns.axes_style('whitegrid'):
        plt.figure()
        plt.errorbar(x, y, errors, label='data')
        plt.plot(xp, p3(xp), '-', label='order ' + str(order))
        plt.plot(xp, p2(xp), '-', label='order 2')
        # plt.plot(xp, extra.negative_exponential(xp, *popt), '-',
        #          label=r'$\alpha$ (neg exp)', lw=2.5)
        # # plt.ylabel(r'$\mu$ Naive / $\mu$ Mem')
        plt.xlabel(r'Igs mutation level')
        # plt.ylim([0., 2.])
        # plt.xlim([0, 50])
        plt.legend(loc='lower left')
        plt.savefig("alpha_mut_plot.pdf", transparent=True)
        plt.close()

    # return popt, threshold_naive
    # return [0, 0, 0], 0
    return p2, thresholds[idx]


def generate_correction_function(db, quantity, sim_func_args=None):
    """Generate correction function on the databse analysed."""
    db_no_ext = ".".join(db.split(".")[:-1])
    filename = db_no_ext + "_correction_function.npy"

    # case 1: file exists
    if os.path.exists(filename) and os.path.exists("threshold_naive.npy"):
        logging.info("Best parameters exists. Loading them ...")
        popt = np.load(filename)
        threshold_naive = np.load("threshold_naive.npy")

    # case 2: file not exists
    else:
        files, muts, my_dict = all_intra_mut(
            db, quantity=quantity, min_seqs=4, sim_func_args=sim_func_args)
        popt, threshold_naive = create_alpha_plot(files, muts, my_dict)
        # save for later, in case of analysis on the same db
        # np.save(filename, popt)  # TODO

    # partial(extra.negative_exponential, a=popt[0], c=popt[1], d=popt[2]),
    return (popt, threshold_naive)
