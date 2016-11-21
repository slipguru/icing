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
from scipy.optimize import least_squares
from sklearn import mixture

from icing.core import cloning
from icing.core import parallel_distance
from icing.utils import io, extra
from string_kernel import sum_string_kernel


# def _remove_duplicate_junctions(igs_list):
#     igs, juncs = [], []
#     for ig in igs_list:
#         junc = extra.junction_re(ig.junction)
#         if junc not in juncs:
#             igs.append(ig)
#             juncs.append(junc)
#     return igs, juncs

def _remove_duplicate_junctions(igs):
    igs = list(igs)
    return igs, map(lambda x: extra.junction_re(x.junction), igs)


def make_hist(juncs1, juncs2, fn, lim_mut1, lim_mut2, type_ig='Mem',
              donor1='B4', donor2=None, bins=100, max_seqs=1000, min_seqs=0,
              ig1=None, ig2=None, is_intra=True, sim_func_args=None):
    if os.path.exists(fn + '.npy'):
        logging.critical(fn + '.npy esists.')
        return fn
    if len(juncs1) < min_seqs or len(juncs2) < min_seqs:
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
    dd = cloning.inverse_index(ig1 if is_intra else ig1 + ig2)
    if sim_func_args.get('method', None) in ('pcc', 'hypergeometric'):
        sim_func_args['sim_score_params'] = {
            'nV': len([x for x in dd if 'V' in x]),
            'nJ': len([x for x in dd if 'J' in x])}
    sim_func = partial(cloning.sim_function, **sim_func_args)
    logging.info("Computing %s", fn)
    if is_intra:
        dist2nearest = parallel_distance.dnearest_inter_padding(
            ig1, ig1, sim_func, filt=lambda x: 0 < x, func=max)
        # ig1, ig1, sim_func, filt=lambda x: 0 < x < 1, func=max)
    else:
        dist2nearest = parallel_distance.dnearest_inter_padding(
            ig1, ig2, sim_func, filt=lambda x: 0 < x < 1, func=max)
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
    plt.xlim([0, 1])
    plt.xticks(np.linspace(0, 1, 21))
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
        max_mut = io.get_max_mut(db)
        # if max_mut < 1:
        n_tot = io.get_num_records(db)
        lin = np.linspace(0, max_mut, n_tot / 5.)
        sets = [(0, 0)] + zip(lin[:-1], lin[1:])
        # else:
        #     sets = [(0, 0)] + [(i - 1, i) for i in range(1, int(max_mut) + 1)]
        if len(sets) == 1:
            # no correction needs to be applied
            return None
        # sets = [(0, 0)] + zip(np.arange(0, max_mut, step),
        #                       np.arange(step, max_mut + step, step))
        # print(sets)

        for i, j in list(zip(sets, sets)):
            o, mut = intra_donor_distance(
                db, i, j, quantity=quantity, donor=db.split('/')[-1],
                bins=bins, max_seqs=max_seqs, min_seqs=min_seqs,
                sim_func_args=sim_func_args)
            out_fles.append(o)
            mut_lvls.append(mut)
    except Exception as e:
        logging.critical(e)

    d = dict()
    for i, f in enumerate(out_fles):
        d.setdefault(mut_lvls[i], []).append(f)
    return d


def gaussian_fit(array):
    if array.shape[0] < 2:
        logging.error("Cannot fit a Gaussian with two distances.")
        return 0

    array_2 = -(np.array(sorted(array)).reshape(-1, 1))
    array = np.array(list(array_2) + list(array)).reshape(-1, 1)

    try:
        # new sklearn GMM
        gmm = mixture.GaussianMixture(n_components=3,
                                      covariance_type='diag')
        gmm.fit(array)
        # gmmmean = np.max(gmm.means_)
        # gmmsigma = gmm.covariances_[np.argmax(gmm.means_)]
    except AttributeError:
        # use old sklearn method
        gmm = mixture.GMM(n_components=3)
        gmm.fit(array)
        # gmmmean = np.max(gmm.means_)
        # gmmsigma = gmm.covars_[np.argmax(gmm.means_)]

    # Extract optimal threshold
    plt.hist(array, bins=50, normed=True)  # debug, print

    lin = np.linspace(0, 1, 10000)[:, np.newaxis]
    # plt.plot(lin, np.exp(gmm.score_samples(lin)[0]), 'r')
    pred = gmm.predict(lin)
    try:
        idx = np.min(np.where(pred == np.argmax(gmm.means_))[0])
    except ValueError:
        # print("Error", np.unique(pred))
        idx = 0

    plt.axvline(x=lin[idx], linestyle='--', color='r')
    # plt.gcf().savefig("threshold_naive{}.png".format(k))
    plt.close()
    threshold = lin[idx][0]  # threshold
    # np.save("threshold_naive", threshold)
    return threshold


def create_alpha_plot(my_dict, order=3, alpha_plot='alphaplot.pdf'):
    if my_dict is None:
        return (lambda x: 1), 0
    d_dict = dict()
    samples, thresholds = [], []
    for k, v in my_dict.iteritems():
        for o in (_ for _ in v if _):
            dist2nearest = np.array(np.load("{}.npy".format(o))).reshape(-1, 1)
            mean = np.mean(dist2nearest)
            samples.append(dist2nearest.shape[0])

            # for the threshold, fit a gaussian (unused for AP)
            thresholds.append(gaussian_fit(dist2nearest))
            d_dict.setdefault(o.split('/')[0], dict()).setdefault(k, mean)

    # print(d_dict)
    for k, v in d_dict.iteritems():  # there is only one
        keys = np.array(sorted([x for x in v]))
        mean_values = np.array([np.mean(v[x]) for x in keys])
        errors = np.array([np.var(v[x]) for x in keys])

    # print(samples)
    idxs = np.array(samples).argsort()[-3:][::-1]
    idx = idxs[0]
    if thresholds[idx] == 0:
        idx = idxs[1]
    idx2 = mean_values > 0
    keys, mean_values = keys[idx2], mean_values[idx2]
    errors = errors[idx2]
    x, y = np.array(keys), np.min(mean_values) / mean_values

    xp = np.linspace(np.min(x), np.max(x), 1000)[:, None]

    def model(x, u):
        return x[0] * (u ** 2 + x[1] * u) / (u ** 2 + x[2] * u + x[3])

    def jac(x, u, y):
        J = np.empty((u.size, x.size))
        den = u ** 2 + x[2] * u + x[3]
        num = u ** 2 + x[1] * u
        J[:, 0] = num / den
        J[:, 1] = x[0] * u / den
        J[:, 2] = -x[0] * num * u / den ** 2
        J[:, 3] = -x[0] * num / den ** 2
        return J

    x0 = np.array([2.5, 3.9, 4.15, 3.9])
    res = least_squares(
        lambda x, u, y: model(x, u) - y, x0,
        jac=jac, bounds=(0, 100), args=(x, y))  # , ftol=1e-12, loss='cauchy')
    poly = np.poly1d(np.polyfit(x, y, order))

    with sns.axes_style('whitegrid'):
        plt.figure()
        plt.errorbar(x, y, errors, label='data', marker='s')
        plt.plot(xp, poly(xp), '-', label='order ' + str(order))
        # plt.plot(xp, fff(xp), '-', label='interpolate')
        plt.plot(xp, model(res.x, xp), '-', label='least squares')
        plt.xlabel(r'Igs mutation level')
        plt.legend(loc='lower left')
        plt.savefig(alpha_plot, transparent=True)
        plt.close()

    # return poly, thresholds[idx]
    return partial(model, res.x), thresholds[idx]


def generate_correction_function(db, quantity, sim_func_args=None, order=3,
                                 root=''):
    """Generate correction function on the database analysed."""
    db_no_ext = ".".join(db.split(".")[:-1])
    filename = db_no_ext + "_correction_function.npy"

    # case 1: file exists
    alpha_plot = os.path.join(root, db_no_ext.split('/')[-1] + '_alphaplot.pdf')
    if os.path.exists(filename) and os.path.exists("threshold_naive.npy"):
        logging.critical("Best parameters exists. Loading them ...")
        popt = np.load(filename)
        threshold_naive = np.load("threshold_naive.npy")

    # case 2: file not exists
    else:
        my_dict = all_intra_mut(
            db, quantity=quantity, min_seqs=2, sim_func_args=sim_func_args)
        popt, threshold_naive = create_alpha_plot(my_dict, order, alpha_plot)
        # save for later, in case of analysis on the same db
        # np.save(filename, popt)  # TODO

    # partial(extra.negative_exponential, a=popt[0], c=popt[1], d=popt[2]),
    return (popt, threshold_naive, alpha_plot)
