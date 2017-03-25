"""Learning function sklearn-like."""
from __future__ import division, print_function

import logging
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import scipy.stats
import six
import warnings

from functools import partial
from itertools import chain
# from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from sklearn import mixture
from sklearn.utils import shuffle

from icing.core import cloning
from icing.core import parallel_distance
from icing.models.model import model_matrix
from icing.utils import io, extra

from icing.core.learning_function import remove_duplicate_junctions
from icing.core.learning_function import mean_confidence_interval
from icing.core.learning_function import _gaussian_fit


class LearningFunction(object):

    def __init__(self, database, quantity=1, igsimilarity=None, order=3,
                 root='', min_seqs=10, max_seqs=1000, bins=50):
        self.database = database
        self.quantity = quantity
        self.igsimilarity = igsimilarity
        self.order = order
        self.root = root
        self.min_seqs = min_seqs
        self.max_seqs = max_seqs
        self.bins = bins

    def learn(self, my_dict):
        if my_dict is None:
            logging.critical("Cannot learn function with empty dict")
            return lambda _: 1, 0
        d_dict = dict()
        samples, thresholds = [], []
        for k, v in six.iteritems(my_dict):
            for o in (_ for _ in v if _):
                dnearest = np.array(np.load("{}.npz".format(o))['X']) \
                    .reshape(-1, 1)
                var = np.var(dnearest)
                if var == 0:
                    continue
                med = np.median(dnearest)
                mean, _, _, h = mean_confidence_interval(dnearest)
                samples.append(dnearest.shape[0])
                d_dict.setdefault(o.split('/')[0], dict()).setdefault(k, [med, h])

                # for the threshold, fit a gaussian (unused for AP)
                thresholds.append(_gaussian_fit(dnearest))

        for k, v in six.iteritems(d_dict):  # there is only one
            xdata = np.array(sorted([x for x in v]))
            ydata = np.array([np.mean(v[x][0]) for x in xdata])
            yerr = np.array([np.mean(v[x][1]) for x in xdata])

        # Take only significant values, higher than 0
        mask = ydata > 0
        xdata = xdata[mask]
        if xdata.shape[0] < 2:
            logging.critical("Too few points to learn function")
            # no correction can be applied
            return lambda _: 1, 0

        ydata = ydata[mask]
        ydata = ydata[0] / ydata  # normalise
        yerr = yerr[mask]

        # res = least_squares(
        #     lambda x, u, y: least_squares_mdl(x, u) - y,
        #     x0=np.array([2.5, 3.9, 4.15, 3.9]),
        #     jac=least_squares_jacobian, bounds=(0, 100), args=(xdata, ydata),
        #     ftol=1e-12, loss='soft_l1')

        order = min(self.order, xdata.shape[0] - 1)
        warnings.filterwarnings("ignore")
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                poly = np.poly1d(np.polyfit(
                    xdata, ydata, order, w=1. / (yerr + 1e-15)))
            except np.RankWarning:
                logging.critical(
                    "Cannot fit polynomial with degree %d, npoints %d",
                    order, xdata.shape[0])
                return lambda _: 1, 0

        # XXX plotting
        # with sns.axes_style('whitegrid'):
        #     sns.set_context('paper')
        #     xp = np.linspace(np.min(xdata), np.max(xdata), 1000)[:, None]
        #     plt.figure()
        #     plt.errorbar(xdata, ydata, yerr,
        #                  label='Nearest similarity', marker='s')
        #     plt.plot(xp, poly(xp), '-',
        #              label='Learning function (poly of order {})'.format(order))
        #     # plt.plot(xp, least_squares_mdl(res.x, xp), '-', label='least squares')
        #     plt.xlabel(r'Mutation level')
        #     plt.ylabel(r'Average similarity (not normalised)')
        #     plt.legend(loc='lower left')
        #     plt.savefig(aplot, transparent=True, bbox_inches='tight')
        #     plt.close()

        # poly = partial(model, res.x)
        return poly, 1 - (filter(
            lambda x: x > 0,
            np.array(thresholds)[np.array(samples).argsort()[::-1]]) or [0])[0]

    def distributions(self):
        """Create histograms and relative mutation levels using intra groups."""
        db = self.db
        logging.info("Analysing %s ...", db)
        try:
            max_mut, n_tot = io.get_max_mut(db)
            # if max_mut < 1:
            lin = np.linspace(0, max_mut, min(n_tot / 15., 12))
            # lin = np.linspace(0, max_mut, 10.)
            sets = [(0, 0)] + zip(lin[:-1], lin[1:])
            # sets = [(0, 0)] + [(i - 1, i) for i in range(1, int(max_mut) + 1)]
            if len(sets) == 1:
                # no correction needs to be applied
                return None
            out_muts = [
                intra_donor_distance(
                    db, i, j, quantity=self.quantity, donor=db.split('/')[-1],
                    bins=self.bins, max_seqs=self.max_seqs,
                    min_seqs=self.min_seqs,
                    correction=False, n_tot=n_tot,
                    igsimilarity=self.igsimilarity) for i, j in zip(sets, sets)]
        except StandardError as msg:
            logging.critical(msg)
            out_muts = []

        d = dict()
        for f, m in out_muts:
            d.setdefault(m, []).append(f)
        return d

    def fit(self, records=None):
        learning_function = None
        my_dict = self.distributions()
        learning_function, threshold_naive = self.learn(my_dict)

        self.learning_function = learning_function
        self.threshold_naive = threshold_naive


def make_hist(juncs1, juncs2, filename, lim_mut1, lim_mut2, type_ig='Mem',
              mut=None, donor1='B4', donor2=None, bins=100,
              min_seqs=0, ig1=None, ig2=None, is_intra=True,
              igsimilarity=None, correction=False):
    """Make histogram and main computation of nearest similarities."""
    if os.path.exists(filename + '.npz'):
        logging.critical(filename + '.npz esists.')
        return filename
    if len(juncs1) < min_seqs or len(juncs2) < min_seqs:
        return ''

    import copy
    igsimilarity_learn = copy.deepcopy(igsimilarity)
    # cloning.set_defaults_sim_func(
    #     sim_func_args_2, ig1 if is_intra else ig1 + ig2)
    #
    # sim_func_args_2['correct'] = correction
    # sim_func_args_2['rm_duplicates'] = True
    # if not correction:
    #     sim_func_args_2['tol'] = 1000
    # else:
    #     sim_func_args_2['correction_function'] = correction
    #
    # sim_func = partial(cloning.sim_function, **sim_func_args_2)
    igsimilarity_learn.correct = correction
    igsimilarity_learn.rm_duplicates = True
    if not correction:
        igsimilarity_learn.tol = 1000
    else:
        igsimilarity_learn.correct_by = correction

    sim_func = igsimilarity_learn.pairwise
    logging.info("Computing %s", filename)
    if is_intra:
        # dnearest = parallel_distance.dnearest_inter_padding(
        #     ig1, ig1, sim_func, filt=lambda x: 0 < x, func=max)
        dnearest = parallel_distance.dnearest_intra_padding(
            ig1, sim_func, filt=lambda x: x > 0, func=max)
        # ig1, ig1, sim_func, filt=lambda x: 0 < x < 1, func=max)
    else:
        dnearest = parallel_distance.dnearest_inter_padding(
            ig1, ig2, sim_func, filt=lambda x: 0 < x < 1, func=max)
    if not os.path.exists(filename.split('/')[0]):
        os.makedirs(filename.split('/')[0])
    np.savez(filename, X=dnearest, mut=mut)

    # Plot distance distribution
    plt.figure(figsize=(20, 10))
    plt.hist(dnearest, bins=bins, normed=True)
    plt.title("Distances between " +
              ("{}-{}".format(donor1, donor2) if donor2 else "") +
              " {} {:.3f}-{:.3f}% and {:.3f}-{:.3f}%"
              .format(type_ig, lim_mut1[0], lim_mut1[1], *lim_mut2))
    plt.ylabel('Count')
    # plt.xlim([0, 1])
    plt.xticks(np.linspace(0, 1, 21))
    # plt.xlabel('Ham distance (normalised)')
    plt.savefig(filename + ".png")
    plt.close()
    return filename


def shuffle_ig(igs, juncs, max_seqs):
    if len(juncs) > max_seqs:
        igs, juncs = shuffle(igs, juncs)
        igs = igs[:max_seqs]
        juncs = juncs[:max_seqs]
    return igs, juncs


def intra_donor_distance(db='', lim_mut1=(0, 0), lim_mut2=(0, 0), type_ig='Mem',
                         quantity=.15, donor='B4', bins=100, max_seqs=1000,
                         n_tot=0, min_seqs=100, igsimilarity=None,
                         correction=False):
    """Nearest distances intra donor.

    Subsets of Igs can be selected choosing two ranges of mutations.
    """
    filename = \
        "{0}/dist2nearest_{0}_{1}-{2}_vs_{3}-{4}_{5}bins_norm_{6}maxseqs" \
        .format(donor, lim_mut1[0], lim_mut1[1], lim_mut2[0],
                lim_mut2[1], bins, max_seqs) + \
        ('_correction' if correction else '')
    # mut = min(lim_mut1[0], lim_mut2[0])
    if os.path.exists(filename + '.npz'):
        logging.info("File %s exists.", filename + '.npz')
        # Plot distance distribution
        plt.figure(figsize=(20, 10))
        dnearest = np.load(filename + '.npz')['X']
        plt.hist(dnearest, bins=bins, normed=True)
        plt.title("Similarities for " +
                  ("{}".format(donor)) +
                  " {} {:.3f}-{:.3f}% and {:.3f}-{:.3f}%"
                  .format(type_ig, lim_mut1[0], lim_mut1[1], *lim_mut2))
        plt.ylabel('Count')
        # plt.xlim([0, 1])
        # plt.xticks(np.linspace(0, 1, 21))
        # plt.xlabel('Ham distance (normalised)')
        plt.savefig(filename + ".pdf")
        plt.close()
        return filename, float(np.load(filename + '.npz')['mut'])

    readdb = partial(io.read_db, db, max_records=quantity * n_tot)
    if max(lim_mut1[1], lim_mut2[1]) == 0:
        igs = readdb(filt=(lambda x: x.mut == 0))
        igs1, juncs1 = remove_duplicate_junctions(igs)
        if len(igs1) < 2:
            return '', 0
        igs1, juncs1 = shuffle_ig(igs1, juncs1, max_seqs)
        igs2 = igs1
        juncs2 = juncs1
        mut = 0
    elif (lim_mut1[0] == lim_mut2[0] and lim_mut1[1] == lim_mut2[1]):
        igs = readdb(filt=(lambda x: lim_mut1[0] < x.mut <= lim_mut1[1]))
        igs1, juncs1 = remove_duplicate_junctions(igs)
        if len(igs1) < 2:
            return '', 0
        igs1, juncs1 = shuffle_ig(igs1, juncs1, max_seqs)
        igs2 = igs1
        juncs2 = juncs1
        mut = np.mean(list(chain((x.mut for x in igs1),
                                 (x.mut for x in igs2))))
    else:
        igs = readdb(filt=(lambda x: lim_mut1[0] < x.mut <= lim_mut1[1]))
        igs1, juncs1 = remove_duplicate_junctions(igs)
        if len(igs1) < 2:
            return '', 0
        igs = readdb(filt=(lambda x: lim_mut2[0] < x.mut <= lim_mut2[1]))
        igs2, juncs2 = remove_duplicate_junctions(igs)
        if len(igs2) < 2:
            return '', 0
        if not len(juncs1) or not len(juncs2):
            return '', 0
        igs1, juncs1 = shuffle_ig(igs1, juncs1, max_seqs)
        igs2, juncs2 = shuffle_ig(igs2, juncs2, max_seqs)
        mut = np.mean(list(chain((x.mut for x in igs1),
                                 (x.mut for x in igs2))))
    # logging.info("Computing similarity ")
    return make_hist(
        juncs1, juncs2, filename, lim_mut1, lim_mut2, type_ig, mut,
        donor, None, bins, min_seqs, ig1=igs1, ig2=igs2,
        igsimilarity=igsimilarity, correction=correction), mut
