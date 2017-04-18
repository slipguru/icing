"""Learning function sklearn-like."""
from __future__ import division, print_function

import copy
import logging
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import six
import warnings

from functools import partial
from itertools import chain
# from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator
from sklearn.utils import shuffle

from icing.core import parallel_distance
from icing.externals.DbCore import IgRecord

from icing.core.learning_function import remove_duplicate_junctions
from icing.core.learning_function import mean_confidence_interval
from icing.core.learning_function import _gaussian_fit


class LearningFunction(BaseEstimator):

    def __init__(self, database, quantity=1, igsimilarity=None, order=3,
                 root='', min_seqs=10, max_seqs=1000, bins=50, aplot=None):
        self.database = database
        self.quantity = quantity
        self.igsimilarity = igsimilarity
        self.order = order
        self.root = root
        self.min_seqs = min_seqs
        self.max_seqs = max_seqs
        self.bins = bins
        self.aplot = aplot

    def learn(self, my_dict, aplot=None):
        if my_dict is None:
            logging.critical("Cannot learn function with empty dict")
            return lambda _: 1, 0
        d_dict = dict()
        samples, thresholds = [], []
        for k, v in six.iteritems(my_dict):
            for o in (_ for _ in v if _):
                dnearest = np.array(np.load("{}.npz".format(o))['X']).reshape(
                    -1, 1)
                var = np.var(dnearest)
                if var == 0:
                    continue
                med = np.median(dnearest)
                mean, _, _, h = mean_confidence_interval(dnearest)
                samples.append(dnearest.shape[0])
                d_dict.setdefault(o.split('/')[0], dict()).setdefault(k, [med, h])

                # for the threshold, fit a gaussian (unused for AP)
                thresholds.append(_gaussian_fit(dnearest))
        if len(d_dict) < 1:
            logging.critical("dictionary is empty")
            return lambda _: 1, 0
        for k, v in six.iteritems(d_dict):  # there is only one
            xdata = np.array(sorted(v))
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

        if self.aplot is not None:
            plot_learning_function(xdata, ydata, yerr, order, self.aplot, poly)

        # poly = partial(model, res.x)
        return poly, 1 - (filter(
            lambda x: x > 0,
            np.array(thresholds)[np.array(samples).argsort()[::-1]]) or [0])[0]

    def intra_donor_distance(self, records=None, lim_mut1=(0, 0), lim_mut2=(0, 0)):
        """Nearest distances intra donor.

        Subsets of Igs can be selected choosing two ranges of mutations.
        """
        filename = \
            "{0}/dist2nearest_{0}_{1}-{2}_vs_{3}-{4}_{5}bins_norm_{6}maxseqs" \
            .format(self.donor, lim_mut1[0], lim_mut1[1], lim_mut2[0],
                    lim_mut2[1], self.bins, self.max_seqs) + \
            ('_correction' if self.correction else '')
        # mut = min(lim_mut1[0], lim_mut2[0])
        if os.path.exists(filename + '.npz'):
            logging.critical("File %s exists.", filename + '.npz')
            dnearest = np.load(filename + '.npz')['X']
            title = "Similarities for {:.3f}-{:.3f}% and {:.3f}-{:.3f}%" \
                    .format(lim_mut1[0], lim_mut1[1], *lim_mut2)
            plot_hist(dnearest, self.bins, title, filename)
            return filename, float(np.load(filename + '.npz')['mut'])

        if records is not None and isinstance(records, pd.DataFrame):
            if lim_mut1[0] == lim_mut2[0] and lim_mut1[1] == lim_mut2[1]:
                igs = records[np.logical_and(records['MUT'] >= lim_mut1[0],
                                             records['MUT'] <= lim_mut1[1])]
                mut = np.mean(igs['MUT'])
            else:
                raise NotImplementedError("not yet")

            return self.mutation_histogram(igs, mut, filename), mut
        else:
            # load from file
            igs1, igs2, juncs1, juncs2, mut = read_db_file(
                self, lim_mut1, lim_mut2)

            return self.make_hist(
                juncs1, juncs2, filename, lim_mut1, lim_mut2, mut,
                self.donor, None, ig1=igs1, ig2=igs2), mut

    def distributions(self, records=None):
        logging.info("Analysing %s ...", self.database)
        try:
            if records is not None and isinstance(records, pd.DataFrame):
                max_mut = np.max(records['MUT'])
                self.n_samples = records.shape[0]
            else:
                # load from file
                max_mut, self.n_samples = io.get_max_mut(self.database)

            lin = np.linspace(0, max_mut, min(self.n_samples / 15., 12))
            sets = [(0, 0)] + zip(lin[:-1], lin[1:])
            if len(sets) == 1:
                # no correction needs to be applied
                return None
            out_muts = [self.intra_donor_distance(
                records, i, j) for i, j in zip(sets, sets)]
        except StandardError as msg:
            logging.critical(msg)
            out_muts = []

        my_dict = dict()
        for f, m in out_muts:
            my_dict.setdefault(m, []).append(f)
        return my_dict

    def fit(self, records=None, correction=False):
        """Create histograms and mutation levels using intra groups.

        Parameters
        ----------
        records : None or pd.DataFrame
            If records is an instance of a dataframe, use it instead of loading
            data from disk.
        """
        self.correction = correction
        self.donor = self.database.split('/')[-1]

        my_dict = self.distributions(records)
        learning_function, threshold_naive = self.learn(my_dict)

        self.learning_function = learning_function
        self.threshold_naive = threshold_naive

        return self

    def make_hist(self, juncs1, juncs2, filename, lim_mut1, lim_mut2,
                  mut=None, donor1='B4', donor2=None, ig1=None, ig2=None,
                  is_intra=True):
        """Make histogram and main computation of nearest similarities."""
        if os.path.exists(filename + '.npz'):
            logging.critical(filename + '.npz esists.')
            return filename
        if len(juncs1) < self.min_seqs or len(juncs2) < self.min_seqs:
            return ''

        igsimilarity_learn = copy.deepcopy(self.igsimilarity)
        igsimilarity_learn.correct = self.correction
        igsimilarity_learn.rm_duplicates = True
        if not self.correction:
            igsimilarity_learn.tol = 1000
        else:
            igsimilarity_learn.correct_by = self.correction

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
        title = "Similarities for {:.3f}-{:.3f}% and {:.3f}-{:.3f}%" \
                .format(lim_mut1[0], lim_mut1[1], *lim_mut2)
        plot_hist(dnearest, self.bins, title, filename)
        return filename

    def mutation_histogram(self, records, mut, filename):
        """Records is a pd.Dataframe."""
        if os.path.exists(filename + '.npz'):
            logging.critical(filename + '.npz esists.')
            return filename
        if records.shape[0] < self.min_seqs:
            return ''

        igs = [IgRecord(x.to_dict()) for _, x in records.iterrows()]
        igsimilarity_learn = copy.deepcopy(self.igsimilarity)
        igsimilarity_learn.correct = self.correction
        igsimilarity_learn.rm_duplicates = True
        if not self.correction:
            igsimilarity_learn.tol = 1000
        else:
            igsimilarity_learn.correct_by = self.correction

        sim_func = igsimilarity_learn.pairwise
        logging.info("Computing %s", filename)
        dnearest = parallel_distance.dnearest_intra_padding(
            igs, sim_func, filt=lambda x: x > 0, func=max)

        if not os.path.exists(filename.split('/')[0]):
            os.makedirs(filename.split('/')[0])
        np.savez(filename, X=dnearest, mut=mut)

        # Plot distance distribution
        title = "Similarities for {:.3f}-{:.3f}%" \
                .format(np.min(records['MUT']), np.max(records['MUT']))
        plot_hist(dnearest, self.bins, title, filename)
        return filename



def shuffle_ig(igs, juncs, max_seqs):
    if len(juncs) > max_seqs:
        igs, juncs = shuffle(igs, juncs)
        igs = igs[:max_seqs]
        juncs = juncs[:max_seqs]
    return igs, juncs


def plot_hist(dnearest, bins, title, filename):
    # Plot distance distribution
    plt.figure(figsize=(20, 10))
    plt.hist(dnearest, bins=bins, normed=True)
    plt.title(title)
    plt.ylabel('Count')
    # plt.xlim([0, 1])
    # plt.xticks(np.linspace(0, 1, 21))
    # plt.xlabel('Ham distance (normalised)')
    plt.savefig(filename + ".pdf")
    plt.close()


def plot_learning_function(xdata, ydata, yerr, order, aplot, poly):
    with sns.axes_style('whitegrid'):
        sns.set_context('paper')
        xp = np.linspace(np.min(xdata), np.max(xdata), 1000)[:, None]
        plt.figure()
        plt.errorbar(xdata, ydata, yerr,
                     label='Nearest similarity', marker='s')
        plt.plot(xp, poly(xp), '-',
                 label='Learning function (poly of order {})'.format(order))
        # plt.plot(xp, least_squares_mdl(res.x, xp), '-', label='least squares')
        plt.xlabel(r'Mutation level')
        plt.ylabel(r'Average similarity (not normalised)')
        plt.legend(loc='lower left')
        plt.savefig(aplot, transparent=True, bbox_inches='tight')
        plt.close()


def read_db_file(estimator, lim_mut1, lim_mut2):
    """Read database from file."""
    readdb = partial(io.read_db, estimator.database,
                     max_records=estimator.quantity * estimator.n_samples)
    if max(lim_mut1[1], lim_mut2[1]) == 0:
        igs = readdb(filt=(lambda x: x.mut == 0))
        igs1, juncs1 = remove_duplicate_junctions(igs)
        if len(igs1) < 2:
            return '', 0
        igs1, juncs1 = shuffle_ig(igs1, juncs1, estimator.max_seqs)
        igs2 = igs1
        juncs2 = juncs1
        mut = 0
    elif (lim_mut1[0] == lim_mut2[0] and lim_mut1[1] == lim_mut2[1]):
        igs = readdb(filt=(lambda x: lim_mut1[0] < x.mut <= lim_mut1[1]))
        igs1, juncs1 = remove_duplicate_junctions(igs)
        if len(igs1) < 2:
            return '', 0
        igs1, juncs1 = shuffle_ig(igs1, juncs1, estimator.max_seqs)
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
        igs1, juncs1 = shuffle_ig(igs1, juncs1, estimator.max_seqs)
        igs2, juncs2 = shuffle_ig(igs2, juncs2, estimator.max_seqs)
        mut = np.mean(list(chain((x.mut for x in igs1),
                                 (x.mut for x in igs2))))
    return igs1, igs2, juncs1, juncs2, mut
