#!/usr/bin/env python
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_context('poster')
import joblib as jl

from scipy.optimize import curve_fit
from sklearn import mixture

from changeo import newDefineClones as cl
from ignet import parallel_distance
from ignet.core import distances
from ignet.models import model
from ignet.utils import io

ham_model = model.model_matrix('ham')
sym = cl.default_sym


def remove_duplicate_junctions(igs_list):
    igs, juncs = [], set()
    for ig in igs_list:
        junc = re.sub('[\.-]', 'N', str(ig.junction))
        if junc not in juncs:
            igs.append(ig)
            juncs.add(junc)
    return igs, juncs


def calcDist(el1, el2):
    # consider ham model
    return distances.string_distance(el1, el2, ham_model)


def make_hist(juncs1, juncs2, fn, lim_mut1, lim_mut2, type_ig='Mem',
              donor1='B4', donor2=None, bins=100, max_seqs=1000):
    if os.path.exists(fn + '.npy'):
        return fn
    # sample if length is exceeded (for computational costs)
    min_seqs = 100
    if len(juncs1) < min_seqs or len(juncs2) < min_seqs:
        return ''
    if len(juncs1) > max_seqs:
        juncs1 = np.random.choice(juncs1, max_seqs, replace=False)
    if len(juncs2) > max_seqs:
        juncs2 = np.random.choice(juncs2, max_seqs, replace=False)

    print("Computing {}".format(fn))
    dist2nearest = parallel_distance.dnearest_inter_padding(juncs1, juncs2, calcDist)
    if not os.path.exists(fn.split('/')[0]):
        os.makedirs(fn.split('/')[0])
    np.save(fn, dist2nearest)
    # dist2nearest = np.array([np.min(r[r>0]) for r in X])
    f = plt.figure(figsize=(20, 10))
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
    except:
        pass
    plt.ylabel('Count')
    plt.xticks(np.linspace(0, 1, 21))
    plt.xlabel('Ham distance (normalised)')
    plt.savefig(fn + ".png")
    plt.close()
    del dist2nearest
    return fn


def intra_donor_distance(f='', lim_mut1=(0, 0), lim_mut2=(0, 0), type_ig='Mem',
                         donor='B4', bins=100, max_seqs=1000):
    """Nearest distances intra donor.

    Subsets of Igs can be selected choosing two ranges of mutations.
    """
    fn = "{0}/dist2nearest_{0}_{2}-{3}_vs_{4}-{5}_{6}bins_norm_{7}maxseqs" \
         .format(donor, type_ig.lower(), lim_mut1[0], lim_mut1[1], lim_mut2[0],
                 lim_mut2[1], bins, max_seqs)
    if os.path.exists(fn + '.npy'):
        return fn, max(lim_mut1[1], lim_mut2[1])

    if max(lim_mut1[1], lim_mut2[1]) == 0:
        igs = io.read_db(f, filt=(lambda x: x.mut == 0))
        igs, juncs1 = remove_duplicate_junctions(igs)
        juncs2 = juncs1
    else:
        igs = io.read_db(f, filt=(lambda x: lim_mut1[0] <= x.mut < lim_mut1[1]))
        _, juncs1 = remove_duplicate_junctions(igs)
        igs = io.read_db(f, filt=(lambda x: lim_mut2[0] <= x.mut < lim_mut2[1]))
        _, juncs2 = remove_duplicate_junctions(igs)

    return make_hist(juncs1, juncs2, fn, lim_mut1, lim_mut2, type_ig, donor,
                     None, bins, max_seqs), max(lim_mut1[1], lim_mut2[1])


def inter_donor_distance(f1='', f2='', lim_mut1=(0, 0), lim_mut2=(0, 0),
                         type_ig='Mem', donor1='B4', donor2='B5', bins=100,
                         max_seqs=1000):
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
        igs = io.read_db(f1, filt=(lambda x: lim_mut1[0] <= x.mut < lim_mut1[1]))
        _, juncs1 = remove_duplicate_junctions(igs)
        igs = io.read_db(f2, filt=(lambda x: lim_mut2[0] <= x.mut < lim_mut2[1]))
        _, juncs2 = remove_duplicate_junctions(igs)

    return make_hist(juncs1, juncs2, fn, lim_mut1, lim_mut2, type_ig, donor1,
                     donor2, bins, max_seqs), max(lim_mut1[1], lim_mut2[1])


def job(f):
    print("Analysing {} ...".format(f))
    out_fles, mut_lvls = [], []
    try:
        max_mut = int(io.get_max_mut(f))
        sets = [(0, 0)] + [(i - 1, i) for i in range(1, 24)] + \
               [(23, max_mut + 1)]
        combinations = [x for x in zip(sets, sets)]
        for i, j in combinations:
            o, mut = intra_donor_distance(f, i, j, donor=f.split('/')[-1],
                                          bins=50, max_seqs=4000)
            out_fles.append(o)
            mut_lvls.append(mut)
    except Exception as e:
        print(e)
    return out_fles, mut_lvls


def all_intra_mut():
    """Create histograms and relative mutation levels."""
    mypath = '/home/fede/Dropbox/projects/davide/new_seqs/new_samples'
    inputs = [os.path.join(mypath, f) for f in os.listdir(mypath)
              if os.path.isfile(os.path.join(mypath, f)) and f[0] != '.'] + \
             ['/home/fede/Dropbox/projects/davide/new_seqs/B4_db-pass.tab_CON-FUN-N.tab',
              '/home/fede/Dropbox/projects/davide/new_seqs/B5_db-pass.tab_CON-FUN-N.tab']

    out_files, mut_levels = zip(*jl.Parallel(n_jobs=-1)
                                (jl.delayed(job)(f) for f in inputs))
    out_files = [item for sublist in out_files for item in sublist]
    mut_levels = [item for sublist in mut_levels for item in sublist]
    d = dict()
    for i, f in enumerate(out_files):
        d.setdefault(mut_levels[i], []).append(f)
    return out_files, mut_levels, d


def create_alpha_plot(out_files, mut_levels, __my_dict__):
    mu_naive = []
    mu_mem = []
    list_naive_npy = []

    mus = []
    sigmas = []
    muts = []
    mut_means = dict()
    mut_sigmas = dict()
    d_dict = dict()
    for k, v in __my_dict__.iteritems():
        for o in v:
            print(o)
            if not o: continue
            dist2nearest = np.array(np.load("{}.npy".format(o))).reshape(-1, 1)
            if dist2nearest.shape[0] < 2:
                print("Cannot fit a Gaussian with two distances."); continue

            dist2nearest_2 = -(np.array(sorted(dist2nearest)).reshape(-1, 1))
            dist2nearest = np.array(list(dist2nearest_2) +
                                    list(dist2nearest)).reshape(-1, 1)
            gmm = mixture.GMM(n_components=3)
            gmm.fit(dist2nearest)

            mean = np.max(gmm.means_)
            sigma = gmm.covars_[np.argmax(gmm.means_)]
            mut_means.setdefault(k, []).append(mean)
            mut_sigmas.setdefault(k, []).append(sigma)

            # mut = mut_levels[i]
            if '0-0' in o:
                mu_naive.append(mean)
                list_naive_npy.append(o + '.npy')
            mu_mem.append(mean)
            # mut_level_mem.append(mut)
            mus.append(mean)
            sigmas.append(sigma)
            muts.append(k)
            d_dict.setdefault(o.split('/')[0],dict()).setdefault(k, mean)

    norm_dict = dict()
    for k, v in d_dict.iteritems():
        mu_mut_0 = float(v.get(0, -1))
        if mu_mut_0 > -1:
            for l, z in v.iteritems():
                z = mu_mut_0 / z
                norm_dict.setdefault(l,[]).append(z)


    x, y, e = [], [], []
    for k, v in norm_dict.iteritems():
        x.append(k)
        y.append(np.mean(v))
        e.append(np.var(v))

    x_s, y_s, e_s = [], [], []
    for k, v in mut_sigmas.iteritems():
        x_s.append(k)
        y_s.append(np.mean(v))
        e_s.append(np.var(v))

    # mean_mu_naive = np.mean(mu_naive)
    if x[0] != 0:
        print(x[0] + " should be 0. Normalising factor is not given by mu naive")
    mean_mu_naive = float(y[0])
    x, y = np.array(x), np.array(y)
    p1 = np.poly1d(np.polyfit(x, y, 1))
    p2 = np.poly1d(np.polyfit(x, y, 2))
    p3 = np.poly1d(np.polyfit(x, y, 3))
    p4 = np.poly1d(np.polyfit(x, y, 4))
    def func(x, a, c, d):
        return a*np.exp(-c*x)+d
    # return x, y
    popt, pcov = curve_fit(func, x, y, p0=(1, 1e-1, 1))

    np.save("poly1d_1",p1)
    np.save("poly1d_2",p2)
    np.save("poly1d_3",p3)
    np.save("poly1d_4",p4)
    np.save("negexp_pars",popt)

    xp = np.linspace(0, 50, 1000)
    # plt.plot(x, y, linestyle=' ', marker='o', label='data')
    with sns.axes_style('whitegrid'):
        plt.figure()
        plt.errorbar(x, y, e, label='data')
        plt.plot(xp, p1(xp), '-', label='order 1')
        plt.plot(xp, p2(xp), '-', label='order 2')
        # plt.plot(xp, p3(xp), '-', label='order 3')
        # plt.plot(xp, p4(xp), '-', label='order 4')
        plt.plot(xp, func(xp, *popt), '-', label=r'$\alpha$ (neg exp)', lw=2.5)
        # plt.plot(xp, np.exp(-xp / 55.), '-', label='neg exp')
        # plt.title(r'Curve fitting to estimate $\alpha$ function for mutations.')# $\mu_{{NAIVE}}={:.3f}$'.format(mean_mu_naive))
        plt.ylabel(r'$\mu$ Naive / $\mu$ Mem')
        plt.xlabel(r'Igs mutation level')
        plt.ylim([0., 1.])
        plt.xlim([0, 50])
        plt.legend(loc='lower left')
        plt.savefig("alpha_mut_plot_poster_notitle_dpi.pdf", transparent=True)
        plt.close()

    plt.plot(x, y, linestyle=' ', marker='o', label='data')
    plt.title(r'$\mu$ plot based on mutation levels')
    plt.ylabel(r'$\mu$')
    plt.xlabel(r'Igs mutation level')
    # plt.ylim([0, 0.6])
    plt.xlim([0, 30])
    plt.legend()
    plt.savefig("mu_on_mutation_levels.png")
    plt.close()

    plt.errorbar(x_s, y_s, e_s, linestyle=' ', marker='o', label='sigmas')
    # plt.plot(muts, sigmas, linestyle=' ', marker='o', label='data')
    plt.title(r'$\sigma$ plot based on mutation levels')
    plt.ylabel(r'$\sigma$')
    plt.xlabel(r'Igs mutation level')
    plt.ylim([0, 0.1])
    plt.xlim([0, 30])
    plt.legend()
    plt.savefig("sigma_on_mutation_levels.png")
    plt.close()
    return list_naive_npy


if __name__ == '__main__':
    out_files, mut_levels, __my_dict__ = all_intra_mut()
    create_alpha_plot(out_files, mut_levels, __my_dict__)

    # ''' Specify the filename of donors' Igs.
    # Only naive without mutations are selected. For mem Igs, subsets or 1%% mutation
    # range are selected.'''
    #
    # filename_b4 = '../new_seqs/B4_db-pass.tab_CON-FUN-N.tab'
    # filename_b5 = '../new_seqs/B5_db-pass.tab_CON-FUN-N.tab'
    #
    # ## donor B4
    # igs_b4 = [i for i in DbCore.readDbFile(filename_b4)]
    #
    # # 1. Naive
    # naive_b4 = [i for i in igs_b4 if (i.subset == 'N' and i.mut == 0)]
    # naive_filtered_b4, _ = remove_duplicate_junctions(naive_b4)
    # intra_donor_distance(naive_filtered_b4, type_ig='Naive', donor='B4', bins=100)
    #
    # # 2. Mem
    # mem_b4 = [i for i in igs_b4 if (i.subset in ['MemA', 'MemG'])]
    # mem_filtered_b4, _ = remove_duplicate_junctions(mem_b4)
    #
    # # sets1 = [(x, i+1) for i in range(int(max([m.mut for m in mem_filtered if m.mut < 24])))] + [(24,29)]
    # max_mut = int(max([m.mut for m in mem_filtered_b4]))
    # sets1 = [(i-1, i) for i in range(1, 24)] + [(24,max_mut+1)]
    # sets2 = [(0, i) for i in range(1, 24)] + [(0,max_mut+1)]
    # combinations = [x for x in zip(sets1, sets)][1:]
    # for i, j in combinations:
    #     intra_donor_distance(mem_filtered_b4, i, j,
    #                          type_ig='Mem', donor='B4', bins=100)
    #
    # ## donor B4 vs B5
    # igs_b5 = [i for i in DbCore.readDbFile(filename_b5)]
    #
    # # 1. Naive
    # naive_b5 = [i for i in igs_b5 if (i.subset == 'N' and i.mut == 0)]
    # naive_filtered_b5, _ = remove_duplicate_junctions(naive_b5)
    #
    # inter_donor_distance(naive_filtered_b4, naive_filtered_b5,
    #                      type_ig='Naive', donor1='B4', donor2='B5', bins=100)
    #
    # # 2. Mem
    # mem_b5 = [i for i in igs_b5 if (i.subset in ['MemA','MemG'])]
    # mem_filtered_b5, _ = remove_duplicate_junctions(mem_b5)
    #
    # # Mem of B4 and B5 with the same mut level
    # max_mut = max(int(max([m.mut for m in mem_filtered_b4])),
    #               int(max([m.mut for m in mem_filtered_b5])))
    # sets1 = [(i-1, i) for i in range(1, 24)] + [(24, max_mut+1)]
    # combinations = [x for x in itertools.product(sets1, sets1) if x[0][0] == x[1][0]]
    # for i, j in combinations:
    #     inter_donor_distance(mem_filtered_b4, mem_filtered_b5, i, j,
    #                          type_ig='Mem', donor1='B4', donor2='B5', bins=100)
