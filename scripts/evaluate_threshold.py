#!/usr/bin/env python
import os
import numpy as np
import joblib as jl
import matplotlib.pyplot as plt

from icing.core import distances
from icing.models import model
from icing.utils import extra, io
from icing.parallel_distance import dnearest_inter_padding

ham_model = model.model_matrix('ham')
ham_model['-'] = .75


def show_fit(filename, ax=None):
    """Util function. Show a gaussian fit on nearest distances.

    Usage example:
        np.mean([show_fit(f) for f in list_naive_npy])
    """
    from sklearn.mixture import GMM
    X = np.load("{}".format(filename))
    dist2nearest = np.array(X).reshape(-1, 1)
    if dist2nearest.shape[0] < 2:
        print("Cannot fit a Gaussian with two distances.")
        return

    dist2nearest_2 = -(np.array(sorted(dist2nearest)).reshape(-1, 1))
    dist2nearest = np.array(list(dist2nearest_2) +
                            list(dist2nearest)).reshape(-1, 1)
    gmm = GMM(n_components=3)
    gmm.fit(dist2nearest)

    plt.hist(dist2nearest, bins=50, normed=True)
    linspace = np.linspace(-1, 1, 1000)[:, np.newaxis]
    plt.plot(linspace, np.exp(gmm.score_samples(linspace)[0]), 'r')

    lin = np.linspace(0, 1, 10000)[:, np.newaxis]
    pred = gmm.predict(linspace)
    argmax = np.argmax(gmm.means_)
    idx = np.min(np.where(pred == argmax)[0])
    plt.axvline(x=lin[idx], linestyle='--', color='r')
    plt.show()
    return lin[idx]  # threshold


def distance_function(el1, el2):
    return distances.string_distance(el1, el2, ham_model)


def load_junctions(f, max_mut, njuncs, allowed_subsets=['n', 'naive']):
    igs = list(io.read_db(f, filt=(lambda x: x.mut <= max_mut and
                                   x.subset.lower() in allowed_subsets)))
    if len(igs) > njuncs:
        igs = np.random.choice(igs, njuncs, replace=False)
    junc = map(lambda x: extra.junction_re(x.junction), igs)
    return junc, np.max([x.mut for x in igs])


def _neg_exp(x, a, c, d):
    return a * np.exp(-c * x) + d


def evaluate_method(max_seqs=1000, max_mut=0, threshold=0.0536):
    """Percentual of records inside the threshold."""
    mypath = '/home/fede/Dropbox/projects/davide/new_seqs/new_samples'
    inputs = [os.path.join(mypath, f) for f in os.listdir(mypath)
              if os.path.isfile(os.path.join(mypath, f)) and f[0] != '.'] + \
             ['/home/fede/Dropbox/projects/davide/new_seqs/B4_db-pass.tab_CON-FUN-N.tab',
              '/home/fede/Dropbox/projects/davide/new_seqs/B5_db-pass.tab_CON-FUN-N.tab']
    res = []
    for i, f in enumerate(inputs):
        l1 = set([f])
        l2 = set(inputs) - l1
        print("Analysing", l1, "vs all")
        fn = "dist2nearest_{0}_vs_all_{1}maxseqs_N".format(f.split('/')[-1], max_seqs)
        if os.path.exists(fn + '.npy') and os.path.exists(fn + 'mut.npy'):
            dist2nearest = np.load(fn + '.npy')
            max_mut = np.load(fn + 'mut.npy')
        else:
            juncs1, muts1 = load_junctions(f, max_mut, max_seqs)
            juncs2, muts2 = zip(*jl.Parallel(n_jobs=-1)
                                (jl.delayed(load_junctions)
                                 (g, max_mut, int(max_seqs / len(l2)))
                                 for g in l2))
            juncs2 = extra.flatten(list(juncs2))
            if len(juncs2) > max_seqs:
                juncs2 = np.random.choice(juncs2, max_seqs, replace=False)

            print("Computing {}".format(f))

            dist2nearest = np.array(dnearest_inter_padding(juncs1, juncs2, distance_function))
            np.save(fn, dist2nearest)
            max_mut = max(list(muts2) + [muts1])
            np.save(fn + 'mut', max_mut)
            print max_mut

        popt = np.load('negexp_pars.npy')
        _threshold = threshold / _neg_exp(max_mut, *popt)
        # threshold = 0.053447011367803443 / np.exp(-max_mut / 35.)
        # if i == 1:
        # print dist2nearest
        # print dist2nearest < threshold
        # print dist2nearest[dist2nearest < threshold]
        perc = len(dist2nearest[dist2nearest < _threshold]) * 100. / len(dist2nearest)
        res.append(perc)
        print("Percentual of records inside the threshold: {}".format(perc))

        fig = plt.figure(figsize=(20, 10))
        plt.hist(dist2nearest, bins=50, normed=True)
        plt.title("Distances between {} vs all".format(str(f.split('/')[-1])))
        plt.ylabel('Count')
        plt.xticks(np.linspace(0,1,21))
        plt.xlabel('Ham distance (normalised)')
        plt.axvline(x=threshold, linestyle='--', color='r')
        plt.savefig(fn + ".png")
        plt.close()

    return res  # percentuale di ig a sx del threshold


if __name__ == '__main__':
    evaluate_method()
