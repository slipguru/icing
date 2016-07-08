import newDefineClones as cl
import DbCore
import os, re, sys
import numpy as np
import matplotlib.pyplot as plt
import scipy
import seaborn
import itertools

from sklearn import mixture

from ignet import parallel_distance

ham_model = DbCore.getModelMatrix('ham')
sym = cl.default_sym

def remove_duplicate_junctions(igs_list):
    igs, juncs = [], []
    for ig in igs_list:
        junc = re.sub('[\.-]','N', str(ig.junction))
        if not junc in juncs:
            igs.append(ig)
            juncs.append(junc)
    return igs, juncs

def calcDist(el1, el2, mut=[]):
    #consider ham model
    return DbCore.calcDistances([el1, el2], 1, ham_model, 'min', sym, mut)[0,1]


def intra_donor_distance(igs, lim_mut1=None, lim_mut2=None, type_ig='Mem', donor='B4', bins=100):
    ''' Nearest distances intra donor.
    Subsets of Igs can be selected choosing two ranges of mutations. '''

    if lim_mut1 is None and lim_mut2 is None:
        juncs1 = juncs2 = [re.sub('[\.-]','N', str(ig.junction)) for ig in igs]
        lim_mut1 = lim_mut2 = (0, 0)
    else:
        juncs1 = [re.sub('[\.-]','N', str(ig.junction)) for ig in igs if lim_mut1[0] <= ig.mut < lim_mut1[1]]
        juncs2 = [re.sub('[\.-]','N', str(ig.junction)) for ig in igs if lim_mut2[0] <= ig.mut < lim_mut2[1]]

    path = donor
    fn = path+"/dist2nearest_"+donor+'_'+type_ig.lower()+'_'+str(lim_mut1[0])+'-'+str(lim_mut1[1])+'_vs_'+str(lim_mut2[0])+'-'+str(lim_mut2[1])+"_"+str(bins)+"bins_norm"
    return make_hist(juncs1, juncs2, fn, lim_mut1, lim_mut2, type_ig, donor, None, bins), max(lim_mut1[1], lim_mut2[1])

def inter_donor_distance(igs1, igs2, lim_mut1=None, lim_mut2=None, type_ig='Mem', donor1='B4', donor2='B5', bins=100):
    ''' Nearest distances inter donors. Igs involved can be selected by choosing
     two possibly different ranges of mutations. '''

    if lim_mut1 is None and lim_mut2 is None:
        juncs1 = [re.sub('[\.-]','N', str(ig.junction)) for ig in igs1]
        juncs2 = [re.sub('[\.-]','N', str(ig.junction)) for ig in igs2]
        lim_mut1 = lim_mut2 = (0, 0)
    else:
        juncs1 = [re.sub('[\.-]','N', str(ig.junction)) for ig in igs1 if lim_mut1[0] <= ig.mut < lim_mut1[1]]
        juncs2 = [re.sub('[\.-]','N', str(ig.junction)) for ig in igs2 if lim_mut2[0] <= ig.mut < lim_mut2[1]]

    path = '_'.join([donor1, donor2])
    fn = path+"/dist2nearest_"+donor1+'_'+donor2+'_'+type_ig.lower()+'_'+str(lim_mut1[0])+'-'+str(lim_mut1[1])+'_vs_'+str(lim_mut2[0])+'-'+str(lim_mut2[1])+"_"+bins+"bins_norm"
    return make_hist(juncs1, juncs2, fn, lim_mut1, lim_mut2, type_ig, donor1, donor2, bins)

def make_hist(juncs1, juncs2, fn, lim_mut1, lim_mut2, type_ig='Mem', donor1='B4', donor2=None, bins=100):
    if os.path.exists(fn+'.npy'):
        return fn
    # sample if length is exceeded (for computational costs)
    if len(juncs1) > 4000:
        juncs1 = np.random.choice(juncs1, 4000, replace=False)
    if len(juncs2) > 4000:
        juncs2 = np.random.choice(juncs2, 4000, replace=False)
    print("Computing {}".format(fn))
    dist2nearest = parallel_distance.dist2nearest_dual(juncs1, juncs2, calcDist)
    print fn
    if not os.path.exists(fn.split('/')[0]):
        os.makedirs(fn.split('/')[0])
    np.save(fn, dist2nearest)
    # dist2nearest = np.array([np.min(r[r>0]) for r in X])
    f = plt.figure(figsize=(20,10))
    plt.hist(dist2nearest, bins=bins)
    try:
        if not donor2:
            plt.title("Distances between "+type_ig+" "+str(lim_mut1[0])+"-"+str(lim_mut1[1])+"% and "+str(lim_mut2[0])+"-"+str(lim_mut2[1])+"%");
        else:
            plt.title("Distances between "+donor1+'-'+donor2+' '+type_ig+" "+str(lim_mut1[0])+"-"+str(lim_mut1[1])+"% and "+str(lim_mut2[0])+"-"+str(lim_mut2[1])+"%");
    except:
        pass
    plt.ylabel('Count')
    plt.xticks(np.linspace(0,1,21))
    plt.xlabel('Ham distance (normalised)')
    plt.savefig(fn+".png")
    plt.close()
    del dist2nearest
    return fn

def all_intra_mut():
    mypath = '/home/fede/Dropbox/projects/ig_davide/new_seqs/new_samples'
    input_files = [os.path.join(mypath, f) for f in os.listdir(mypath)
                   if os.path.isfile(os.path.join(mypath, f)) and f[0] != '.']
    input_files.append('/home/fede/Dropbox/projects/ig_davide/new_seqs/B4_db-pass.tab_CON-FUN-N.tab')
    input_files.append('/home/fede/Dropbox/projects/ig_davide/new_seqs/B5_db-pass.tab_CON-FUN-N.tab')
    print input_files
    out_files, mut_levels = [], []
    for f in input_files:
        ## donor B4
        igs = [i for i in DbCore.readDbFile(f)]

        # 1. Naive
        naive = [i for i in igs if (i.subset.lower() == 'n' and i.mut == 0)]
        naive_filtered, _ = remove_duplicate_junctions(naive)

        if naive_filtered:
            o, mut = intra_donor_distance(naive_filtered, type_ig='Naive', donor=f.split('/')[-1], bins=100)
            out_files.append(o)
            mut_levels.append(mut)

        # 2. Mem
        mem = [i for i in igs if (i.subset.lower()[3] == 'mem')]
        mem_filtered, _ = remove_duplicate_junctions(mem)

        if mem_filtered:
            # sets1 = [(x, i+1) for i in range(int(max([m.mut for m in mem_filtered if m.mut < 24])))] + [(24,29)]
            max_mut = int(max([m.mut for m in mem_filtered]))
            sets = [(i-1, i) for i in range(1, 24)] + [(23,max_mut+1)]
            combinations = [x for x in zip(sets, sets)]
            for i, j in combinations:
                o, mut = intra_donor_distance(mem_filtered, i, j,
                                     type_ig='Mem', donor=f.split('/')[-1], bins=100)
                out_files.append(o)
                mut_levels.append(mut)

    return out_files, mut_levels


def create_alpha_plot():
    out_files, mut_levels = all_intra_mut()
    mu_naive = []
    mu_mem = []
    mut_level_mem = []

    mus = []
    sigmas = []
    muts = []
    for i in range(len(out_files)):
        o = out_files[i]
        mut = mut_levels[i]
        # X = np.load("{}.npy".format(o))
        # dist2nearest = np.array([np.min(r[r>0]) for r in X]).reshape(-1, 1)
        # dist2nearest_2 = -(np.array(sorted(dist2nearest)).reshape(-1, 1))
        # dist2nearest = np.array(list(dist2nearest_2)+list(dist2nearest)).reshape(-1, 1)
        X = np.load("{}.npy".format(o))
        dist2nearest = np.array([r for r in X if r > 0]).reshape(-1, 1)
        if dist2nearest.shape[0] < 2: continue
        dist2nearest_2 = -(np.array(sorted(dist2nearest)).reshape(-1, 1))
        dist2nearest = np.array(list(dist2nearest_2)+list(dist2nearest)).reshape(-1, 1)
        gmm = mixture.GMM(n_components=3) # gmm for two components
        gmm.fit(dist2nearest) # train it!
        mean = np.max(gmm.means_)
        sigma = gmm.covars_[np.argmax(gmm.means_)]


        mut = mut_levels[i]
        if 'naive' in o:
            mu_naive.append(mean)
        else:
            mu_mem.append(mean)
            mut_level_mem.append(mut)

        mus.append(mean)
        sigmas.append(sigma)
        muts.append(mut)

    mean_mu_naive = np.mean(mu_naive)
    x, y = np.array(mut_level_mem), mean_mu_naive / np.array(mu_mem)
    p2 = np.poly1d(np.polyfit(x, y, 2))
    p3 = np.poly1d(np.polyfit(x, y, 3))
    p4 = np.poly1d(np.polyfit(x, y, 4))
    p5 = np.poly1d(np.polyfit(x, y, 5))

    np.save("poly1d_2",p2)
    np.save("poly1d_3",p3)
    np.save("poly1d_4",p4)
    np.save("poly1d_5",p5)

    xp = np.linspace(0, 50, 100)
    plt.plot(x, y, ':', marker='o', label='data')
    plt.plot(xp, p2(xp), '-', label='order 2')
    plt.plot(xp, p3(xp), '-', label='order 3')
    plt.plot(xp, p4(xp), '-', label='order 4')
    plt.plot(xp, p5(xp), '--', label='order 5')
    plt.title(r'Curve fitting to estimate $\alpha$ function for mutations. $\mu_{{NAIVE}}={:.3f}$'.format(mean_mu_naive))
    plt.ylabel(r'$\mu$ Naive / $\mu$ Mem')
    plt.xlabel(r'Igs mutation level')
    plt.ylim([0, 1])
    plt.xlim([0, 30])
    plt.legend()
    plt.savefig("alpha_mut_plot_new.png")
    plt.close()

    plt.plot(muts, mus, ':', marker='o', label='data')
    plt.title(r'$\mu$ plot based on mutation levels')
    plt.ylabel(r'$\mu$')
    plt.xlabel(r'Igs mutation level')
    plt.ylim([0, 0.6])
    plt.xlim([0, 30])
    plt.legend()
    plt.savefig("mu_on_mutation_levels.png")
    plt.close()

    plt.plot(muts, sigmas, ':', marker='o', label='data')
    plt.title(r'$\sigma$ plot based on mutation levels')
    plt.ylabel(r'$\sigma$')
    plt.xlabel(r'Igs mutation level')
    plt.ylim([0, 0.03])
    plt.xlim([0, 30])
    plt.legend()
    plt.savefig("sigma_on_mutation_levels.png")
    plt.close()




if __name__ == '__main__':
    ''' Specify the filename of donors' Igs.
    Only naive without mutations are selected. For mem Igs, subsets or 1%% mutation
    range are selected.'''

    filename_b4 = '../new_seqs/B4_db-pass.tab_CON-FUN-N.tab'
    filename_b5 = '../new_seqs/B5_db-pass.tab_CON-FUN-N.tab'

    ## donor B4
    igs_b4 = [i for i in DbCore.readDbFile(filename_b4)]

    # 1. Naive
    naive_b4 = [i for i in igs_b4 if (i.subset == 'N' and i.mut == 0)]
    naive_filtered_b4, _ = remove_duplicate_junctions(naive_b4)
    intra_donor_distance(naive_filtered_b4, type_ig='Naive', donor='B4', bins=100)

    # 2. Mem
    mem_b4 = [i for i in igs_b4 if (i.subset in ['MemA', 'MemG'])]
    mem_filtered_b4, _ = remove_duplicate_junctions(mem_b4)

    # sets1 = [(x, i+1) for i in range(int(max([m.mut for m in mem_filtered if m.mut < 24])))] + [(24,29)]
    max_mut = int(max([m.mut for m in mem_filtered_b4]))
    sets1 = [(i-1, i) for i in range(1, 24)] + [(24,max_mut+1)]
    sets2 = [(0, i) for i in range(1, 24)] + [(0,max_mut+1)]
    combinations = [x for x in zip(sets1, sets)][1:]
    for i, j in combinations:
        intra_donor_distance(mem_filtered_b4, i, j,
                             type_ig='Mem', donor='B4', bins=100)

    ## donor B4 vs B5
    igs_b5 = [i for i in DbCore.readDbFile(filename_b5)]

    # 1. Naive
    naive_b5 = [i for i in igs_b5 if (i.subset == 'N' and i.mut == 0)]
    naive_filtered_b5, _ = remove_duplicate_junctions(naive_b5)

    inter_donor_distance(naive_filtered_b4, naive_filtered_b5,
                         type_ig='Naive', donor1='B4', donor2='B5', bins=100)

    # 2. Mem
    mem_b5 = [i for i in igs_b5 if (i.subset in ['MemA','MemG'])]
    mem_filtered_b5, _ = remove_duplicate_junctions(mem_b5)

    # Mem of B4 and B5 with the same mut level
    max_mut = max(int(max([m.mut for m in mem_filtered_b4])),
                  int(max([m.mut for m in mem_filtered_b5])))
    sets1 = [(i-1, i) for i in range(1, 24)] + [(24, max_mut+1)]
    combinations = [x for x in itertools.product(sets1, sets1) if x[0][0] == x[1][0]]
    for i, j in combinations:
        inter_donor_distance(mem_filtered_b4, mem_filtered_b5, i, j,
                             type_ig='Mem', donor1='B4', donor2='B5', bins=100)
