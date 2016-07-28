# -*- coding: utf-8 -*-
'''
    Clustering learning. Using K-Means or K-Medoids to perform a learning on
    some random Gaussian distributions.
    The learning can be done with two different algorithm:

    'kmeans' - K-Means
    'kmed' - K-Medoids

    @author: Federico Tomasi
    @email: fdtomasi@gmail.com
'''

import numpy as np
import matplotlib.pyplot as plt
import random

def calc_distances(val1, val2):
    return np.linalg.norm(val1-val2)


class Kmeans:

    def __init__(self, X, Y=[], K=3):
        self.X = X
        self.d, self.n = X.shape
        self.Y = Y if len(Y) is not 0 else [0]*self.n
        self.labels = [0]*self.n
        self.K = K
        self.oldmu = self.mu = np.zeros((self.d,self.n))
        self.clusters = {}
        self.centers_found = False

    def has_converged(self):
        l1, l2 = [], []
        for i in range(self.mu.shape[1]):
            l1.append(tuple(self.mu[:,i]))
            l2.append(tuple(self.oldmu[:,i]))
        # return (set([tuple(a) for a in self.mu) == set([tuple(a) for a in self.oldmu]))
        return set(l1) == set(l2)

    def cluster_points(self):
        clusters = {}
        for j in range(self.X.shape[1]):
            key = np.argmin([calc_distances(self.X[:,j],self.mu[:,i]) for i in range(self.mu.shape[1])])
            clusters.setdefault(key, []).append(self.X[:,j])
        self.clusters = clusters.copy()

    def reevaluate_centers(self):
        for i, k in enumerate(self.clusters.keys()):
            self.mu[:,i] = np.mean(self.clusters[k], axis=0)

    def find_centers(self):
        # Initialize to K random centers
        idxs = list(range(self.n))
        random.shuffle(idxs)
        self.oldmu = self.X[:, idxs[:self.K]]

        random.shuffle(idxs)
        self.mu = self.X[:, idxs[:self.K]]

        self.clusters = {} #initialising
        i = 1
        while not self.has_converged():
            print('Iteration number: {0:3d}'.format(i))
            self.oldmu = self.mu.copy()
            # Assign all points in X to clusters
            self.cluster_points()
            # Reevaluate centers
            self.reevaluate_centers()
            i += 1
        print('... converged.')

    def plot(self):
        if not self.centers_found:
            print("Warning: run 'find_centers()' before.")
        plt.close('all')
        if self.d == 2:
            plt.figure()
            plt.title('Training set, kmeans. K=' + str(self.K) + ', N='+str(self.n))
            for i in range(self.K):
                plt.plot([j[0] for j in self.clusters[i]],
                [j[1] for j in self.clusters[i]], color=(random.random(), random.random(), random.random()), marker='o', linestyle='none')
            plt.plot(self.mu[0,:], self.mu[1,:], 'ys')
            plt.show()


def find_best_medoid(cluster):
    best_m = 0
    distances = 0
    old_dist = np.inf
    for m in cluster:
        distances = sum([np.linalg.norm(m-x) for x in cluster])
        if old_dist > distances:
            best_m = m
            old_dist = distances
    return best_m
