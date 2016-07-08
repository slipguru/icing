import numpy as np
import matplotlib.pyplot as plt
import seaborn

from sklearn import mixture
# from scipy.optimize import curve_fit

MU_NAIVE = 0.239688981377

res, res2, res3 = [], [0], [1]
sets = [(x, x+1) for x in range(1, 20)]
for i, j in sets:
    X1 = np.load("dist2nearest_plots/distances_B4_mem_{0}-{1}_vs_{0}-{1}_norm.npy".format(i,j))
    X2 = np.load("dist2nearest_plots_2/distances_B4_mem_0-{1}_vs_{0}-{1}_norm.npy".format(i,j))
    # X = X + np.eye(X.shape[0])
    # dist2nearest = np.array([np.min(r) for r in X]).reshape(-1, 1)
    dist2nearest = np.array([np.min(r[r>0]) for r in X1]).reshape(-1, 1)
    dist2nearest_2 = -(np.array(sorted(dist2nearest)).reshape(-1, 1))
    dist2nearest_1 = np.array(list(dist2nearest_2)+list(dist2nearest)).reshape(-1, 1)

    dist2nearest = np.array([np.min(r[r>0]) for r in X2]).reshape(-1, 1)
    dist2nearest_2 = -(np.array(sorted(dist2nearest)).reshape(-1, 1))
    dist2nearest_2 = np.array(list(dist2nearest_2)+list(dist2nearest)).reshape(-1, 1)

    gmm = mixture.GMM(n_components=3) # gmm for two components
    gmm.fit(dist2nearest_1) # train it!
    mean_1 = np.max(gmm.means_)

    gmm.fit(dist2nearest_2)
    mean_2 = np.max(gmm.means_)
    r = np.abs(mean_2-mean_1)
    print("Mean difference between {0}-{1}_vs_{0}-{1} and 0-{1}_vs_{0}-{1}: {2}".format(i,j,r))
    res.append(r)
    res2.append(j)
    res3.append(MU_NAIVE / mean_2)
res = np.array(res)
print("\nObtained a mean of {} and std {} in total".format(np.mean(res), np.std(res)))

x, y = np.array(res2), np.array(res3)
p2 = np.poly1d(np.polyfit(x, y, 2))
p3 = np.poly1d(np.polyfit(x, y, 3))
p4 = np.poly1d(np.polyfit(x, y, 4))
p5 = np.poly1d(np.polyfit(x, y, 5))

# np.save("polyfit_arguments_5",np.polyfit(x, y, 2))

xp = np.linspace(0, 50, 100)
plt.plot(x, y, ':', marker='o', label='data')
plt.plot(xp, p2(xp), '-', label='order 2')
plt.plot(xp, p3(xp), '-', label='order 3')
plt.plot(xp, p4(xp), '-', label='order 4')
plt.plot(xp, p5(xp), '--', label='order 5')
plt.title(r'Curve fitting to estimate $\alpha$ function for mutations. $\mu_{{NAIVE}}={:.3f}$'.format(MU_NAIVE))
plt.ylabel(r'$\mu$ Naive / $\mu$ Mem')
plt.xlabel(r'Igs mutation level')
plt.legend()
plt.show()
