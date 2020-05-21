import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

np.random.seed(7)

x1 = np.random.standard_normal((100, 2)) * 0.6 + np.ones((100, 2))
x2 = np.random.standard_normal((100, 2)) * 0.5 - np.ones((100, 2))
x3 = np.random.standard_normal((100, 2)) * 0.4 - 2 * np.ones((100, 2)) + 5
X = np.concatenate((x1, x2, x3), axis=0)

plt.plot(X[:, 0], X[:, 1], 'k.')
plt.show()

n = 3
k_means = KMeans(n_clusters=n)
k_means.fit(X)

centroids = k_means.cluster_centers_
labels = k_means.labels_

plt.plot(X[labels == 0, 0], X[labels == 0, 1], 'r.', label='cluster 1')
plt.plot(X[labels == 1, 0], X[labels == 1, 1], 'b.', label='cluster 2')
plt.plot(X[labels == 2, 0], X[labels == 2, 1], 'g.', label='cluster 3')

plt.plot(centroids[:, 0], centroids[:, 1], 'mo', markersize=8, label='centroids')

plt.legend(loc='best')
plt.show()
