from scipy.spatial import distance
import scipy
import numpy as np
import matplotlib.pyplot as plt

import numpy as np

from teste1 import GT

np.random.seed(0)
import matplotlib.pyplot as plt
N_CLUSTER  = 3
N_FEATURES = 2
_N = 20
N_SAMPLE  = _N * N_CLUSTER


S1 = np.identity(N_FEATURES)
S1[0,1] = S1[1,0] = 0
S2 = np.identity(N_FEATURES)*0.1
S3 = np.identity(N_FEATURES)*3
S3[0,1] = S3[1,0] = -2.8

mu1 = np.array([1,  2])
mu2 = np.array([-2,-2])
mu3 = np.array([-3, 2])

npMEANS  = np.array([mu1,mu2,mu3])
npSIGMAS = np.array([S1,S2,S3])

colors   = np.array(["red","green","blue"])
npX = []
plt.figure(figsize=(5,5))


npMAHALANOBIS = [[0 for _ in range(npMEANS.shape[0])]
            for _ in range(npX.shape[0])]
for isample in range(npX.shape[0]):
    for icluster in range(npMEANS.shape[0]):
        npMAHALANOBIS[isample][icluster] = distance.mahalanobis(npX[isample],npMEANS[icluster],
                                                                VI=scipy.linalg.pinv(npSIGMAS[icluster]))
npMAHALANOBIS = np.array(npMAHALANOBIS)

pred = npMAHALANOBIS.argmin(axis=1)
plt.figure(figsize=(5,5))
plt.scatter(npX[:,0],npX[:,1],c=colors[pred])
plt.title("Classification using Mahalanobis distance, acc={:3.2f}".format(np.mean(GT == pred)))
plt.show()
print("npMAHALANOBIS")
npMAHALANOBIS