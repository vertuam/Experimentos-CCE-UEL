import numpy as np
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
for icluster in range(N_CLUSTER):
    color     = colors[icluster]
    npmean    = npMEANS[icluster]
    s_cluster = np.random.multivariate_normal(npmean,npSIGMAS[icluster],_N)
    npX.extend(s_cluster)
    plt.plot(npmean[0],npmean[1],"X",color=color)
    plt.plot(s_cluster[:,0],s_cluster[:,1],"p",alpha=0.3,color=color,label="cluster={}".format(icluster))
plt.title("Sample distribution from {} clusters".format(N_CLUSTER))
plt.legend()
plt.show()

GT = [0 for _ in range(_N)] + [1 for _ in range(_N)] + [2 for _ in range(_N)]
npX = np.array(npX)
print("npX:      Data Dimension = (N_SAMPLE,N_FEATURES)  = {}".format(npX.shape))
print("npMEANS:  Data Dimension = (N_CLUSTER,N_FEATURES) = {}".format(npMEANS.shape))
print("npSIGMAS: Data Dimension = (N_CLUSTER,N_FEATURES,N_FEATURES) = {}".format(npSIGMAS.shape))

for icluster in range(N_CLUSTER):
    print("\n***CLUSTER={}***".format(icluster))
    print(">>>MEAN")
    print(npMEANS[icluster])
    print(">>>SIGMA")
    print(npSIGMAS[icluster])