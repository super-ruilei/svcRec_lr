import pandas as pd
import random
import matplotlib.pyplot as plt
import scipy
from scipy import sparse
from sklearn import neighbors
import numpy as np

RAW_USRLIST = "../dataset/userlist.txt"
RAW_WSLIST = "../dataset/wslist.txt"
RAW_TPMATRIX = "../dataset/tpMatrix.txt"
RAW_RTMATRIX = "../dataset/rtMatrix.txt"

UserMatrix = pd.read_csv(RAW_USRLIST, sep='\t', skiprows=[1], index_col=0)
WSMatrix = pd.read_csv(RAW_WSLIST, sep='\t', skiprows=[1], index_col=0)
TPMatrix = pd.read_csv(RAW_TPMATRIX, sep='\t', header=None)
RTMatrix = pd.read_csv(RAW_RTMATRIX, sep='\t', header=None)

TPMatrix.drop(TPMatrix.shape[1] - 1, axis=1, inplace=True)
RTMatrix.drop(RTMatrix.shape[1] - 1, axis=1, inplace=True)

# remove [] for the feature key
for table in [UserMatrix, WSMatrix]:
    table.rename(
        columns={origin: origin[1:-1] for origin in table.columns},
        inplace=True
    )

RTSparse = RTMatrix.apply(lambda row: row.sample(frac=0.5, random_state=row.name), axis=1)
TPSparse = TPMatrix.apply(lambda row: row.sample(frac=0.5, random_state=row.name), axis=1)

distMatrix = pd.DataFrame([UserMatrix['Latitude'], UserMatrix['Longitude']]).T
euclideanDist = neighbors.DistanceMetric.get_metric('haversine')
userDistCorr = euclideanDist.pairwise(distMatrix)

NUM_USER_TRAIN = 338
NUM_USER_TEST = 1
NUM_SERVICE = 5825
K = 5
tpTrain, tpTest = TPSparse[:NUM_USER_TRAIN], TPSparse[-NUM_USER_TEST:]
tpTest.index = range(1)
tpTestReplica = tpTest.append([tpTest] * (NUM_USER_TRAIN - NUM_USER_TEST), ignore_index=True)


def cal_weight(u, v):
    uIndices, vIndices = np.where(~np.isnan(u)), np.where(~np.isnan(v))
    intersect = np.intersect1d(uIndices, vIndices)
    union = np.union1d(uIndices, vIndices)
    return len(intersect) / len(union)


weight_u_v = tpTrain.apply(lambda u: cal_weight(u, tpTest), axis=1)
krcc_u_v = tpTrain.corrwith(tpTestReplica, axis=1, method='kendall')
sim_u_v = weight_u_v * krcc_u_v


def calPre(u):
    pre_val = np.zeros((NUM_SERVICE, NUM_SERVICE))
    testM = np.nan_to_num(u)
    for (i, v) in enumerate(testM.ravel()):
        pre_val[i] = np.sign(v - testM)
        # pre_val[i] = np.sign(testM[i] - testM)
    return pre_val


preference_direct = calPre(tpTest)

top_k = [(i, v) for i, v in sorted(enumerate(sim_u_v), key=lambda t: t[1], reverse=True)][0:K]
pre_sim = np.zeros((NUM_SERVICE, NUM_SERVICE))
for (i, v) in top_k:
    user = tpTrain.values[i]
    pre_sim = pre_sim + v * calPre(user)
preference_neighbor = np.sign(pre_sim)

preference = np.where(preference_direct != 0, preference_direct, preference_neighbor)

plt.matshow(preference[:10, :10])

service_with_index = [(i, v) for i, v in enumerate(preference.sum(axis=0))]
target = sorted(service_with_index, key=lambda t: t[1])[::-1][0][0]
print(target)