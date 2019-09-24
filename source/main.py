import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def cal_weight(u, v):
    uIndices, vIndices = np.where(~np.isnan(u)), np.where(~np.isnan(v))
    intersect = np.intersect1d(uIndices, vIndices)
    union = np.union1d(uIndices, vIndices)
    return len(intersect) / len(union)

def calPre(u):
    testM = np.nan_to_num(u)
    f = np.repeat(testM,NUM_SERVICE,axis=0).T
    v = np.repeat(testM,NUM_SERVICE,axis=0)
    pre_val = f - v
    return np.sign(pre_val)

NUM_USER_TRAIN = 299
NUM_USER_TEST = 1
NUM_SERVICE = 500
NUM_USER = 300
K = 60

# step1. load groundtruth data
RAW_TPMATRIX = "../dataset2/tpMatrix"
TPMatrix = np.loadtxt(RAW_TPMATRIX,  delimiter='\t')


# step2. form sparse data with sparse_rate
spare_rate = np.array([0.1,0.3,0.5,1.0])
sample_num = (NUM_SERVICE*spare_rate).astype(int)
TPSparse = np.full([NUM_USER,NUM_SERVICE], np.nan)


for i in range(NUM_USER):
    p = np.random.choice(TPMatrix.shape[0], sample_num[0], replace=False);
    TPSparse[i,p] = TPMatrix[i,p]


# step3. split train and test
tpTrain = TPSparse[NUM_USER_TEST:,:]
gt_tp_test, tpTest = TPMatrix[0:NUM_USER_TEST,:],TPSparse[0:NUM_USER_TEST,:]


# step4. cal krcc
tptrain = pd.DataFrame(tpTrain)
tptest = pd.DataFrame(np.repeat(tpTest,NUM_USER_TRAIN,axis=0))
weight_u_v = tptrain.apply(lambda u: cal_weight(u, tptest), axis=1)
krcc_u_v = tptrain.corrwith(tptest, axis=1, method='kendall')
# sim_u_v = weight_u_v * krcc_u_v
sim_u_v = krcc_u_v

# step5. cal preference
preference_direct = calPre(tpTest)
top_k = [(i, v) for i, v in sorted(enumerate(sim_u_v), key=lambda t: t[1], reverse=True)][0:K]
pre_sim = np.zeros((NUM_SERVICE, NUM_SERVICE))


for (i, v) in top_k:
    user = tpTrain[i]
    user = user[np.newaxis,:]
    pre_sim = pre_sim + v * calPre(user)
preference_neighbor = np.sign(pre_sim)

preference = np.where(preference_direct != 0, preference_direct, preference_neighbor)

# plt.matshow(preference[:10, :10])

ind_d = np.sum(-preference,axis=1).argsort()
ind_d = np.array(ind_d[0:100])
gt_ind = (-gt_tp_test).argsort()[0:100]
gt_ind = np.array(gt_ind[0,0:100])

recall = np.intersect1d(ind_d,gt_ind)
print(recall.shape)

