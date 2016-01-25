#!/usr/bin/env python3
import numpy as np
from os import path
import matplotlib.pyplot as plt
from sklearn import svm, cluster
from hmmlearn import hmm
import random

def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_data(data_type, num, label):
    data_path = '../feature'.format(data_type)
    filenames = ['{}_{:02d}.txt_f.txt'.format(data_type, i) for i in range(1, num+1)]
    filenames = [path.join(data_path, x) for x in filenames]

    data = []

    for fn in filenames:
        d = [[float(x) for x in l.strip().split()] for l in open(fn)]
        d = np.array(d)
        d = np.concatenate([d[:, 1:46],d[:, 49:110]], axis=1)
        data.append(d)

    lab = [np.array([label] * x.shape[0]) for x in data]

    return data, lab

def compress(l, thr=2):
    ret = []
    lst = None
    cnt = 0
    for c in l:
        if lst != c:
            cnt = 1
            lst = c
        else:
            cnt += 1
        if cnt == thr:
            ret.append(c)
    if thr > 1:
        ret = compress(ret, thr=1)
    return ret

d1, _ = load_data('random', 1, 1)
d_l, _ = load_data('lock', 10, 1)
d_s, _ = load_data('scuba', 7, 1)
d_sk, _ = load_data('skitter', 1, 1)
d_p, _ = load_data('punch', 1, 1)
d_m, _ = load_data('mix', 1, 1)

d_train = np.concatenate(d1, axis=0)
d_l = np.concatenate(d_l, axis=0)
d_s = np.concatenate(d_s, axis=0)
d_sk = np.concatenate(d_sk, axis=0)
d_p = np.concatenate(d_p, axis=0)
d_m = np.concatenate(d_m, axis=0)

N = 20
cluster_model = cluster.KMeans(n_clusters=N, random_state=514, n_jobs=8)

lab = cluster_model.fit_predict(d_train)
clab = compress(lab)

llab = cluster_model.transform(d_l)
slab = cluster_model.transform(d_s)
mlab = cluster_model.transform(d_m)

# d_l, d_s, d_m = llab, slab, mlab

N_State = 20

hmm_l = hmm.GaussianHMM(n_components=N_State, covariance_type="diag", n_iter=1000, algorithm="viterbi")
hmm_l.fit([d_l])
hmm_s = hmm.GaussianHMM(n_components=N_State, covariance_type="diag", n_iter=1000, algorithm="viterbi")
hmm_s.fit([d_s])
hmm_sk = hmm.GaussianHMM(n_components=N_State, covariance_type="diag", n_iter=1000, algorithm="viterbi")
hmm_sk.fit([d_sk])
hmm_p = hmm.GaussianHMM(n_components=N_State, covariance_type="diag", n_iter=1000, algorithm="viterbi")
hmm_p.fit([d_p])

sc_l = []
sc_s = []
sc_sk = []
sc_p = []

print(hmm_s.score(d_l), hmm_s.score(d_s), hmm_s.score(d_m))
print(list(hmm_s.predict(d_s)))

dd = np.concatenate([d_l, d_s, d_sk, d_p, d_m, d_train], axis=0)

step = 5
for i in range(step, dd.shape[0]-2*step, step):
    sc_l.append(hmm_l.score(dd[i:i+2*step, :]))
    sc_s.append(hmm_s.score(dd[i:i+2*step, :]))
    sc_sk.append(hmm_sk.score(dd[i:i+2*step, :]))
    sc_p.append(hmm_p.score(dd[i:i+2*step, :]))

sc_l = np.array(sc_l)
sc_l /= -np.mean(sc_l)
sc_s = np.array(sc_s)
sc_s /= -np.mean(sc_s)
sc_sk = np.array(sc_sk)
sc_sk /= -np.mean(sc_sk)
sc_p = np.array(sc_p)
sc_p /= -np.mean(sc_p)

plt.plot(sc_l, label='lock')
plt.plot(sc_s, label='scuba')
plt.plot(sc_sk, label='skitter')
plt.plot(sc_p, label='punch')
plt.legend()
plt.show()
