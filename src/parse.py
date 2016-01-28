#!/usr/bin/env python3
import numpy as np
from os import path
import matplotlib.pyplot as plt
from sklearn import svm, cluster, decomposition, mixture
from hmmlearn import hmm
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.optimizers import *
import random

MEAN = None
STD = None

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
        if d.shape[1] == 113:
            d = np.concatenate([d[:, :64],d[:, 65:]], axis=1)
        d = np.concatenate([d[:, 1:46],d[:, 49:109]], axis=1)
        if MEAN is not None:
            d = (d - MEAN) / STD
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
d_train = np.concatenate(d1, axis=0)
MEAN = np.mean(d_train, axis=0)
STD = np.std(d_train, axis=0)

d_em, _ = load_data('stand', 2, 1)
d_l, _ = load_data('lock', 15, 1)
d_s, _ = load_data('scuba', 12, 1)
d_sk, _ = load_data('skitter', 5, 1)
d_p, _ = load_data('punch', 5, 1)
d_po, _ = load_data('point', 4, 1)
d_lpo, _ = load_data('lpoint', 4, 1)
d_m, _ = load_data('mix', 1, 1)
d_t, _ = load_data('tempo3', 1, 1)

d_em = np.concatenate(d_em, axis=0)
d_l = np.concatenate(d_l, axis=0)
d_s = np.concatenate(d_s, axis=0)
d_sk = np.concatenate(d_sk, axis=0)
d_p = np.concatenate(d_p, axis=0)
d_po = np.concatenate(d_po, axis=0)
d_lpo = np.concatenate(d_lpo, axis=0)
d_m = np.concatenate(d_m, axis=0)
d_t = np.concatenate(d_t, axis=0)

print('Preprocess complete')

EPOCH = 100
CLASS = 6
BATCH = 32

model = Sequential()

# model.add(BatchNormalization(input_shape=(d_l.shape[1],)))

# model.add(Dense(output_dim=128, input_dim=d_l.shape[1], init="glorot_uniform"))
# model.add(Activation("sigmoid"))
# model.add(Dense(output_dim=64, init="glorot_uniform"))
# model.add(Activation("sigmoid"))
# model.add(Dense(output_dim=64, init="glorot_uniform"))
# model.add(Activation("sigmoid"))
# model.add(Dense(output_dim=CLASS, init="glorot_uniform"))

model.add(LSTM(output_dim=128, input_dim=d_l.shape[1], activation='tanh', return_sequences=True))
model.add(LSTM(output_dim=32, input_dim=128, activation='tanh', return_sequences=True))
model.add(TimeDistributedDense(output_dim=CLASS, init="glorot_uniform"))

model.add(Activation("softmax"))

# model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.0003, momentum=0.9, nesterov=True))
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.005))

train = False

if train:
    X = []
    Y = []
    W = {}
    for i, z in enumerate([d_l, d_s, d_sk, d_p, d_po, d_lpo]):
        X.append(z)
        oz = np.zeros([CLASS])
        oz[i] = 1
        Y.extend([oz]*z.shape[0])
        # W.extend([1./z.shape[0]]*z.shape[0])
        W[i] = 1./z.shape[0]

    X = np.concatenate(X, axis=0)
    Y = np.array(Y)
    # W = np.array(W)
    rat = len(Y) / sum(W)
    for i in W:
        W[i] *= rat

    perm = list(range(len(Y)))
    random.shuffle(perm)
    X = X[perm,:]
    Y = Y[perm]
    # W = W[perm]

    xmean = np.mean(X, axis=0)
    xstd = np.std(X, axis=0)

    print(X.shape, Y.shape)
    bn = Y.shape[0] // BATCH
    Xt = X[:BATCH*bn,:].reshape((BATCH, bn, X.shape[1]))
    Yt = Y[:BATCH*bn,:].reshape((BATCH, bn, Y.shape[1]))
    print(Xt.shape, Yt.shape)
    model.fit(Xt, Yt, nb_epoch=EPOCH, batch_size=BATCH)#, class_weight=W)
    model.save_weights('my_model_weights.h5')

else:
    model.load_weights('my_model_weights.h5')


dtt = d_t[np.newaxis,:,:]
proba = model.predict_proba(dtt, batch_size=BATCH)
proba = proba[0]

K = 25
pf = [np.mean(proba[x:x+K, :], axis=0) for x in range(proba.shape[0]-K)]
pf = np.array(pf)

plt.plot(pf[:, 0], label='lock')
plt.plot(pf[:, 1], label='scuba')
plt.plot(pf[:, 2], label='skitter')
plt.plot(pf[:, 3], label='punch')
plt.plot(pf[:, 4], label='point')
plt.plot(pf[:, 5], label='lpoint')
plt.legend()
plt.show()

exit()

N_cluster = 20
cluster_model = mixture.GMM(n_components=N_cluster, random_state=514)
# lab = cluster_model.fit_predict(np.concatenate([d_s, d_l, d_sk, d_p], axis=0))
lab = cluster_model.fit_predict(d_s)
# clab = compress(lab)
llab = cluster_model.predict(d_l)
slab = cluster_model.predict(d_s)
mlab = cluster_model.predict(d_m)

print(compress(llab))
print(compress(slab))
print(compress(mlab))

# pca_model = decomposition.PCA(n_components=50)
# pca_model.fit(d_train)
# d_l, d_s, d_m, d_sk, d_p = [pca_model.transform(x) for x in [d_l, d_s, d_m, d_sk, d_p]]

N_State = N_cluster

hmm_l = hmm.GaussianHMM(n_components=N_State, covariance_type="diag", n_iter=1000, algorithm="map")
hmm_l.fit([d_l])
hmm_s = hmm.GaussianHMM(n_components=N_State, covariance_type="diag", n_iter=1000, algorithm="map")
# hmm_s.means_ = cluster_model.means_
# hmm_s.covars_ = cluster_model.covars_
hmm_s.fit([d_s])
# tmp = hmm_s.transmat_ + 0.02
# hmm_s.transmat_ = tmp / np.sum(tmp, axis=1)
# print(hmm_s.transmat_)
hmm_sk = hmm.GaussianHMM(n_components=N_State, covariance_type="diag", n_iter=1000, algorithm="map")
hmm_sk.fit([d_sk])
hmm_p = hmm.GaussianHMM(n_components=N_State, covariance_type="diag", n_iter=1000, algorithm="map")
hmm_p.fit([d_p])

sc_l = []
sc_s = []
sc_sk = []
sc_p = []

print(hmm_s.score(d_l), hmm_s.score(d_s), hmm_s.score(d_m))
print(compress(list(hmm_s.predict(d_s))))
print(compress(list(hmm_s.predict(d_m))))

# dd = np.concatenate([d_l, d_s, d_sk, d_p, d_m, d_train], axis=0)
dd = np.concatenate([d_l, d_s, d_m], axis=0)

seq = hmm_s.predict(dd)
gg = []
for i in range(dd.shape[0]-1):
    pb = -np.sum((dd[i,:] - hmm_s.means_[seq[i],:])**2 / np.diag(hmm_s.covars_[seq[i]])) / 2
    # pb = 0
    pb += np.log(hmm_s.transmat_[seq[i], seq[i+1]])
    gg.append(pb)

step = 5
window = 25
for i in range(window, dd.shape[0]-window, step):
    sc_l.append(hmm_l.score(dd[i-window:i+window, :]))
    sc_s.append(hmm_s.score(dd[i-window:i+window, :]))
    # sc_s.append(np.mean(gg[i-window:i+window]))
    # sc_sk.append(hmm_sk.score(dd[i-window:i+window, :]))
    # sc_p.append(hmm_p.score(dd[i-window:i+window, :]))

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
# plt.plot(sc_sk, label='skitter')
# plt.plot(sc_p, label='punch')
plt.legend()
plt.show()
