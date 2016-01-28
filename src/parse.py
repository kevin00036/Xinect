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
import glob

EPOCH = 100
CLASS = 5
BATCH = 32

lmap = {}
lmap2 = {}

def load_label_map():
    path = '../label/map.txt'
    for line in open(path):
        a, b, c = line.split()
        lmap[a] = int(b)
        lmap2[int(b)] = c

def load_data(name):
    feature_path = '../feature/{}.txt_f.txt'.format(name)
    label_path = '../label/{}.txt'.format(name)

    d = [[float(x) for x in l.strip().split()] for l in open(feature_path)]
    d = np.array(d)
    if d.shape[1] == 113:
        d = np.concatenate([d[:, :64],d[:, 65:]], axis=1)
    d = np.concatenate([d[:, 1:46],d[:, 49:109]], axis=1)

    lab = np.zeros((d.shape[0], CLASS))
    for line in open(label_path):
        if len(line) <= 1:
            continue
        a, b, c = line.split()
        l = lmap[c]
        for i in range(int(a), int(b)):
            if l <= CLASS:
                lab[i, l-1] = 1.

    return d, lab

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

train_datas = []
train_labs = []
test_datas = []
test_labs = []

load_label_map()
files = glob.glob('../feature/*.txt_f.txt')
for f in files:
    fn = f.split('/')[-1].split('.')[0]
    print('Loading {}...'.format(fn))
    data, lab = load_data(fn)
    if fn in ['test_01']:
        test_datas.append(data)
        test_labs.append(lab)
    else:
        train_datas.append(data)
        train_labs.append(lab)

train_datas = np.concatenate(train_datas, axis=0)
train_labs = np.concatenate(train_labs, axis=0)
test_datas = np.concatenate(test_datas, axis=0)
test_labs = np.concatenate(test_labs, axis=0)
test_intlabs = np.argmax(test_labs, axis=1)

print('=== Load Complete. ===')

MEAN = np.mean(train_datas, axis=0)
STD = np.std(train_datas, axis=0)
train_datas = (train_datas - MEAN) / STD
test_datas = (test_datas - MEAN) / STD

print('=== Preprocess Complete. ===')


model = Sequential()

# model.add(BatchNormalization(input_shape=(d_l.shape[1],)))

# model.add(Dense(output_dim=128, input_dim=d_l.shape[1], init="glorot_uniform"))
# model.add(Activation("sigmoid"))
# model.add(Dense(output_dim=64, init="glorot_uniform"))
# model.add(Activation("sigmoid"))
# model.add(Dense(output_dim=64, init="glorot_uniform"))
# model.add(Activation("sigmoid"))
# model.add(Dense(output_dim=CLASS, init="glorot_uniform"))

model.add(LSTM(output_dim=128, input_dim=train_datas.shape[1], activation='tanh', return_sequences=True))
model.add(LSTM(output_dim=32, input_dim=128, activation='tanh', return_sequences=True))
model.add(TimeDistributedDense(output_dim=CLASS, init="glorot_uniform"))

model.add(Activation("softmax"))

# model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.0003, momentum=0.9, nesterov=True))
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.005))

train = True

if train:
    print('=== Training Started. ===')

    X = train_datas
    Y = train_labels

    perm = list(range(len(Y)))
    random.shuffle(perm)
    X = X[perm,:]
    Y = Y[perm]

    print(X.shape, Y.shape)
    bn = Y.shape[0] // BATCH
    Xt = X[:BATCH*bn,:].reshape((BATCH, bn, X.shape[1]))
    Yt = Y[:BATCH*bn,:].reshape((BATCH, bn, Y.shape[1]))
    print(Xt.shape, Yt.shape)
    model.fit(Xt, Yt, nb_epoch=EPOCH, batch_size=BATCH)#, class_weight=W)
    model.save_weights('my_model_weights.h5')

    print('=== Training Completed. ===')

else:
    model.load_weights('my_model_weights.h5')

print('=== Prediction Started. ===')

dtt = test_datas[np.newaxis,:,:]
proba = model.predict_proba(dtt, batch_size=BATCH)
proba = proba[0]

print('=== Prediction Completed. ===')

K = 25
pf = [np.mean(proba[x:x+K, :], axis=0) for x in range(proba.shape[0]-K)]
pf = np.array(pf)

for i in range(CLAS):
    plt.plot(pf[:, i], label=lmap2[i])
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
