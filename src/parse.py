#!/usr/bin/env python3
import numpy as np
from os import path
import matplotlib.pyplot as plt
from sklearn import svm, cluster
import random

def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_data(data_type, num, label):
    data_path = '../feature_{}'.format(data_type)
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

def main():
    d1, _ = load_data('random', 1, 1)
    d_l, _ = load_data('lock', 10, 1)
    d_s, _ = load_data('scuba', 6, 1)
    d_m, _ = load_data('mix', 1, 1)

    d_train = np.concatenate(d1, axis=0)
    d_l = np.concatenate(d_l, axis=0)
    d_s = np.concatenate(d_s, axis=0)
    d_m = np.concatenate(d_m, axis=0)

    cluster_model = cluster.KMeans(n_clusters=20, random_state=514, n_jobs=4)

    lab = cluster_model.fit_predict(d_train)
    clab = compress(lab)

    llab = cluster_model.transform(d_l)
    slab = compress(cluster_model.predict(d_s))
    mlab = compress(cluster_model.predict(d_m))

    print(llab)



if __name__ == '__main__':
    main()
