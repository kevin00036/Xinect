import numpy as np
from os import path
import matplotlib.pyplot as plt
from sklearn import svm
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

d1, l1 = load_data('lock', 10, 1)
d2, l2 = load_data('scuba', 6, 2)
d2, l2 = load_data('mix', 1, 0)

model = svm.LinearSVC()

d_train = np.concatenate(d1 + d2, axis=0)
l_train = np.concatenate(l1 + l2)

d_test = np.concatenate(d3, axis=0)
l_test = np.concatenate(l3)

model.fit(d_train, l_train)

p = model.predict(d_test)
print(p, l_test)
cor = np.mean(p == l_test)
print(cor)
