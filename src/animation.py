#!/usr/bin/env python3

import matplotlib
# matplotlib.use('TkAgg')

import numpy as np
from os import path
import matplotlib.pyplot as plt
from matplotlib import animation
import random

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
        # d = np.concatenate([d[:, 1:46],d[:, 49:109]], axis=1)
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

def transform(data):
    center = data[:, 46:49]
    x = data[:,1:46:3] #- center[:, 0, np.newaxis]
    y = data[:,2:46:3] #- center[:, 1, np.newaxis]
    z = data[:,3:46:3] #- center[:, 2, np.newaxis]

    return x, y

d1, _ = load_data('random', 1, 1)
d_em, _ = load_data('stand', 2, 1)
d_l, _ = load_data('lock', 15, 1)
d_s, _ = load_data('scuba', 12, 1)
d_sk, _ = load_data('skitter', 5, 1)
d_p, _ = load_data('punch', 5, 1)
d_po, _ = load_data('point', 4, 1)
d_lpo, _ = load_data('lpoint', 4, 1)
d_m, _ = load_data('mix', 1, 1)
d_t, _ = load_data('tempo3', 1, 1)

data = np.concatenate(d_l, axis=0)
X, Y = transform(data)

fig = plt.figure()
S = 3
minX = -600 #np.mean(X) - S * np.std(X)
maxX = 600 #np.mean(X) + S * np.std(X)
minY = -1000 #np.mean(Y) - S * np.std(Y)
maxY = 1000 #np.mean(Y) + S * np.std(Y)
ax = plt.axes(xlim=(minX, maxX), ylim=(minY, maxY))
lines = []
for i in range(14):
    lines.extend(ax.plot([], [], lw=2))
ttl = ax.text(minX+100, maxY-200, 'xd', fontsize=20)

st = [0, 1, 1, 1, 2, 3, 4, 5, 8,  8,  9, 10, 11, 12]
ed = [1, 2, 3, 8, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14]

def init():
    for i in range(14):
        lines[i].set_data([], [])
    ttl.set_text('')
    return lines + [ttl]

def animate(i):
    x = X[i]
    y = Y[i]
    for j in range(14):
        lines[j].set_data([x[st[j]], x[ed[j]]], [y[st[j]], y[ed[j]]])
    ttl.set_text('{:04d}'.format(i))
    return lines + [ttl]

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=X.shape[0], interval=20, blit=True)

anim.save('lock.mp4', fps=40, bitrate=2000)
# plt.show()
