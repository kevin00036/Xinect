#!/usr/bin/env python3

import matplotlib
# matplotlib.use('TkAgg')

import numpy as np
from os import path
import matplotlib.pyplot as plt
from matplotlib import animation
import random
import glob

def load_data(fn, label):
    d = [[float(x) for x in l.strip().split()] for l in open(fn)]
    d = np.array(d)
    if d.shape[1] == 113:
        d = np.concatenate([d[:, :64],d[:, 65:]], axis=1)
    # d = np.concatenate([d[:, 1:46],d[:, 49:109]], axis=1)

    lab = np.array([label] * d.shape[0])

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

def transform(data):
    center = data[:, 46:49]
    x = data[:,1:46:3] #- center[:, 0, np.newaxis]
    y = data[:,2:46:3] #- center[:, 1, np.newaxis]
    z = data[:,3:46:3] #- center[:, 2, np.newaxis]

    return x, y

def save_animation(filename):
    fn = filename.split('/')[-1]
    fn = fn.split('.')[0]
    label, number = fn.split('_')
    data, lb = load_data(filename, label)
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

    anim.save('../animate/{}_{}.mp4'.format(label, number), fps=40, bitrate=5000, dpi=200)
    # plt.show()
    fig.close()

files = glob.glob('../feature/*.txt_f.txt')
for f in files:
    print('Processing : {}'.format(f))
    save_animation(f)
