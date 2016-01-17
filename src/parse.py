import numpy as np
from os import path
import matplotlib.pyplot as plt

def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

data_type = 'lock'
data_path = '../feature_{}'.format(data_type)
filenames = ['{}_{:02d}.txt_f.txt'.format(data_type, i) for i in range(1, 10+1)]
filenames = [path.join(data_path, x) for x in filenames]

data = []

for fn in filenames:
    d = [[float(x) for x in l.strip().split()] for l in open(fn)]
    d = np.array(d)
    data.append(d)

res = []
for i in range(data[1].shape[0]):
    s = cos_sim(data[0][0,:], data[1][i,:])
    res.append(s)

plt.plot(res)
plt.show()
