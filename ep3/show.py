import numpy as np
from matplotlib import pyplot as plt

D = np.load('data/data.npy')
L = np.load('data/labels.npy')

labels = ["UP", "LEFT", "RIGHT"]

for i in range(len(D)):
  M = np.asarray(D[i]).reshape(120, 160)
  print(labels[int(L[i])])
  plt.imshow(M, cmap='gray')
  plt.show()
