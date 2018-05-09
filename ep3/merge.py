import numpy as np
import sys

df = [sys.argv[1], sys.argv[3]]
lf = [sys.argv[2], sys.argv[4]]
D = [None]*2
L = [None]*2

for i in range(2):
  D[i] = np.load(df[i], 'r')
  L[i] = np.load(lf[i], 'r')

nD = np.concatenate((D[0], D[1]), 0)
nL = np.concatenate((L[0], L[1]), 0)

ndf, nlf = sys.argv[5], sys.argv[6]
np.save(ndf, nD)
np.save(nlf, nL)
