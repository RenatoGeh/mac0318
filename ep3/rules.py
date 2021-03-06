import numpy as np
import sklearn.metrics as metrics
import sklearn.cluster as cluster
import pickle
import sys
import image_manipulation
import time
import USBInterface
import cv2
import usb
from Camera import Camera

# Labels.
UP, LT, RT = 0, 1, 2
LABELS_STR = ["UP", "RIGHT", "LEFT"]

# Image dimensions.
W = 160
H = 120

DEBUG = True

class Dataset:
  # Data
  D = None
  # Labels
  L = None
  # Size
  n = None

  def __init__(self, D, L):
    self.D, self.L = D, L
    self.n = len(self.D)

  def Data(self):
    return self.D

  def Labels(self):
    return self.L

  def Size(self):
    return self.n

# Prepare data.
def init_data():
  # Load data.
  D = np.load("data/data.npy", 'r')
  # Load labels.
  L = np.load("data/labels.npy", 'r').flatten()

  # Dataset sizes per label.
  D_SIZE = [0, 0, 0]
  # Dataset size.
  N = 0
  # Smaller dataset size (per label).
  M = None

  for i in L:
    D_SIZE[i] += 1
  M = D_SIZE[0]
  for n in D_SIZE:
    N += n
    if n < M:
      M = n

  return D, L, D_SIZE, N, M

# Prepare testing and training dataset.
# Arguments: dataset, labels, percentage of dataset for training ]0,1[.
def partition(D, L, p, M):
  P = int(p*M)
  Q = int(M-P)

  nD = [[], [], []]
  for i, l in enumerate(L):
    nD[l].append(i)
  for v in nD:
    np.random.shuffle(v)

  m = int(P/3)
  n = int(M/3)
  if p == 1:
    r = []
    for i in range(3):
      r.extend(nD[i][0:n])
    np.random.shuffle(r)
    rl, rd = [], []
    for i in r:
      rd.append(D[i])
      rl.append(L[i])
    R = Dataset(rd, rl)
    return R
  else:
    r, s = [], []
    for i in range(3):
      r.extend(nD[i][0:m])
    np.random.shuffle(r)
    for i in range(3):
      s.extend(nD[i][m:n])
    np.random.shuffle(s)

    rl, sl = [], []
    rd, sd = [], []
    for i in r:
      rd.append(D[i])
      rl.append(L[i])
    for i in s:
      sd.append(D[i])
      sl.append(L[i])
    R = Dataset(rd, rl)
    T = Dataset(sd, sl)

    return R, T

# Base classifier class.
class Classifier:
  # Train with this model.
  def train(self, D):
    return None

  # Returns success score of given test dataset.
  def test(self, D):
    print("Testing...")
    d, l, n = D.Data(), D.Labels(), D.Size()
    s = 0
    P = []
    for i in range(n):
      print("Testing instance " + str(i) + "...")
      c = self.classify(d[i])
      print("Classified as " + LABELS_STR[c] + ", should be " + LABELS_STR[l[i]] + ".")
      if l[i] == c:
        s += 1
      P.append(c)
    return s/n, P

  # Returns the classified label.
  def classify(self, I):
    return None

# Rules-based classifier.
class RulesBased(Classifier):
  # Image width.
  w = None
  # Image height.
  h = None
  # Sample means (indices are labels, values are lists of means of sample means).
  mu = None

  # Constructs a new rules-based classifier.
  # Arguments: w      - width
  #            h      - height
  #            sx, sy - minimum size of rectangular regions in each axis ((w, h) must be divisible by (sx, sy))
  #            q      - number of mean quantiles
  #            ctype  - classification type (by score - 0 - or accumulated error - 1)
  def __init__(self, w, h, sx, sy, q, ctype=1):
    self.w = w
    self.h = h
    self.sx, self.sy = sx, sy
    self.mu = [[], [], []]
    self.ctype = ctype
    self.q = q

  def train(self, D):
    print("Training...")
    d, l, n = D.Data(), D.Labels(), D.Size()
    for i in range(n):
      print("Training instance " + str(i) + "...")
      self.train_instance(d[i], l[i])
    # Compute means of means.
    print("Computing means of means...")
    if self.q <= 1:
      for i in range(3):
        Mu = np.asarray(self.mu[i])
        self.mu[i] = np.mean(Mu, axis=0)
    else:
      self.n = len(self.mu)
      f_mu, n_mu = [None]*3, [None]*3
      for z in range(3):
        t_i = 0
        print("  ... for label " + LABELS_STR[z] + "...")
        Mu = np.array(self.mu[z])
        print(Mu.shape)
        m = Mu.shape[1]
        f_mu[z], n_mu[z] = [[None]*m]*self.q, [[None]*m]*self.q
        for i in range(m):
          if t_i % int(m/10) == 0:
            print("    ... and column " + str(i) + "... [" + str(100*t_i/m) + "% done]")
          t_i += 1
          c = Mu[:,i]
          k = []
          for v in c:
            k.append([v])
          _, cl, _ = cluster.k_means(k, n_clusters=self.q, init='k-means++', n_jobs=-1)
          C = [[]]*self.q
          for j, v in enumerate(cl):
            C[v].append(c[j])
          for j, v in enumerate(C):
            f_mu[z][j][i] = np.mean(v)
            n_mu[z][j][i] = len(v)
      self.mu = f_mu
      self.n_mu = n_mu

  def collect_means(self, I):
    M = np.reshape(I, (self.w, self.h))
    u = [] # means
    # Get all triangular matrices means on each diagonal.
    for i in range(self.h-1):
      # Triangular matrix above the i-th diagonal.
      L = np.tril(M, i)
      u.append(np.sum(L))
      # Mirrored triangular matrix.
      mL = M-L
      u.append(np.sum(mL))
      # Triangular matrix below the i-th diagonal.
      L = np.tril(M, -i)
      u.append(np.sum(L))
      # Mirrored triangular matrix.
      mL = M-L
      u.append(np.sum(mL))
    # Get rectangular regions.
    rx, ry = int(self.w/self.sx), int(self.h/self.sy)
    for i in range(rx):
      for j in range(ry):
        R = M[i*self.sx:(i+1)*self.sx, j*self.sy:(j+1)*self.sy]
        u.append(np.sum(R))
    # Get main rectangular partitions.
    u.append(np.sum(M[0:int(self.w/2), 0:int(self.h/2)]))
    u.append(np.sum(M[0:int(self.w/2), 0:self.h]))
    u.append(np.sum(M[0:int(self.w/2), int(self.h/2):self.h]))
    u.append(np.sum(M[int(self.w/2):self.w, 0:int(self.h/2)]))
    u.append(np.sum(M[int(self.w/2):self.w, 0:self.h]))
    u.append(np.sum(M[int(self.w/2):self.w, int(self.h/2):self.h]))
    # Quadrants.
    for i in range(2, 5):
      u.append(np.sum(M[0:int(self.w/i), 0:int(self.h/i)]))
      u.append(np.sum(M[0:int(self.w/i), self.h-int(self.h/i):self.h]))
      u.append(np.sum(M[self.w-int(self.w/i):self.w, self.h-int(self.h/i):self.h]))
      u.append(np.sum(M[self.w-int(self.w/i):self.w, 0:int(self.h/i)]))
    # Get top and bottom parts.
    for i in range(3, 10):
      y1 = self.h-int(self.h/i)
      y2 = int(self.h/i)
      x1 = self.w-int(self.w/i)
      x2 = int(self.w/i)
      u.append(np.sum(M[0:self.w, y1:self.h]))
      u.append(np.sum(M[0:self.w, 0:y2]))
      u.append(np.sum(M[0:self.w, y2:y1]))
      u.append(np.sum(M[x1:self.w, 0:self.h]))
      u.append(np.sum(M[0:x2, 0:self.h]))
      u.append(np.sum(M[x2:x1, 0:self.h]))
      u.append(np.sum(M[x2:x1, y2:y1]))
    return u

  def train_instance(self, I, l):
    u = self.collect_means(I)
    # Add means to collection.
    self.mu[l].append(u)

  def classify(self, I):
    u = np.asarray(self.collect_means(I))
    # Compute scores and find most probable label.
    if self.q <= 1:
      if self.ctype == 0:
        V = [None] * 3
        for i in range(3):
          V[i] = np.abs(u-np.asarray(self.mu[i]))
        s = [0, 0, 0]
        for j in range(len(u)):
          m_i, m = -1, -1
          for i in range(3):
            d = V[i][j]
            if m_i < 0 or d < m:
              m_i, m = i, d
          s[m_i] += 1
        m_i, m = -1, -1
        for i in range(3):
          d = s[i]
          if m_i < 0 or d > m:
            m_i, m = i, d
        return m_i
      else:
        m_i, m = -1, -1
        for i in range(3):
          d = np.sum(np.abs(u-np.asarray(self.mu[i])))
          if m_i < 0 or d < m:
            m_i, m = i, d
        return m_i
    else:
      m_i, m = -1, -1
      for i in range(3):
        d = 0
        for j in range(self.q):
          d += np.abs(u-np.asarray(self.mu[i][j])).T.dot(np.asarray(self.n_mu[i][j]))
        d /= self.n
        if m_i < 0 or d < m:
          m_i, m = i, d
      return m_i

def test_classifier():
  # Get raw dataset data and labels.
  D, L, D_SIZE, N, M = init_data()
  # Convert raw data to Dataset, partitioning test and train datasets and taking a uniform number
  # of images of each label.
  R, T = partition(D, L, 0.20, M)
  print(R.Size(), T.Size())
  rules = RulesBased(H, W, 2, 2, 1)
  rules.train(R)
  s, P = rules.test(T)
  print("Classifier score: " + str(s*100) + "% sucess.")
  print("Confusion matrix:")
  print(metrics.confusion_matrix(T.Labels(), P))

def init_classifier():
  D, L, D_SIZE, N, M = init_data()
  R = partition(D, L, 1, M)
  rules = RulesBased(120, 160, 2, 2, 1)
  rules.train(R)
  return rules

def init_camera():
  K = Camera(W, H, 0)
  return K

def init_bot():
  r_exc = False
  try:
    B = next(USBInterface.find_bricks(debug=False))
    B.connect()
  except usb.core.NoBackendError:
    r_exc = True
  assert r_exc == 0, "No NXT found..."
  return B

def capture_image(K):
  img = K.take_picture()
  s_img = cv2.resize(img, (W, H))
  return image_manipulation.binarize_image(s_img)

LABELS_BYTE = ['\x02', '\x03', '\x04']

def send(B, c):
  try:
    print(LABELS_STR[c])
    B.send(LABELS_BYTE[c])
  except Exception as e:
    print(e)

# Saves classifier.
def save(C, filename):
  with open(filename, 'wb') as f:
    pickle.dump(C, f)

# Loads classifier.
def load(filename):
  with open(filename, 'rb') as f:
    return pickle.load(f)

def has_arg(i, s1, s2):
  return len(sys.argv) > 1 and (sys.argv[i] == s1 or sys.argv[i] == s2)

def run():
  if has_arg(1, '-l', '--load'):
    R = load(sys.argv[2])
  else:
    R = init_classifier()
  if has_arg(1, '-s', '--save'):
    save(R, sys.argv[2])
    return
  B = init_bot()
  K = init_camera()
  while True:
    time.sleep(0.05)
    I = capture_image(K).flatten()
    c = R.classify(I)
    send(B, c)

if __name__ == "__main__":
  if ('-h' in sys.argv) or ('--help' in sys.argv) or len(sys.argv) <= 1:
    print("Usage: " + sys.argv[0] + " [-h | --help] [-d | --debug] [-l | --load filename] [-s | --save filename]")
    print("Rules based classifier.\n")
    print("  -d, --debug   runs classifier on debug mode (shows accuracy based on in-sample dataset)")
    print("  -l, --load    loads a pickle with trained means")
    print("  -s, --save    saves this classifier's means in a pickle file")
    print("  -h, --help    shows this help message")
    sys.exit(0)
  DEBUG = ('-d' in sys.argv) or ('--debug' in sys.argv)
  if DEBUG:
    test_classifier()
  else:
    run()
