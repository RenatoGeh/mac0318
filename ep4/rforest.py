import numpy as np
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
import sys
import pickle
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

  return np.asarray(D), np.asarray(L), D_SIZE, N, M

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

# Random forest classifier.
class RandomForest(Classifier):

  # Constructs a new random forest classifier.
  # Arguments:   n  - number of trees in the forest
  #              c  - 'gini' or 'entropy'
  def __init__(self, n, c):
    self.F = RandomForestClassifier(n_estimators=n, criterion=c)

  def train(self, D):
    print("Training...")
    d, l, n = D.Data(), D.Labels(), D.Size()
    d = np.asarray(d)
    self.F.fit(d, l)

  def classify(self, I):
    tI = np.array([I])
    l = self.F.predict(tI)
    return l[0]

def test_classifier():
  # Get raw dataset data and labels.
  D, L, D_SIZE, N, M = init_data()
  # Convert raw data to Dataset, partitioning test and train datasets and taking a uniform number
  # of images of each label.
  R, T = partition(D, L, 0.50, M)
  print(R.Size(), T.Size())
  forest = RandomForest(100, 'entropy')
  forest.train(R)
  s, P = forest.test(T)
  print("Classifier score: " + str(s*100) + "% sucess.")
  print("Confusion matrix:")
  print(metrics.confusion_matrix(T.Labels(), P))

def init_classifier():
  D, L, D_SIZE, N, M = init_data()
  R = partition(D, L, 1, M)
  forest = RandomForest(10, 'entropy')
  forest.train(R)
  return forest

def init_camera():
  K = Camera(W, H, 1)
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
  if ('-h' in sys.argv) or ('--help' in sys.argv):
    print("Usage: " + sys.argv[0] + " [-h | --help] [-d | --debug] [-l | --load filename] [-s | --save filename]")
    print("Random forest classifier.\n")
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
