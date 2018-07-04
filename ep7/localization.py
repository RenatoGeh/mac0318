import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib
import math

import USBInterface
import usb
import sys

# Utility functions.
def _find_getch():
  try:
    import termios
  except ImportError:
    import msvcrt
    return msvcrt.getch
  import sys, tty
  def _getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
      tty.setraw(fd)
      ch = sys.stdin.read(1)
    finally:
      termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch
  return _getch
getch = _find_getch()

def pause_nofocus(interval):
  backend = plt.rcParams['backend']
  if backend in matplotlib.rcsetup.interactive_bk:
    figManager = matplotlib._pylab_helpers.Gcf.get_active()
    if figManager is not None:
      canvas = figManager.canvas
      if canvas.figure.stale:
        canvas.draw()
      canvas.start_event_loop(interval)
      return

def round_even(n):
  return round(n/2)*2

# Signals.
NOOP = 0
SEND = 1
RECV = 2
QUIT = 3

class Range:
  # Arguments:
  #  C - Commands (in positional increments), e.g. [-1, 0, 1] (left, noop, right).
  #  M - Maximum steps for each command, e.g. [-25, 0, 25] (-25, 0, 25) steps.
  def __init__(self, C, M, p):
    self.C = C
    self.M = M
    self.R = []
    for i, c in enumerate(C):
      if c != 0:
        self.R.extend(np.arange(0, M[i], c))
    self.R = np.unique(self.R)
    self.pivot = np.abs(np.min(M))-1 # pivot
    self.N = self.R+self.pivot # normalized range
    self.p = p

  # Range.get(k) == normalized_range[pivot+k] == index of k (k can be negative).
  def get(self, i):
    return self.N[self.pivot+i]

class Map:
  # Arguments:
  #  M - Map (each entry is an index for the gaussian array).
  #  N - Gaussians for each 'color'.
  #  d - Distance between cells.
  #  P - Initial belief probability distribution.
  #  p - Precision (how many standard deviations from the mean should we precompute).
  #  R - Range.
  def __init__(self, M, N, d, P, p, R):
    self.M, self.m = M, len(M)
    self.N, self.n = N, len(N)
    self.P = np.asarray(P)
    # Store means, std and vars.
    self.Mu, self.Sigma, self.Var = [0 for i in range(self.n)], [0 for i in range(self.n)], [0 for i in range(self.n)]
    for i in range(self.n):
      self.Mu[i], self.Sigma[i], self.Var[i] = self.N[i].mean(), self.N[i].std(), self.N[i].var()
    self.p = p
    self.R = R
    self.C, self.c = R.R, len(R.R)
    self.d = d
    self.B = None
    self.simulate = False
    # Precompute matrices.
    self.precompute_sensor()
    #self.precompute_action()

  def precompute_sensor(self):
    # Pz[z] = P(Z=z|X)
    # Pz[z][x] = P(Z=z|X=x)
    # Precompute all (integer) values from [-p*sd, +p*sd], where sd is the standard deviation.
    m_s = np.max(self.Sigma)
    _Pz = [[0 for j in range(self.m)] for i in range(int(np.max(self.Mu)+round(self.p*m_s))+2)]
    for x in range(self.m):
      t = self.M[x]
      r = int(round(self.p*m_s))+2
      for z in range(r):
        s = self.Mu[t] + z
        i, j = int(round(s)), int(round(self.Mu[t]-z))
        _Pz[i][x] = self.N[t].pdf(s)
        _Pz[j][x] = self.N[t].pdf(s)
    self.Pz = np.asarray(_Pz)

  def precompute_action(self):
    # Pxxu[u] = P(X'|X,u)
    # Pxxu[u][x] = P(X'=x'|X,u)
    # Pxxu[u][x][x'] = P(X'=x'|X=x,u)
    _Pxxu = [[[0 for i in range(self.m)] for j in range(self.m)] for l in range(self.c)]
    self.Pxxu = [None for i in range(self.c)]
    for u in range(self.c):
      for x in range(self.m):
        mu, partial = x+self.C[u], self.R.p*self.C[u]
        std = abs(partial) if u != 0 else 2.0
        N = stats.norm(mu, std)
        a = N.pdf(np.arange(self.m))
        _Pxxu[u][x] = a/np.sum(a)
      _Pxxu[u] = np.asarray(_Pxxu[u])
      self.Pxxu[u] = np.asmatrix(_Pxxu[u]).T

  def action(self, u):
    _Pxxu = [[0 for i in range(self.m)] for j in range(self.m)]
    for x in range(self.m):
      mu, partial = x+u, self.R.p*u
      std = math.sqrt(abs(partial))
      N = stats.norm(mu, std)
      a = N.pdf(np.arange(self.m))
      _Pxxu[x] = a/np.sum(a)
    M = np.asmatrix(_Pxxu).T
    return M

  def correction(self, z):
    if z > len(self.Pz):
      Pc = np.ones(len(self.P))
    else:
      s = np.sum(self.Pz[z])
      if s == 0:
        Pc = np.ones(len(self.P))
      else:
        Pc = np.multiply(self.Pz[z], self.P)
    self.P = Pc/np.sum(Pc)

  def prediction(self, u, d):
    #P_act = self.Pxxu[self.R.get(u*d)]
    P_act = self.action(u*d)
    P = np.asmatrix(self.P).T
    M = np.matmul(P_act, P)
    self.P = np.asarray(M).flatten()

  # Arguments:
  #  pos - True position (in cells).
  def start_simulation(self, pos):
    print("Starting simulation at position: " + str(pos))
    self.pos = pos
    self.simulate = True

  def stop_simulation(self):
    print("Stopping simulation...")
    self.pos, self.simulate = None, None

  def simulate_sensor(self):
    i = self.M[self.pos]
    return int(round(self.Mu[i]+abs(self.N[i].rvs()-self.Mu[i])))

  # <robot>

  # Attach Bot to this Map.
  def attach(self, B):
    self.B = B

  def sensor(self):
    try:
      self.B.send(SEND)
    except Exception as e:
      print(e)
    b = self.B.recv(dtype='i')
    return int(b)

  def send_move(self, u, d):
    td = int(round(d*self.d))
    try:
      self.B.send(RECV)
      self.B.send(u)
      self.B.send(td)
    except Exception as e:
      print(e)

  # Detach and shutdown Bot.
  def detach(self):
    try:
      self.B.send(QUIT)
    except Exception as e:
      print(e)
    self.B = None

  # </robot>

  def move(self, u, d, corr, pred):
    if corr:
      if self.simulate:
        z = self.simulate_sensor()
      else:
        z = self.sensor()
      self.correction(z)
    if self.simulate:
      if u*d != 0:
        mu = self.pos+u*d
        std = abs(u*d*self.R.p)
        dp = int(round(stats.norm(mu, math.sqrt(std)).rvs()))
        self.pos = min(max(0, dp), self.m-1)
    else:
      self.send_move(u, d)
    if pred and u*d != 0:
      self.prediction(u, d)

  def print(self):
    print("Map Properties:")
    print("  Sensor Gaussians:")
    for i in range(len(self.Mu)):
      print("    Means: " + str(self.Mu[i]) + " | Variance: " + str(self.Var[i]))
    print("  Discretization bin size: " + str(self.m))
    print("  True size of each bin: " + str(self.d))
    print("  Size of precomputed probability matrices:")
    print("    P(Z|X)    (sensor probability distribution): " + str(self.Pz.shape))
    # print("    P(X'|X,u) (action probability distribution): " + str(len(self.Pxxu)) + " x " + str(self.Pxxu[0].shape))
    print("  Bot attached? " + str(self.B != None))
    print("Range Properties:")
    print("  Unique commands available: " + str(self.R.C))
    print("  Bounds for number of steps: " + str(self.R.M))
    print("  Pivot: " + str(self.R.pivot))
    print("  Variance proportion for action probability: " + str(self.R.p))

# Arguments:
#  n - Number of entries in map.
#  mu - Mean.
#  sigma - Variance.
#  size - Number of samples for gaussian sampling.
#  uniform - Whether the distribution should be uniform. False means gaussian.
def gen_init_pdist(n, mu=100, sigma=40, size=100000, uniform=False):
  if uniform:
    return np.ones(n)/n
  N = stats.norm(mu, math.sqrt(sigma))
  R = N.rvs(size=size)
  R = R[R>=0]
  R = R[R<n]
  R = np.bincount(np.rint(R).astype(int))
  m = n-len(R)
  if m > 0:
    R = np.pad(R, (0, m), mode="constant", constant_values=0)
  return R/np.sum(R)

# Arguments:
#  b_mu, b_sigma - Box's gaussian's mean and variance.
#  g_mu, g_sigma - Gap's gaussian's mean and variance.
#  C - Configuration. Entries are distances until next object (box or gap). Always alternated.
#  starts_with - Whether to start counting with a 'gap' or a 'box'.
#  bin_size - How many bins for discretization.
def new_config(b_mu, b_sigma, g_mu, g_sigma, C, starts_with='gap', bin_size=1000):
  D = np.sum(C)
  bD = D/bin_size
  M = []
  d, j = 0, 0
  s = 0 if starts_with == 'gap' else 1
  for i in range(bin_size):
    M.append(abs(s))
    d += bD
    if d >= C[j]:
      d, j = 0, j+1
      s = ~s # switch bit
  N_b, N_g = stats.norm(b_mu, math.sqrt(b_sigma)), stats.norm(g_mu, math.sqrt(g_sigma))
  return M, [N_g, N_b], bD

# Read controller commands.
def read(M):
  read_map = {'h': -1, 'j': 0, 'k': 0, 'l': 1}
  s = ''
  print("Waiting for input.")
  while True:
    c = getch()
    if c == '?':
      print("-------------------------")
      print("This controller works very much like vim. Available commands are:")
      print("  h - Go left and apply correction and prediction.")
      print("  j - No-op. Don't move, but apply correction and prediction.")
      print("  l - Go right and apply correction and prediction.")
      print("  k - Force correction only, with no prediction.")
      print("  c - Shows the current model's constraints and localization settings.")
      print("  q - Quit.")
      print("  ? - Show this help message")
      print("Capitalized equivalents apply only prediction with no correction:")
      print("  H - Go left and apply only prediction.")
      print("  J - No-op. Don't move, but apply prediction only.")
      print("  K - No-op. Applies correction. The same as k.")
      print("  L - Go right and apply only prediction.")
      print("Every command can be quantified (just like vim!). A number before a command means "+\
            "the command should be repeated that many times. For example:")
      print("  2l  - Go right two units and then apply correction and prediction.")
      print("  10H - Go left ten units and then apply only prediction.")
      print("  j   - Compute prediction and correction values and don't move.")
      print("  5k  - Compute correction values five times.")
      print("When omitting a quantifier, the command assumes the quantifier is 1.")
      print("-------------------------")
      continue
    elif c == 'c':
      M.print()
      continue
    _c = c.lower()
    if _c in read_map and not s:
      return read_map[c], 1, c.islower(), _c != 'k', False
    if c.isdigit():
      s += c
    elif _c in read_map:
      d = int(s)
      u = read_map[_c]
      return u, d, c.islower(), _c != 'k', False
    elif c == 'q':
      print("Bye.")
      return -1, -1, False, False, True
    else:
      print("Error when parsing command: " + s + c + ". Try again.")

# Draw graph plot.
def draw_graph(K, r, simulate):
  P, M = K.P, K.M
  if simulate:
    pos = K.pos
  m = len(M)
  plt.clf()
  plt.axis([0, m, 0, 1])
  B = plt.bar(np.arange(m), np.ones(m))
  for i in range(m):
    B[i].set_color('blue' if (simulate and pos == i) else 'green' if M[i] == 0 else 'orange')
  plt.draw()
  B = plt.bar(np.arange(m), P)
  for i in range(m):
    B[i].set_color('yellow' if M[i] == 0 else 'red')
  plt.draw()
  if not r:
    plt.pause(0.05)
  else:
    pause_nofocus(0.05)
  return True

# Start.
def start(M, simulate, pos):
  cmds = ['noop', 'right', 'left']
  if simulate:
    M.start_simulation(pos)
  else:
    B = init_bot()
    M.attach(B)
  plt.ion()
  running = False
  print("Ready. Press ? for help message.")
  while True:
    running = draw_graph(M, running, simulate)
    u, d, c, p, e = read(M)
    if e:
      break
    print("Moving " + cmds[u] + " " + str(d) + " units...")
    if simulate:
      print("  True position before moving: " + str(M.pos))
    M.move(u, d, c, p)
    if simulate:
      print("  True position after moving: " + str(M.pos))
  if simulate:
    M.stop_simulation()
  else:
    M.detach()

def init_bot():
  r_exc = False
  try:
    B = next(USBInterface.find_bricks(debug=False))
    B.connect()
    print("Bot found.")
  except usb.core.NoBackendError:
    print("Bot not found.")
    r_exc = True
  assert r_exc == 0, "No NXT found..."
  return B

def run(simulate):
  # D=[88, 187, 300, 415, 446, 549], w=30
  D = [51, 31, 72, 62, 111, 30, 54, 30, 69, 30, 85]
  C, N, d = new_config(33, 9, 61, 9, D, bin_size=int(np.sum(D)/5))
  print("Step distance: ", d)
  # Uniform initial belief.
  U = gen_init_pdist(len(C), uniform=True)
  # Gaussian initial belief centered on second box.
  G1 = gen_init_pdist(len(C), mu=20, sigma=math.sqrt(40))
  # Gaussian initial belief centered on fourth box.
  G2 = gen_init_pdist(len(C), mu=47, sigma=math.sqrt(40))
  M = Map(C, N, d, U, 3, Range([-1, 0, 1], [-100, 0, 100], 0.25))
  start(M, simulate, 5)

def parse_args():
  if '-s' in sys.argv or '--simulate' in sys.argv:
    return True
  return False

if __name__ == '__main__':
  run(parse_args())
