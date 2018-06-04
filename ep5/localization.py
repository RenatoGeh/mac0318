import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib
import math

import USBInterface
import usb

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
NOOP = '\x00'
SEND = '\x01'
RECV = '\x02'
QUIT = '\x03'

class Range:
  # Arguments:
  #  C - Commands (in positional increments), e.g. [-1, 0, 1] (left, noop, right).
  #  M - Maximum steps for each command, e.g. [-25, 0, 25] (-25, 0, 25) steps.
  def __init__(self, C, M):
    self.C = C
    self.M = M
    self.R = []
    for i, c in enumerate(C):
      if c != 0:
        self.R.extend(np.arange(0, M[i], c))
    self.R = np.unique(self.R)
    self.p = np.abs(np.min(M))-1 # pivot
    self.N = self.R+self.p # normalized range

  # Range.get(k) == normalized_range[pivot+k] == index of k (k can be negative).
  def get(self, i):
    return self.N[self.p+i]

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
    # Precompute matrices.
    self.precompute_sensor()
    self.precompute_action()

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
        _Pz[int(round(s))][x] = self.N[t].pdf(s)
    self.Pz = np.asarray(_Pz)

  def precompute_action(self):
    # Pxxu[u] = P(X'|X,u)
    # Pxxu[u][x] = P(X'=x'|X,u)
    # Pxxu[u][x][x'] = P(X'=x'|X=x,u)
    _Pxxu = [[[0 for i in range(self.m)] for j in range(self.m)] for l in range(self.c)]
    self.Pxxu = [None for i in range(self.c)]
    for u in range(self.c):
      for x in range(self.m):
        N = stats.norm(x+self.C[u])
        a = N.pdf(np.arange(self.m))
        _Pxxu[u][x] = a/np.sum(a)
      _Pxxu[u] = np.asarray(_Pxxu[u])
      self.Pxxu[u] = np.asmatrix(_Pxxu[u]).T

  def correction(self, z):
    Pc = np.multiply(self.Pz[z], self.P)
    self.P = Pc/np.sum(Pc)

  def prediction(self, u, d):
    self.P = np.asarray(np.dot(self.Pxxu[self.R.get(u*d)], self.P)).flatten()

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
    b = B.recv()
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

  def move(self, u, d, corr):
    if corr:
      if self.simulate:
        z = self.simulate_sensor()
      else:
        z = self.sensor()
      self.correction(z)
    if self.simulate:
      self.pos += u*d
    else:
      self.send_move(u, d)
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
    print("    P(X'|X,u) (action probability distribution): " + str(len(self.Pxxu)) + " x " + str(self.Pxxu[0].shape))
    print("  Bot attached? " + str(self.B != None))
    print("Range Properties:")
    print("  Unique commands available: " + str(self.R.C))
    print("  Bounds for number of steps: " + str(self.R.M))
    print("  Pivot: " + str(self.R.p))

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
  read_map = {'j': -1, 'k': 0, 'l': 1}
  s = ''
  while True:
    c = getch()
    if c == 'h':
      print("-------------------------")
      print("This controller works very much like vim. Available commands are:")
      print("  j - Go left and apply correction and prediction.")
      print("  k - No-op. Don't move, but apply correction and prediction.")
      print("  l - Go right and apply correction and prediction.")
      print("  i - Force correction only, with no prediction.")
      print("  c - Shows the current model's constraints and localization settings.")
      print("  q - Quit.")
      print("  h - Show this help message")
      print("Capitalized equivalents apply only prediction with no correction:")
      print("  J - Go left and apply only prediction.")
      print("  K - No-op. Don't move, but apply prediction only.")
      print("  L - Go right and apply only prediction.")
      print("Every command can be quantified (just like vim!). A number before a command means "+\
            "the command should be repeated that many times. For example:")
      print("  2j  - Go right two units and then apply correction and prediction.")
      print("  10L - Go left ten units and then apply only prediction.")
      print("  1k  - Compute prediction and correction values and don't move.")
      print("  5i  - Compute correction values five times.")
      print("When omitting a quantifier, the command assumes the quantifier is 1.")
      print("-------------------------")
      continue
    elif c == 'c':
      M.print() 
      continue
    _c = c.lower()
    if _c in read_map and not s:
      return read_map[c], 1, c.islower(), False
    if c.isdigit():
      s += c
    elif _c in read_map:
      d = int(s)
      u = read_map[_c]
      return u, d, c.islower(), False
    elif c == 'q':
      print("Bye.")
      return -1, -1, False, True
    else:
      print("Error when parsing command: " + s + c + ". Try again.")

# Draw graph plot.
def draw_graph(P, M, pos, r):
  m = len(M)
  plt.clf()
  plt.axis([0, m, 0, 1])
  B = plt.bar(np.arange(m), P)
  for i in range(m):
    B[i].set_color('blue' if pos == i else 'yellow' if M[i] == 0 else 'red')
  plt.draw()
  if not r:
    plt.pause(0.05)
  else:
    pause_nofocus(0.05)
  return True

# Start.
def start(M, simulate, p):
  cmds = ['noop', 'right', 'left']
  if simulate:
    M.start_simulation(p)
  else:
    B = init_bot()
    M.attach(B)
  plt.ion()
  running = False
  print("Ready. Press h for help message.")
  while True:
    running = draw_graph(M.P, M.M, M.pos, running)
    u, d, c, e = read(M)
    if e:
      break
    print("Moving " + cmds[u] + " " + str(d) + " units...")
    if simulate:
      print("  True position before moving: " + str(M.pos))
    M.move(u, d, c)
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
  except usb.core.NoBackendError:
    r_exc = True
  assert r_exc == 0, "No NXT found..."
  return B

def run():
  C, N, d = new_config(15, 1, 10, 1, [5, 10, 15, 10, 20, 10, 30, 10, 15, 10, 20, 10, 5], bin_size=100)
  # Uniform initial belief.
  U = gen_init_pdist(len(C), uniform=True)
  # Gaussian initial belief centered on second box.
  G1 = gen_init_pdist(len(C), mu=20, sigma=math.sqrt(40))
  # Gaussian initial belief centered on fourth box.
  G2 = gen_init_pdist(len(C), mu=60, sigma=math.sqrt(40))
  M = Map(C, N, d, G1, 3, Range([-1, 0, 1], [-25, 0, 25]))
  start(M, True, 5)

if __name__ == '__main__':
  run()
