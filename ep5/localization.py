import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib
import math

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

class Map:
  # Arguments:
  #  M - Map (each entry is an index for the gaussian array).
  #  N - Gaussians for each 'color'.
  #  d - Distance between cells.
  #  P - Initial belief probability distribution.
  #  p - Precision (how many standard deviations from the mean should we precompute).
  #  C - Commands (in positional increments).
  def __init__(self, M, N, d, P, p, C):
    self.M, self.m = M, len(M)
    self.N, self.n = N, len(N)
    self.P = np.asarray(P)
    # Store means, std and vars.
    self.Mu, self.Sigma, self.Var = [0 for i in range(self.n)], [0 for i in range(self.n)], [0 for i in range(self.n)]
    for i in range(self.n):
      self.Mu[i], self.Sigma[i], self.Var[i] = self.N[i].mean(), self.N[i].std(), self.N[i].var()
    self.p = p
    self.C, self.c = C, len(C)
    self.d = d
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

  def predict(self, u):
    self.P = np.asarray(np.dot(self.Pxxu[u], self.P)).flatten()

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

  def sensor(self):
    return -1

  def send_move(self, d):
    return

  # </robot>

  def move(self, u, d):
    for i in range(d):
      if self.simulate:
        z = self.simulate_sensor()
      else:
        z = self.sensor()
      self.correction(z)
      if self.simulate:
        self.pos += self.C[u]
      else:
        self.send_move(u)
      self.predict(u)


# Arguments:
#  n - Number of entries in map.
#  mu - Mean.
#  sigma - Variance.
#  size - Number of samples for gaussian sampling.
#  uniform - Whether the distribution should be uniform. False means gaussian.
def gen_init_pdist(n, mu=100, sigma=40, size=10000, uniform=False):
  if uniform:
    return np.ones(n)/n
  N = stats.norm(mu, math.sqrt(sigma))
  R = N.rvs(size=size)
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

def read():
  read_map = {'j': 0, 'k': 1, 'l': 2}
  s = ''
  while True:
    c = getch()
    if c == 'h':
      print("This controller works very much like vim. Available commands are:")
      print("  j - Go left.")
      print("  k - No-op. Don't move, but still compute values.")
      print("  l - Go right.")
      print("  q - Quit.")
      print("  h - Show this help message")
      print("Every command can be quantified (just like vim!). A number before a command means "+\
            "the command should be repeated that many times. For example:")
      print("  2j  - Go right and compute values twice sequentially.")
      print("  10l - Go left and compute values ten times.")
      print("  1k  - Compute values once and don't move.")
      print("When omitting a quantifier, the command assumes the quantifier is 1.")
      continue
    if c in read_map and not s:
      return read_map[c], 1, False
    if c.isdigit():
      s += c
    elif c in read_map:
      d = int(s)
      u = read_map[c]
      return u, d, False
    elif c == 'q':
      print("Bye.")
      return -1, -1, True
    else:
      print("Error when parsing command: " + s + c + ". Try again.")

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

def start(M, simulate, p):
  cmds = ['left', 'noop', 'right']
  if simulate:
    M.start_simulation(p)
  plt.ion()
  running = False
  print("Ready. Press h for help message.")
  while True:
    u, d, e = read()
    if e:
      break
    print("Moving " + cmds[u] + " " + str(d) + " units...")
    if simulate:
      print("  True position before moving: " + str(M.pos))
    M.move(u, d)
    if simulate:
      print("  True position after moving: " + str(M.pos))
    running = draw_graph(M.P, M.M, M.pos, running)
  if simulate:
    M.stop_simulation()

def run():
  C, N, d = new_config(15, 1, 10, 1, [5, 10, 15, 10, 20, 10, 30, 10, 15, 10, 20, 10, 5], bin_size=100)
  M = Map(C, N, d, gen_init_pdist(len(C), uniform=True), 3, [-1, 0, 1])
  start(M, True, 5)

if __name__ == '__main__':
  run()
