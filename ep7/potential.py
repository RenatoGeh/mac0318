import cv2
import collections
import heapq
import numpy as np
import numpy.matlib as mat
import numpy.linalg as linalg
import scipy.ndimage.morphology as morph
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import math

import USBInterface
import usb
import sys

class Map:
  def __init__(self, img_path, d, bx, by, B):
    self.O = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)/255
    self.w_o, self.h_o = self.O.shape
    self.d = d
    self.dlt_struct = ndimage.generate_binary_structure(2, d).astype(int)
    self.I = self.dilate(self.O)
    self.M = self.discretize(self.I, bx, by)
    self.L = self.find_lines(self.M)
    self.L_o = self.find_lines(self.I)
    self.B = B

  def discretize(self, I, bx, by):
    self.w, self.h = int(self.w_o/bx), int(self.h_o/by)
    self.bx, self.by = bx, by
    _, M = cv2.threshold(cv2.resize(I.astype("float64"), (self.h, self.w)), np.max(I)/2, 1, cv2.THRESH_BINARY)
    return M.astype(int)

  def dilate(self, M):
    return 1-morph.binary_dilation(~(M.astype(bool)), self.dlt_struct, iterations=self.d).astype(int)

  def find_lines(self, M):
    I = (255*M).astype("uint8")
    R, C, _ = cv2.findContours(I, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    L = []
    for i, c in enumerate(C):
      f, p = None, None
      for j, _ in enumerate(c):
        v = c[j][0]
        if p is None:
          p = v
          f = p
          continue
        L.append((p[0], p[1], v[0], v[1]))
        p = v
      L.append((p[0], p[1], f[0], f[1]))
    return L

  @staticmethod
  def Tan(p1, p2):
    dx, dy = p1[0]-p2[0], p1[1]-p2[1]
    return math.tan((p1[0]-p2[0])/(p1[1]-p2[1]))

  @staticmethod
  def Orientation(p1, p2, p3):
    # Clockwise = 1
    # Counter-clockwise = -1
    # Colinear = 0
    return np.sign((p2[1]-p1[1])*(p3[0]-p2[0])-(p2[0]-p1[0])*(p3[1]-p2[1]))

  @staticmethod
  def Colinear(p1, p2, p3):
    return p2[0] <= max(p1[0], p3[0]) and p2[0] >= min(p1[0], p3[0]) and \
      p2[1] <= max(p1[1], p3[1]) and p2[1] >= min(p1[1], p3[1])

  @staticmethod
  def Intersects(l1, l2):
    # Colinear check.
    # x11, x12 = l1[0], l1[2]
    # x21, x22 = l2[0], l2[2]
    # x1_max, x1_min = min(x11, x12), max(x11, x12)
    # x2_max, x2_min = min(x21, x22), max(x21, x22)
    # y11, y12 = l1[1], l1[3]
    # y21, y22 = l2[1], l2[3]
    # y1_max, y1_min = min(y11, y12), max(y11, y12)
    # y2_max, y2_min = min(y21, y22), max(y21, y22)

    # x_cond = (x1_min <= x21 <= x1_max) or (x1_min <= x22 <= x1_max) or \
        # (x2_min <= x11 <= x2_max) or (x2_min <= x12 <= x2_max)
    # y_cond = (y1_min <= y21 <= y1_max) or (y1_min <= y22 <= y1_max) or \
        # (y2_min <= y11 <= y2_max) or (y2_min <= y12 <= y2_max)
    # if x_cond and y_cond:
      # return True
    p1, p2, p3, p4 = (l1[0], l1[1]), (l1[2], l1[3]), (l2[0], l2[1]), (l2[2], l2[3])
    o1, o2 = Map.Orientation(p1, p2, p3), Map.Orientation(p1, p2, p4)
    o3, o4 = Map.Orientation(p3, p4, p1), Map.Orientation(p3, p4, p2)
    if o1 != o2 and o3 != o4:
      return True
    if (o1 == 0 and Map.Colinear(p1, p3, p2)) or (o2 == 0 and Map.Colinear(p1, p4, p2)):
      return True
    if (o3 == 0 and Map.Colinear(p3, p1, p4)) or (o4 == 0 and Map.Colinear(p3, p2, p4)):
      return True
    return False

  @staticmethod
  def __bin_search(P, p, L):
    n = len(P)-1
    x1, y1 = p
    x2, y2 = P[n]
    pivot = int(n/2)
    inter = False
    for c in L:
      if Map.Intersects((x1, y1, x2, y2), c):
        inter = True
        break
    if inter:
      return Map.__bin_search(P[:pivot], p)
    return Map.__bin_search(P[pivot:], p)

  @staticmethod
  def __greedy_search(P, p, L):
    n = len(P)
    x1, y1 = p
    for i in range(n-1, -1, -1):
      x2, y2 = P[i]
      inter = False
      for c in L:
        if Map.Intersects((x1, y1, x2, y2), c):
          inter = True
          break
      if not inter:
        return i
    return 0

  def linearize(self, P, C):
    # P is path (set of points).
    _P = P[1:]
    L = []
    s = 0
    p = P[0]
    while len(_P) > 0:
      i = Map.__greedy_search(_P, p, C)
      # i = Map.__bin_search(_P, p, C)
      q = _P[i]
      L.append((p[0], p[1], q[0], q[1]))
      _P = _P[i+1:]
      p = q
    return L

  def orig_pos(self, x, y=None):
    if y is None:
      return self.bx*x[0], self.by*x[1]
    return self.bx*x, self.by*y

  def pos(self, x, y=None):
    if y is None:
      return int(x[0]/self.bx), int(x[1]/self.by)
    return int(x/self.bx), int(y/self.by)

  def neighborhood(self, r, x, y):
    M = []
    w, h = self.w-1, self.h-1
    if x > 0:
      M.append((x-1, y))
    if x < w:
      M.append((x+1, y))
    if y > 0:
      M.append((x, y-1))
    if y < h:
      M.append((x, y+1))
    if r == 8:
      if x > 0:
        if y > 0:
          M.append((x-1, y-1))
        if y < h:
          M.append((x-1, y+1))
      if x < w:
        if y > 0:
          M.append((x+1, y-1))
        if y < h:
          M.append((x+1, y+1))
    return M

  def wavefront(self, sx, sy, tx, ty):
    sx, sy, tx, ty = tx, ty, sx, sy
    t = 1
    Q = collections.deque()
    T = np.zeros(self.M.shape)
    x, y = sx, sy
    # Forward wave.
    while True:
      N = self.neighborhood(4, x, y)
      for p in N:
        if p == (sx, sy):
          T[p] = 0
          continue
        if self.M[p] != 0 and T[p] == 0:
          T[p] = t
          Q.append(p)
          if p == (tx, ty):
            break
      x, y = Q.popleft()
      if (x, y) == (tx, ty):
        break
      t += 1
    # Backtrack.
    Q = []
    x, y = tx, ty
    while (x, y) != (sx, sy):
      N = self.neighborhood(8, x, y)
      m, im = t+1, None
      for p in N:
        if p == (sx, sy):
          im = p
          break
        if T[p] != 0 and T[p] < m:
          m, im = T[p], p
      Q.append((im[1], im[0]))
      x, y = im
    return Q

  @staticmethod
  def Manhattan(px, py, qx=None, qy=None):
    if qx is None and qy is None:
      x1, y1 = px
      x2, y2 = py
    else:
      x1, y1, x2, y2 = px, py, qx, qy
    return abs(x1-x2)+abs(y1-y2)

  @staticmethod
  def Euclidean(px, py, qx=None, qy=None):
    if qx is None and qy is None:
      x1, y1 = px
      x2, y2 = py
    else:
      x1, y1, x2, y2 = px, py, qx, qy
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

  def best_choice(self, sx, sy, tx, ty, mfunc):
    if mfunc is None:
      mfunc = Map.Manhattan
    r = 8 if mfunc == Map.Euclidean else 4
    Q = []
    V = np.zeros(self.M.shape)
    heapq.heappush(Q, (mfunc(sx, sy, tx, ty), (sx, sy)))
    t, g = 1, None
    while True:
      q = heapq.heappop(Q)
      x, y = q[1]
      N = self.neighborhood(r, x, y)
      for p in N:
        if V[p] == 0 and self.M[p] != 0:
          if p == (tx, ty):
            g = p
            break
          heapq.heappush(Q, (mfunc(p[0], p[1], tx, ty), p))
          V[p] = t
      t += 1
      if g is not None:
        break
    Q = []
    x, y = g
    while (x, y) != (sx, sy):
      N = self.neighborhood(8, x, y)
      m, im = t+1, None
      for p in N:
        if p == (sx, sy):
          im = p
          break
        if V[p] != 0 and V[p] < m:
          m, im = V[p], p
      Q.append((im[1], im[0]))
      x, y = im
    return Q

  def f_att(self, p, g, k_att):
    return -k_att*(p-g)

  @staticmethod
  def LineEq(a, b, t):
    return a+t*(b-a)

  @staticmethod
  def Dist2Line(a, b, p):
    u = np.matrix(b-a)
    v = np.matrix(p-a)
    q = u.T
    t = ((v*q)/(u*q))[0,0]
    if t < 0:
      return a, linalg.norm(p-a)
    if t > 1:
      return b, linalg.norm(p-b)
    o = Map.LineEq(a, b, t)
    return o, linalg.norm(p-o)

  def f_rep(self, k_rep, rho, p, C):
    f = np.zeros(2)
    for c in C:
      u, v = c[0], c[1]
      o, d = Map.Dist2Line(u, v, p)
      if d <= rho:
        n = (k_rep*(rho-d))/((d**4)*rho)
        f += n*(p-o)
    return f

  def potential(self, k_att, k_rep, alpha, rho, eps, sx, sy, tx, ty, move_func=None):
    C = []
    P = [(sy, sx)]
    for c in self.L_o:
      C.append((np.array([c[0], c[1]]), np.array([c[2], c[3]])))
    p = np.array([sy, sx])
    g = np.array([ty, tx])
    f = np.inf
    last = np.copy(p)
    while linalg.norm(f) > eps:
      f = self.f_att(p, g, k_att) + self.f_rep(k_rep, rho, p, C)
      df = alpha*f
      p = p+df
      if not np.array_equal(last.astype(int), p.astype(int)):
        P.append(p)
        last = np.copy(p)
      if move_func is not None:
        move_func(int(df), self.B)
    P = list(np.unique(P, axis=0).astype(int))
    return P

  def open_raw(self, img_path):
    with open(img_path) as img_f:
      nc = 0
      x, y = 0, 0
      while True:
        l = img_f.readline()
        if l == "":
          break
        if l.strip() == "" or l[0] == '#' or l[0].isalpha():
          continue
        v = [int(x) for x in l.split()]
        if nc == 0:
          self.w, self.h = v[0], v[1]
          self.M = mat.zeros((self.w, self.h))
          nc = 2
        elif nc == 2:
          self.max = v[0]
          nc = 3
        else:
          for _, p in enumerate(v):
            self.M[x,y] = p
            nc += 1
            i = nc-3
            x, y = i%self.w, int(i/self.w)

  def show_path(self, f, sx, sy, tx, ty):
    dsx, dsy = self.pos(sx, sy)
    dtx, dty = self.pos(tx, ty)
    P = f(dsx, dsy, dtx, dty)
    I = (self.O*255).astype("uint8")
    for p in P:
      I[self.orig_pos(p[0], p[1])] = 122
    I[sx, sy] = 75
    I[tx, ty] = 200
    plt.imshow(I)
    plt.show()

  def show_wavefront(self, sx, sy, tx, ty, filename):
    dsx, dsy = self.pos(sx, sy)
    dtx, dty = self.pos(tx, ty)
    P = self.wavefront(dsx, dsy, dtx, dty)
    I = (self.O*255).astype("uint8")
    L = self.linearize(P, self.L)
    for l in L:
      p1 = self.orig_pos(l[0], l[1])
      p2 = self.orig_pos(l[2], l[3])
      cv2.line(I, p1, p2, 125, 1)
    for p in P:
      I[elf.orig_pos(p[1], p[0])] = 150
    I[sx, sy] = 75
    I[tx, ty] = 200
    cv2.imwrite(filename, I)

  def show_best_choice(self, sx, sy, tx, ty, mfunc, filename):
    dsx, dsy = self.pos(sx, sy)
    dtx, dty = self.pos(tx, ty)
    P = self.best_choice(dsx, dsy, dtx, dty, mfunc)
    I = (self.O*255).astype("uint8")
    L = self.linearize(P, self.L)
    for l in L:
      p1 = self.orig_pos(l[0], l[1])
      p2 = self.orig_pos(l[2], l[3])
      cv2.line(I, p1, p2, 125, 1)
    for p in P:
      I[self.orig_pos(p[1], p[0])] = 150
    I[sx, sy] = 75
    I[tx, ty] = 200
    cv2.imwrite(filename, I)

  def show_potential_field(self, k_att, k_rep, alpha, rho, eps, sx, sy, tx, ty, filename):
    P = self.potential(k_att, k_rep, alpha, rho, eps, sx, sy, tx, ty)
    I = (self.O*255).astype("uint8")
    _P = []
    for p in P:
      _P.append((p[1], p[0]))
    L = self.linearize(_P, self.L_o)
    for l in L:
      p1 = (l[1], l[0])
      p2 = (l[3], l[2])
      cv2.line(I, p1, p2, 125, 1)
    for p in _P:
      I[p[0], p[1]] = 150
    I[sx, sy] = 75
    I[tx, ty] = 200
    cv2.imwrite(filename, I)

  def show_lines(self):
    print("Number of lines:", len(self.L))
    img = (self.M*255).astype("uint8")
    for l in self.L:
      cv2.line(img, (l[0], l[1]), (l[2], l[3]), 125, 2)
    cv2.imwrite("contours.png", img)

  def show_original_lines(self):
    print("Number of lines:", len(self.L_o))
    img = (self.I*255).astype("uint8")
    for l in self.L_o:
      cv2.line(img, (l[0], l[1]), (l[2], l[3]), 125, 2)
    cv2.imwrite("contours.png", img)

  def show_preimages(self):
    print("Drawing matrix...")
    plt.subplot(131)
    plt.matshow(self.O, fignum=False)
    plt.title("Original")
    plt.subplot(132)
    plt.title("Discretized and Dilated")
    plt.matshow(self.M, fignum=False)
    plt.subplot(133)
    plt.matshow(self.I, fignum=False)
    plt.title("Dilated")
    plt.show()

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

def init_map():
  return Map("map.pgm", 5, 2, 2)

def move_potential(f, B):
  try:
    B.send(f)
  except Exception as e:
    print(e)

def start(B, M, p, q, t, P):
  if t == "potential":
    M.potential(P[0], P[1], P[2], P[3], P[4], p[0], p[1], q[0], q[1], move_potential)
  else:
    if t == "wavefront":
      L = M.linearize(M.wavefront(p[0], p[1], q[0], q[1]), M.L)
    else:
      L = M.linearize(M.best_choice(p[0], p[1], q[0], q[1], P[0]), M.L)
    try:
      B.send(len(L))
      for l in L:
        B.send(l[0])
        B.send(l[1])
        B.send(l[2])
        B.send(l[3])
    except Exception as e:
      print(e)

def run():
  B = init_bot()
  M = init_map()
  P_wave = []
  P_potential = [0.1, 10000, 0.01, 500, 1.0]
  P_best = [Map.Euclidean]
  p, q = (10, 10), (100, 100)
  start(B, M, p, q, "wavefront", P_wave)

def test():
  M = Map("map.pgm", 5, 2, 2)
  print("Wavefront...")
  print("Test case 1.")
  M.show_wavefront(10, 10, M.w_o-10, M.h_o-10, "wavefront_path_1.png")
  print("Test case 2.")
  M.show_wavefront(365, 61, 365, 495, "wavefront_path_2.png")
  print("Test case 3.")
  M.show_wavefront(61, 365, 495, 365, "wavefront_path_3.png")
  print("Test case 4.")
  M.show_wavefront(211, 555, 588, 268, "wavefront_path_4.png")
  print("Test case 5.")
  M.show_wavefront(555, 211, 268, 588, "wavefront_path_5.png")
  print("Best choice using Manhattan distance...")
  print("Test case 1.")
  M.show_best_choice(10, 10, M.w_o-10, M.h_o-10, M.Manhattan, "best_choice_manhattan_1.png")
  print("Test case 2.")
  M.show_best_choice(365, 61, 365, 495, M.Manhattan, "best_choice_manhattan_2.png")
  print("Test case 3.")
  M.show_best_choice(61, 365, 495, 365, M.Manhattan, "best_choice_manhattan_3.png")
  print("Test case 4.")
  M.show_best_choice(211, 555, 588, 268, M.Manhattan, "best_choice_manhattan_4.png")
  print("Test case 5.")
  M.show_best_choice(555, 211, 268, 588, M.Manhattan, "best_choice_manhattan_5.png")
  print("Best choice using Euclidean distance...")
  print("Test case 1.")
  M.show_best_choice(10, 10, M.w_o-10, M.h_o-10, M.Euclidean, "best_choice_euclidean_1.png")
  print("Test case 2.")
  M.show_best_choice(365, 61, 365, 495, M.Euclidean, "best_choice_euclidean_2.png")
  print("Test case 3.")
  M.show_best_choice(61, 365, 495, 365, M.Euclidean, "best_choice_euclidean_3.png")
  print("Test case 4.")
  M.show_best_choice(211, 555, 588, 268, M.Euclidean, "best_choice_euclidean_4.png")
  print("Test case 5.")
  M.show_best_choice(555, 211, 268, 588, M.Euclidean, "best_choice_euclidean_5.png")
  print("Potential field...")
  # M.show_potential_field(0.1, 10000, 0.1, 500, 0.1, 10, 10, 122, 604, "potential_field.png")
  print("Test case 1.")
  M.show_potential_field(0.1, 10000, 0.01, 500, 1.0, 365, 61, 365, 495, "potential_field_1.png")
  print("Test case 2.")
  M.show_potential_field(0.1, 10000, 0.01, 500, 1.0, 61, 365, 495, 365, "potential_field_2.png")
  print("Test case 3.")
  M.show_potential_field(0.1, 10000, 0.01, 500, 1.0, 211, 555, 588, 268, "potential_field_3.png")
  print("Test case 4.")
  M.show_potential_field(0.1, 10000, 0.01, 500, 1.0, 555, 211, 268, 588, "potential_field_4.png")

if __name__ == '__main__':
  run()
