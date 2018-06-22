import cv2
import collections
import numpy as np
import numpy.matlib as mat
import scipy.ndimage.morphology as morph
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import math

class Map:
  def __init__(self, img_path, d, bx, by):
    self.O = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)/255
    self.w_o, self.h_o = self.O.shape
    self.d = d
    self.dlt_struct = ndimage.generate_binary_structure(2, d).astype(int)
    self.I = self.dilate(self.O)
    self.M = self.discretize(self.I, bx, by)
    self.L = self.find_lines(self.M)

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

  def __bin_search(self, P, p):
    n = len(P)-1
    x1, y1 = p
    x2, y2 = P[n]
    pivot = int(n/2)
    inter = False
    for c in self.L:
      if Map.Intersects((x1, y1, x2, y2), c):
        inter = True
        break
    if inter:
      return self.__bin_search(P[:pivot], p)
    return self.__bin_search(P[pivot:], p)

  def __greedy_search(self, P, p):
    n = len(P)
    x1, y1 = p
    for i in range(n-1, -1, -1):
      x2, y2 = P[i]
      inter = False
      for c in self.L:
        if Map.Intersects((x1, y1, x2, y2), c):
          inter = True
          break
      if not inter:
        return i
    return 0

  def linearize(self, P):
    # P is path (set of points).
    _P = P[1:]
    L = []
    s = 0
    p = P[0]
    while len(_P) > 0:
      i = self.__greedy_search(_P, p)
      # i = self.__bin_search(_P, p)
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

  def show_linearized_path(self, f, sx, sy, tx, ty):
    dsx, dsy = self.pos(sx, sy)
    dtx, dty = self.pos(tx, ty)
    P = f(dsx, dsy, dtx, dty)
    I = (self.O*255).astype("uint8")
    print(sx, sy, tx, ty)
    L = self.linearize(P)
    for l in L:
      p1 = self.orig_pos(l[0], l[1])
      p2 = self.orig_pos(l[2], l[3])
      print(p1, p2)
      cv2.line(I, p1, p2, 125, 1)
    for p in P:
      I[self.orig_pos(p[1], p[0])] = 150
    I[sx, sy] = 75
    I[tx, ty] = 200
    plt.imshow(I)
    plt.show()
    cv2.imwrite('path.png', I)

  def show_lines(self):
    print("Number of lines:", len(self.L))
    img = (self.M*255).astype("uint8")
    for l in self.L:
      cv2.line(img, (l[0], l[1]), (l[2], l[3]), 125, 2)
    plt.imshow(img)
    plt.show()

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

def run():
  M = Map("map.pgm", 5, 2, 2)
  M.show_linearized_path(M.wavefront, 10, 10, M.w_o-10, M.h_o-10)

if __name__ == '__main__':
  run()
