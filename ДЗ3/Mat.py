import numpy as np
from fractions import Fraction
from functools import reduce
import itertools
import sympy
import copy
import math
import time
import random

random.seed(time.time())


def lcm(a, b):
    return abs(a * b) // math.gcd(a, b)


def lcm_multiple(numbers):
    return reduce(lcm, numbers)


class Mat():
    __slots__ = ["_arr", "_slu", "_type"]
    # types:
    # mat, perm

    def __init__(self, arr: np.ndarray, slu: int = -1, type="mat"):
        self._slu = slu
        self._type = type

        if len(np.array(arr).shape) != 2:
            raise Exception("Not matrix.")

        if self._type == "mat":
            for i in range(len(arr)):
                for j in range(len(arr[0])):
                    if isinstance(arr[i][j], sympy.core.Basic):
                        continue
                    arr[i][j] = Fraction(arr[i][j])
        if self._type == "perm":
            pass

        self._arr = np.array(arr)

    def __str__(self):
        if self._type == "mat":
            s = ""
            for i, x in enumerate(self._arr):
                for j, y in enumerate(x):
                    if j == self._slu:
                        s += "| "
                    s += str(sympy.expand(y))+" "
                s += "\n"
            return s
        if self._type == "perm":
            s = ""
            for i, x in enumerate(self._arr):
                for j, y in enumerate(x):
                    if j == self._slu:
                        s += "| "
                    s += str(y)+" "
                s += "\n"
            return s+''.join([f"({' '.join(str(y) for y in x)})" for x in self.cicles if len(x) > 1])

    def __mul__(self, other):
        if self._type == "mat":
            if isinstance(other, Mat):
                return Mat(np.dot(self._arr, other._arr))
            return Mat(self._arr * other)
        elif self._type == "perm":
            new_y = []
            for el in other._arr[1]:
                new_y.append(self(el))
            return Mat([other._arr[0], new_y], type="perm")

    def __rmul__(self, other):
        if self._type == "mat":
            if isinstance(other, Mat):
                return Mat(np.dot(other._arr, self._arr))
            return Mat(self._arr * other)
        elif self._type == "perm":
            new_y = []
            for el in self._arr[1]:
                new_y.append(other(el))
            return Mat([self._arr[0], new_y], type="perm")

    def __truediv__(self, other):
        if self._type == "mat":
            return Mat(self._arr * Fraction(1, other))

    def __add__(self, other):
        if isinstance(other, Mat):
            return Mat(self._arr + other._arr)
        return Mat(self._arr + other)

    def __radd__(self, other):
        if isinstance(other, Mat):
            return Mat(self._arr + other._arr)
        return Mat(self._arr + other)

    def __sub__(self, other):
        if isinstance(other, Mat):
            return Mat(self._arr - other._arr)
        return Mat(self._arr - other)

    def __rsub__(self, other):
        if isinstance(other, Mat):
            return Mat(other._arr-self._arr)
        return Mat(other - self._arr)

    def __pos__(self):
        return Mat(self._arr)

    def __neg__(self):
        return Mat(-self._arr)

    def __len__(self):
        return len(self._arr)

    def __pow__(self, n):
        if self._type == "mat":
            return Mat(np.linalg.matrix_power(self._arr, n))
        if self._type == "perm":
            if n == -1:
                return self**(lcm_multiple([len(x) for x in self.cicles])-1)
            new_x = copy.copy(self._arr[0])
            for _ in range(n):
                for i, el in enumerate(new_x):
                    new_x[i] = self(el)
            return Mat([self._arr[0], new_x], type="perm")

    def __getitem__(self, index):
        return self._arr[index]

    def __setitem__(self, index, value):
        self._arr[index] = value

    def swap_rows(self, ind1, ind2):
        row1 = copy.copy(self._arr[ind1])
        self._arr[ind1] = self._arr[ind2]
        self._arr[ind2] = row1
    
    def swap_cols(self, ind1, ind2):
        row1 = copy.copy(self._arr[:, ind1])
        self._arr[:, ind1] = self._arr[:, ind2]
        self._arr[:, ind2] = row1

    def check_slu_1(self, *xs):
        if self._slu == -1:
            raise Exception("Not SLU.")
        if len(self._arr[0]) - self._slu != 1:
            raise Exception("Not SLU-1")
        if len(xs) != len(self._arr[0])-1:
            raise Exception("The number of x's does not match.")

        for row in self._arr:
            s = 0
            for i in range(len(xs)):
                s += xs[i]*row[i]
            if s != row[-1]:
                return False
        return True

    def __call__(self, x):
        if self._type == "perm":
            return self._arr[1][list(self._arr[0]).index(x)]

    def __eq__(self, other):
        return str(self) == str(other)

    @property
    def det(self):
        if isinstance(self, Mat) and self._arr.shape[0] == self._arr.shape[1]:
            n = self._arr.shape[0]
            s = 0
            for perm in itertools.permutations(list(range(n))):
                mul = 1
                for i in range(n):
                    mul *= self[i][perm[i]]
                s = s + Mat([list(range(n)), perm], type="perm").sgn * mul
            return sympy.expand(s)

    @staticmethod
    def random(n, m=-1,  min=-100, max=100, type="Mat"):
        if type == "perm":
            a = [x for x in range(1, n+1)]
            random.shuffle(a)
            return Mat([[x for x in range(1, n+1)], a], type="perm")
        if type == "Mat":
            if m == -1:
                return Mat(np.random.randint(min, max+1, (n, n)))
            else:
                return Mat(np.random.randint(min, max+1, (n, m)))

    @property
    def inv_count(self):
        x = self._arr[0]
        y = self._arr[1]
        c = 0
        for i in range(len(x)):
            for j in range(i+1, len(y)):
                if (x[i] - x[j] >= 0 and y[i] - y[j] < 0) or \
                        (x[i] - x[j] < 0 and y[i] - y[j] >= 0):
                    c += 1
        return c

    @property
    def sgn(self):
        return (-1)**self.inv_count

    @property
    def cicles(self):
        if self._type != "perm":
            raise Exception("Not permutation.")
        x = self._arr[0]
        y = self._arr[1]
        cicles = []
        used = set()
        for i, el in enumerate(x):
            cicle = [el]
            ci = y[i]
            while ci != el and i < len(x):
                cicle.append(ci)
                i = y[int(i)]-1
                ci = y[int(i)]

            if cicle[0] not in used:
                cicles += [cicle]
            for el in cicle:
                used.add(el)
        return cicles

    @property
    def T(self):
        return Mat(self._arr.T)

    @property
    def tr(self):
        return np.trace(self._arr)
