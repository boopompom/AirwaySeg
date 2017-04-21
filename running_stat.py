import threading
import numpy as np


class RunningStat:

    # FIXME: See why variance sometimes ends up being negative
    def __init__(self):
        self.lock = threading.Lock()
        self.K = 0
        self.n = 0
        self.ex = 0
        self.ex2 = 0

    def add_batch(self, x):

        self.lock.acquire()

        flat_x = x.flatten()

        if self.n == 0:
            self.K = flat_x[0]

        x_k = flat_x - self.K
        x2_k = np.power(flat_x - self.K, 2)

        x_k_sum = np.sum(x_k)
        x2_k_sum = np.sum(x2_k)

        self.n += flat_x.shape[0]
        self.ex += x_k_sum
        self.ex2 += x2_k_sum

        self.lock.release()

    def add_variable(self, x):
        if self.n == 0:
            self.K = x
        self.n += 1
        self.ex += x - self.K
        self.ex2 += (x - self.K) * (x - self.K)

    def remove_variable(self, x):
        self.n -= 1
        self.ex -= (x - self.K)
        self.ex2 -= (x - self.K) * (x - self.K)

    def get_meanvalue(self):
        return self.K + self.ex / self.n

    def get_variance(self):
        return (self.ex2 - (self.ex * self.ex) / self.n) / (self.n - 1)
