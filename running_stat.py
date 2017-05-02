import threading
import numpy as np

# Author: Kyle Kaster
# License: BSD 3-clause
import numpy as np


def online_stats(X):
    """
    Converted from John D. Cook
    http://www.johndcook.com/blog/standard_deviation/
    """
    prev_mean = None
    prev_var = None
    n_seen = 0
    for i in range(len(X)):
        n_seen += 1
        if prev_mean is None:
            prev_mean = X[i]
            prev_var = 0.
        else:
            curr_mean = prev_mean + (X[i] - prev_mean) / n_seen
            curr_var = prev_var + (X[i] - prev_mean) * (X[i] - curr_mean)
            prev_mean = curr_mean
            prev_var = curr_var
    # n - 1 for sample variance, but numpy default is n
    return prev_mean, np.sqrt(prev_var / n_seen)


class RunningStat:

    # FIXME: See why variance sometimes ends up being negative
    def __init__(self):
        self.lock = threading.Lock()
        self.prev_mean = None
        self.prev_var = None
        self.n_seen = 0

    def add_batch(self, X):

        self.lock.acquire()

        flat_x = X.flatten()

        if self.n_seen == 0:
            self.prev_mean = np.float64(flat_x[0])
            self.prev_mean = np.float64(flat_x[0])
            self.prev_var  = 0

        for x in flat_x:
            self.n_seen += 1
            curr_mean = self.prev_mean + (x - self.prev_mean) / self.n_seen
            curr_var = self.prev_var + (x - self.prev_mean) * (x - curr_mean)
            self.prev_mean = curr_mean
            self.prev_var = curr_var

        self.lock.release()

    def get_mean(self):
        return self.prev_mean

    def get_variance(self):
        return self.prev_var
        # print(self.s0)
        # print(self.s1)
        # print(self.s2)

        self.s0 = np.float64(self.s0)
        self.s1 = np.float64(self.s1)
        self.s2 = np.float64(self.s2)

        denom = np.float64(self.s0 * (self.s0 - 1))
        s1_s1 = np.float64(self.s1 * self.s1)
        s0_s2 = np.float64(self.s0 * self.s2)
        num = np.int64(s0_s2 - s1_s1)
        self.var = np.float64(num / denom)
        print("s0 " + str(self.s0))
        print("s1 " + str(self.s1))
        print("s2 " + str(self.s2))
        print("s1_s1 " + str(s1_s1))
        print("s0_s2 " + str(s0_s2))
        print("num " + str(num))
        print("den " + str(denom))
        return self.var
