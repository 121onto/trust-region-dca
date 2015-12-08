# coding: utf-8

# Copyright (C) 2015, 121onto.  All rights reserved.
# Code based on:
#
# Pham Dinh Tao and Le Thi Hoai An,
# "A DC Optimization Algorithm for Solving the Trust-Region Subproblem,"
# SIAM Journal on Optimization 8, no. 2 (1998): 476–505.

###########################################################################
## imports

import numpy as np
import pandas as pd
import scipy

###########################################################################
## class

class TrustRegionDCA(object):
    def __init__(self, A, b, r, n=None,
                 stopping_tol=10-6, irlm_tol=0.1):
        """"""
        self.A = A
        self.b = b
        self.r = r
        self.n = n if n is not None else len(A)

        self.irlm_tol = irlm_tol
        self.stopping_tol = stopping_tol

        self.xk_0 = None
        self.xk_1 = self.initialize_xk(n, r)
        self.rho = self.initialize_rho()
        self.cut = self.rho * self.r
        self.A_alt = np.diag(self.rho) - self.A

    def initialize_rho(self, tol=None):
        tol = tol if tol is not None else self.irlm_tol
        lambda_n = scipy.sparse.linalg.eigsh(
            self.A,
            k=1,
            which='LM',
            tol=tol,
            return_eigenvectors=False
        )
        # TODO: consider alternatives to 0.5
        rho = lambda_n[0] + 0.5
        return rho

    def initialize_xk(self, n, r):
        """"""
        assert(n>0)
        x0 = np.empty(n)
        x0.fill(r/np.sqrt(n))
        return x0

    def update_xk(self):
        """"""
        self.xk_0 = self.xk_1.copy()
        self.xk_1 = (np.dot(self.A_alt, self.xk_1) - self.b)
        norm = np.linalg.norm(self.xk_1)
        if norm <= self.cut:
            self.xk_1 /= self.rho
        else:
            self.xk_1 *= (self.r / norm)

    def stopping_condition(self, tol=None):
        """"""
        tol = tol if tol is not None else self.stopping_tol
        if np.linalg.norm(self.xk_0 - self.xk_1) < tol:
            return True

        return False

    def step(self):
        self.update_xk()
        return self.stopping_condition()

###########################################################################
## main

def main():
    trdca = TrustRegionDCA(
        A = None,
        b = None,
        r = None,
        n = None
    )

    while trdca.step():
        continue

    return trdca.xk_1

if __name__ == '__main__':
    result = main()
    print(result)
