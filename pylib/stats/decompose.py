from sklearn.linear_model import LinearRegression, Lasso
import pandas as pd
import numpy as np

class decomp_y:
    """
    decompose y by x
    r1: decomposed part of y by x
    r2: residual
    dcp_r: decomposed rate
    """
    def __init__(self, alpha, y, x):
        self.ls = Lasso(alpha=alpha, fit_intercept=False)
        self.y = y
        self.x = x
        self.ls.fit(x, y)

        self.coef = pd.Series(self.ls.coef_, index=x.columns)
        self.coef1 = self.coef[~(self.coef == 0)]

        self.r1 = self.x@self.coef
        self.r2 = self.y - self.r1
        self.dcp_r = (self.r1 ** 2).sum() / (self.y ** 2).sum()


class decomp_raw:
    """
    decompose y by x
    r1: decomposed part of y by x
    r2: residual
    dcp_r: decomposed rate
    """
    def __init__(self, x, y):
        beta = (y * x.T).sum(axis=1) / (y ** 2).sum()
        alpha = x - y.values[:, None] * beta.values[None, :]
        self.beta = beta
        self.alpha = alpha

    def trans_alpha(self, x, y):
        return x - y.values[:, None] * self.beta.values[None, :]
