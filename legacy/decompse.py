from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso
import pandas as pd
import numpy as np


class factor:
    def __init__(self, df):
        self.df = df
        self.inner = self.df.T@self.df
        self.inv = pd.DataFrame(np.linalg.inv(self.inner))
        self.inv.index = self.df.columns
        self.inv.columns = self.df.columns        

    def factor(self, l):
        _inner = l.values.T@self.df
        self.coef = _inner@self.inv
        self.coef.index = l.columns
        self.comp1 = pd.DataFrame(self.df@self.coef.T)
        self.comp1.columns = l.columns
        self.comp1.index = l.index
        self.comp2 = l - self.comp1

    def factor_mask(self, l, m):
        m = (m.loc[l.index, l.columns] > 0)
        lr = LinearRegression(fit_intercept=False)
        d = dict()
        for i in l.columns:
            _mask = m[i]
            lr.fit(self.df. loc[_mask.values], l[i]. loc[_mask.values])
            d[i] = lr.coef_. copy()
        coef = pd.DataFrame(d).T
        coef.columns = self.df.columns
        self.coef = coef
        self.comp1 = pd.DataFrame(self.df@self.coef.T)
        self.comp2 = l - self.comp1.values
        self.mask = m

    def trans(self, df):
        self.trans_comp = df@self.coef.T

























