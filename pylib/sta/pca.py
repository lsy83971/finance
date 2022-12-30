import numpy as np
import pandas as pd
import math
from sklearn.decomposition import PCA

class cls_pca:
    def __init__(self, n_components=10):
        self.pca = PCA(n_components=n_components)

    def fit(self, x):
        self.pca.fit(x)
        self.comp = pd.DataFrame(self.pca.components_).T
        self.comp.index = x.columns
        self.ev = pd.Series(self.pca.explained_variance_ratio_)

        self.index = x@self.comp
        self.index_std = self.index / self.index.std()
        self.corr = corr2(x, self.index)

    def trans(self, x):
        i1 = x.columns
        i2 = self.comp.index
        i3 = i1.intersection(i2)
        assert (len(i3) >= 1)
        return x[i3]@self.comp.loc[i3] 


def corr2(df1, df2):
    df3 = df1.T@df2
    g1 = ((df1 ** 2).sum())**(1 / 2)
    g2 = ((df2 ** 2).sum())**(1 / 2)
    g3 = ((df3 / g2).T / g1).T
    return g3

    
