# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso
import re

basic_info = pd.read_pickle("./stockdata/stocks_basic_info.pkl")
cn_name = basic_info.set_index("code")["code_name"]

class comp:
    def __init__(self, info):
        self.info = info
        self.dt = pd.Series(self.info.keys())

    def get(self, date):
        date = str(date)
        if re.search("\d{8}", date):
            date = date[:4] + '-' + date[4:6] + '-' + date[6:8]
        tmp = self.dt[(self.dt <= date)]
        if len(tmp) <= 0:
            raise
        return self.info[tmp.max()]

class decomp_y:
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

def shift_ret(s):
    return s[1:]. values / s[: -1]. values- 1

def _ts_info(s):
    """from day-hours ts to ts_info"""
    s = pd.Series(s)
    s1 = s.astype(str)
    s1 = s1.replace("-", "")
    if len(s1.iloc[0]) == 8:
       s1 = s1 + "0"
       s = s1.astype(int)
    s2 = pd.concat([pd.to_datetime(s1.str[:8]), s1.str[8]. astype(int), s], axis = 1)
    s2.columns = ["time", "hour", "raw"]
    s2.index = s
    s2["year"] = s2["time"]. dt.year
    s2["month"] = s2["time"]. dt.month
    s2["day"] = s2["time"]. dt.day
    s2["season"] = s2["month"].apply(lambda x:(x - 1)// 3) + 1
    s2.index = s
    return s2


class ts_info:
    def __init__(self, s):
        if isinstance(s, pd.DataFrame):
            self.info = s
            self.s = self.info["raw"]

        else:
            s = pd.Series(s)
            #self.sort_values(inplace=True)
            self.s = s
            self.info = _ts_info(s)

    def gp(self, l="year"):
        for i, j in self.info.groupby(l):
            yield (i, ts_info(j))

    def begin(self, l="year"):
        return ts_info(self.info.drop_duplicates(l))

    def end(self, l="year"):
        return ts_info(self.info.iloc[:: -1].drop_duplicates(l).iloc[:: -1])


class df_ts:
    def __init__(self, info):
        self.info = info
        self.ts = ts_info(info.index)
        self.info.index = self.ts.info.index

    def concat(self, l):
        return self.__class__(pd.concat([i.info for i in l]))

    def is_on(self, i):
        return pd.Series(self.info.columns[~self.info.loc[i]. isnull()])

    def always_on(self):
        return pd.Series(self.info.columns[(~self.info.iloc[0]. isnull()) & (~self.info.iloc[ - 1]. isnull())])

    def df_on(self):
        return self.info[self.always_on()]

    def sub(self, ts=None, stocks=None):
        _info = self.info
        if ts is not None:
            ts = pd.Series(ts)
            if ts.dtypes is bool:
                _info = _info.loc[i]
            else:
                _info = _info.loc[_info.index.isin(ts)]

        if stocks is not None:
            _info = _info[stocks]

        return self.__class__(_info)

    def gp1(self, l="year"):
        for i, j in self.ts.gp(l):
            _tmp = self.sub(ts=j.info["raw"])
            yield i, _tmp

    def gp1l(self, l="year", lag=2):
        for i, j in iter_concat(self.gp1(l=l), lag=lag):
            yield i, j

    def end(self, l="year"):
        return self.sub(ts=self.ts.end(l=l).info["raw"])
        
    def begin(self, l="year"):
        return self.sub(ts=self.ts.begin(l=l).info["raw"])

    def proc1(self):
        j1 = self.df_on()
        j1 = j1.clip(-0.1, 0.1)
        j1_std = j1.std()
        j2 = j1[j1_std.index[j1_std > 0]]
        j2_std = j1_std[j2.columns]
        j3 = j2 / j2_std
        return j3

class ret_ts(df_ts):
    pass
    
class price_ts(df_ts):
    def ret(self):
        _log = self.info.applymap(lambda x:math.log(x))
        self.ret_info = ret_ts((_log - _log.shift(1)).iloc[1:])
        #self.ret_info = ret_ts((self.info / self.info.shift(1) - 1).iloc[1:])
        return self.ret_info

class amount_ts(df_ts):
    def end(self, l="year"):
        df = pd.concat([j.info.sum() for i, j in self.gp1(l=l)], axis=1).T
        df.index = self.ts.end(l).info.index
        return amount_ts(df)
            
    

def iter_concat(g, lag=1):
    s = [(i, j) for i, j in g]
    s0 = [k[0] for k in s]
    s1 = [k[1] for k in s]
    if isinstance(s1[0], df_ts):
        _cls = s1[0]. __class__        
        s1 = [k.info for k in s1]
        print(_cls)
    else:
        _cls = lambda x:x

    if isinstance(lag, int):
        l1 = len(s0) - lag + 1
        assert l1 >= 1
        for i in range(l1):
            yield s0[i],_cls(pd.concat(s1[i:(i + lag)]))

    if isinstance(lag, list):
        l1 = len(s0) - sum(lag) + 1
        assert l1 >= 1
        for i in range(l1):
            _tmp = list()
            _begin = i
            _c = _begin
            for k in lag:
                _c1 = _c + k
                _tmp.append(_cls(pd.concat(s1[_c:_c1])))
                _c = _c1
            yield s0[i],_tmp
    
def check_price(df):
    mask = df.isnull()
    iszero = (df <= 0).sum()
    iszero_s = iszero[iszero > 0]
    print(iszero_s)
    if iszero_s.shape[0] == 0:
        return
    df.where(df > 0, inplace=True)
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)
    assert df.isnull().sum().sum() == 0
    df.where(~mask, inplace=True)


class cls_pca:
    def __init__(self, x, n_components=10):
        self.pca = PCA(n_components=n_components)
        self.x = x
        self.fit(x)

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

    



