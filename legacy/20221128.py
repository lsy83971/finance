# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso

basic_info = pd.read_pickle("./stockdata/stocks_basic_info.pkl")
cn_name = basic_info.set_index("code")["code_name"]


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
    s2 = pd.concat([pd.to_datetime(s1.str[:8]), s1.str[8]. astype(int), s], axis = 1)
    s2.columns = ["time", "hour", "raw"]
    s2["year"] = s2["time"]. dt.year
    s2["month"] = s2["time"]. dt.month
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

    

stock_files = os.listdir("./kdata60")
opens = [pd.read_pickle(f"./kdata60/{i}")["open"] for i in stock_files]
open_df = pd.concat(opens, axis=1)
open_df.columns = [i[: -4] for i in stock_files]
#open_df.fillna(method="bfill", inplace=True)

closes = [pd.read_pickle(f"./kdata60/{i}")["close"] for i in stock_files]
close_df = pd.concat(closes, axis=1)
close_df.columns = [i[: -4] for i in stock_files]
#close_df.fillna(method="bfill", inplace=True)

 
# 1. 对比PCA hs300-fit/total-fit 得到主特征的有效性
check_price(close_df)
sp = price_ts(close_df)
ret = sp.ret()


for i, j in ret.gp1l("year", [4, 1]):
    break
    df0 = j[0]. proc1()
    df1 = j[1]. proc1()
    pca0 = cls_pca(df0)
    pca1 = cls_pca(df1)
    print(np.diag(corr2(pca0.trans(df1), pca1.trans(df1))))
    pca0.trans(df1).corr()

sb = pca0.trans(df0)
coef = (df0.T@sb / np.diag(sb.T@sb))




i = 4
gg = (coef.iloc[:, :i]@sb.T.iloc[:i, ]).T
((df0 - gg)**2).mean().mean()
(gg ** 2).mean().mean()


i = 4
sb1 = pca0.trans(df1).iloc[:, :i]
idx = df1.columns & coef.index
gg1 = (coef.iloc[:, :i].loc[idx]@sb1.T).T
((df1[idx] - gg1)**2).mean().mean()
(gg1 ** 2).mean().mean()


idx_next_year = pca0.trans(df1)
idx_next_year_std = idx_next_year / idx_next_year.std()
idx_next_year_std

g1 = corr2(idx_next_year_std.iloc[:, :2], df1).T
g2 = pca0.corr.iloc[:, :2]

g3 = (g1 - g2)
g4 = g3.loc[~(g3.isnull().sum(axis=1) > 0)]
g4.abs().mean()
g2.abs().mean()




corr2(pca0.trans(df1)[1], pca1.trans(df1)[0])
corr2(pca0.trans(df1)[2], pca1.trans(df1)[0])


def cn(s):
    if isinstance(s, pd.DataFrame):
        s = s.copy()
        s["cn"] = pd.Series(s.index).apply(lambda x:cn_name.get(x)).values
        return s
    if isinstance(s, pd.Series):
        s = pd.DataFrame(s)
        s["cn"] = pd.Series(s.index).apply(lambda x:cn_name.get(x)).values
        return s
    raise


cn(pca0.comp[1]).sort_values(1)
cn(pca0.comp[0]).sort_values(0)

cn(pca1.comp[1]).sort_values(1)
cn(pca1.comp[0]).sort_values(0)




#pca0.trans(df1)
    
for i, j in ret.gp1l("year", [1, 1]):
    df0 = j[0]. proc1()
    df1 = j[1]. proc1()
    pca0 = cls_pca(df0)
    pca1 = cls_pca(df1)
    print(np.diag(corr2(pca0.trans(df1), pca1.trans(df1))))

pca_dict = dict()
ret_dict = dict()
for i, j in ret.gp1("year"):
    j1 = j.df_on()
    j1 = j1.clip(-0.1, 0.1)
    j1_std = j1.std()
    j2 = j1[j1_std.index[j1_std > 0]]
    j2_std = j1_std[j2.columns]
    j3 = j2 / j2_std
    ret_dict[i] = j3
    pca_dict[i] = cls_pca(j3)
    

annul_comp_corr = np.zeros([len(pca_dict), len(ret_dict), 10])
for j1, i1 in enumerate(pca_dict.keys()):
    for j2, i2 in enumerate(ret_dict.keys()):
        l1 = pca_dict[i1].trans(ret_dict[i2])            
        l2 = pca_dict[i2].trans(ret_dict[i2])
        annul_comp_corr[j1, j2] = np.diag(corr2(l1, l2))

pd.DataFrame(annul_comp_corr[:, :, 0])
pd.DataFrame(annul_comp_corr[:, :, 1])
pd.DataFrame(annul_comp_corr[:, :, 2])
pd.DataFrame(annul_comp_corr[:, :, 3])
pd.DataFrame(annul_comp_corr[:, :, 4])
pd.DataFrame(annul_comp_corr[:, :, 5])








# 当自变量只有一个时，常用的回归方法有一元线性回归（SLR）；
# 当自变量有多个时，常用的回归方法有多元线性回归（MLR）、
# 主成分回归（PCR）、偏最小二乘回归（PLS）等，这几种回归方法的联系和区别如下：
# https://blog.csdn.net/dongke1991/article/details/126843609


# 季节因子

# 生产规律

# 信息公布的季节性
# 公司决策的季节性



# 1.定制指数
# 1.1 调整频率
# 1.2 未上市处理模式
# 季节因子


# 2.寻找因子



# 3.组合


