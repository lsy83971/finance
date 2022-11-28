import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso

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

        
alpha = 0.1


self = decomp_y(alpha=alpha, x=x, y=y)
 
# 1. 对比PCA hs300-fit/total-fit 得到主特征的有效性

stock_files = os.listdir("./kdata60")
opens = [pd.read_pickle(f"./kdata60/{i}")["open"] for i in stock_files]
open_df = pd.concat(opens, axis=1)
open_df.columns = [i[: -4] for i in stock_files]


open_df.isnull().sum(axis=1)

open_df.loc[8295][~open_df.loc[8295]. isnull()]
open_df.loc[8291][~open_df.loc[8291]. isnull()]

gg = pd.read_pickle(f"./kdata60/sh.600068.pkl")
gg["open"]
gg
gg["open"]


len(stock_files)


df = pd.read_pickle(f"./kdata60/{i}")




ps = dict()
for i in range(df3.shape[1]):
    print(i)
    ps[df3.columns[i]] = decomp_y(alpha=alpha, y=df3.iloc[:, i], x=df3.drop(df3.columns[i], axis=1))

        


