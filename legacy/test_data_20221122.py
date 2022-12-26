import sys
from importlib import reload
#sys.path.append("c:/Users/48944/finance/")
import data
reload(data)
from data import *
import pdb

bs.login()
stock_300 = bs.query_hs300_stocks("2020-01-01").get_data()
stocks = stock_300["code"]
t1 = "2015-01-01"
t2 = "2022-01-01"
t3 = "2022-11-01"

sd = stock_dataset(stocks, t1, t2)
sd_test = stock_dataset(stocks, t2, t3)

def self_corr(df, i):
    return pd.concat([df.shift(k) for k in range(i)], axis=1).corr().iloc[0]





us = list()
for i in range(247):
    us.append(self_corr(sd.ret_d.iloc[i], 2).iloc[1])

sb = sd.ret_d.T.melt()
self_corr(sb, 5)
self_corr(sb, 5)



g1 = (sd.corr_inv_diag < 2)
g1 = (sd.corr_inv_diag > 2) & (sd.corr_inv_diag < 3)
g1 = (sd.corr_inv_diag > 3) & (sd.corr_inv_diag < 4)
g1 = (sd.corr_inv_diag > 4) & (sd.corr_inv_diag < 5)

sb2 = sb.loc[g1[g1]. index]. T.melt()["value"]
self_corr(sb2, 5)













sb = sd.corr_inv@sd.ret_d
sb1 = sd.ret_d.iloc[:, 1:]
prof = sd.corr_inv@(- ((sb > 0)*2 - 1).iloc[:, : -1])
prof_ret = pd.DataFrame(prof.values * sb1.values)





b = sd_test.ret_d

b = sd.ret_d
a = sd.corr_inv

_idx = a.index.intersection(b.index)

b1 = b.loc[_idx]
a1 = a.loc[_idx, _idx]

sb = a1@b1







sb1 = sb.T.melt()["value"]
self_corr(sb1, 10)






