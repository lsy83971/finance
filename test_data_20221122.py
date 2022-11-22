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


self_corr(sd.ret_d.iloc[0], 10)
us = list()
for i in range(247):
    us.append(self_corr(sd.ret_d.iloc[i], 2).iloc[1])

sb = sd.ret_d.T.melt()
self_corr(sb, 5)
self_corr(sb, 5)






sb = sd.corr_inv@sd.ret_d
sb1 = sd.ret_d.iloc[:, 1:]
prof = sd.corr_inv@(- ((sb > 0)*2 - 1).iloc[:, : -1])
prof_ret = pd.DataFrame(prof.values * sb1.values)



b = sd_test.ret_d
a = sd.corr_inv

_idx = a.index.intersection(b.index)

b1 = b.loc[_idx]
a1 = a.loc[_idx, _idx]

sb = a1@b1



sb1 = sb.T.melt()["value"]
self_corr(sb1, 10)

sb2 = pd.concat([sb1.shift(i) for i in range(20)], axis=1).iloc[20:, ]. corr()


pd.concat([pd.Series(np.random.rand(50000)), pd.Series(np.random.rand(50000))], axis=1).corr()


sb1 = b1.iloc[:, 1:]
prof = a1@(- ((sb > 0)*2 - 1).iloc[:, : -1])
prof_ret = pd.DataFrame(prof.values * sb1.values)


gg = sb.melt()["value"]
pd.concat([gg.shift(i) for i in range(20)], axis=1).iloc[20:, ]. corr().iloc[0]
pd.concat([gg.shift(i) for i in range(20)], axis=1).iloc[20:, ]. corr().iloc[1]

sb = sd.corr_inv@sd.ret_d
sb1 = sd.ret_d.iloc[:, 1:]
prof = sd.corr_inv@(- ((sb > 0)*2 - 1).iloc[:, : -1])
prof_ret = pd.DataFrame(prof.values * sb1.values)





import matplotlib.pyplot as plt
plt.plot(prof_ret.mean().cumsum())

gg = prof_ret.mean()

gg[(gg > 0)].sum()
gg[(gg < 0)].sum()
(gg.mean() / gg.std())*16





df = pd.concat([ret_d.shift(i) for i in range(20)], axis=1).iloc[20:, ]
df.corr().iloc[:10, 0]




v1 = pd.DataFrame(ss.iloc[:, 1:]. values * ss.iloc[:, : -1]. values, index=sd.stocks)
v0 = pd.DataFrame(ss.iloc[:, :]. values * ss.iloc[:, :]. values, index=sd.stocks)
v_rate = (v1.sum(axis=1) / v0.sum(axis=1))


ss = sd.corr_inv@sd_test.ret_d.loc[sd.corr_inv.index]
v1_test = pd.DataFrame(ss.iloc[:, 1:]. values * ss.iloc[:, : -1]. values, index=sd.stocks)
v0_test = pd.DataFrame(ss.iloc[:, :]. values * ss.iloc[:, :]. values, index=sd.stocks)
v_rate_test = (v1_test.sum(axis=1) / v0_test.sum(axis=1))



v_concat = pd.concat([v_rate_test, v_rate], axis=1).sort_values(1)
v_concat.head(20)

v_concat.tail(20)
v_concat.mean()







sd_test.ts.shape
sd.ts.shape




















