import sys
from importlib import reload
sys.path.append("c:/Users/48944/finance/")
import data
reload(data)
from data import *
import pdb
c:/Users/48944/finance/test_data.py

bs.login()
stock_300 = bs.query_hs300_stocks("2020-01-01").get_data()
stocks = stock_300["code"]
t1 = "2015-01-01"
t2 = "2022-01-01"
t3 = "2022-11-01"

sd = stock_dataset(stocks[:10], t1, t2)
sd = stock_dataset(stocks, t1, t2)
sd_test = stock_dataset(stocks, t2, t3)

ss = sd.corr_inv@sd.ret_d
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




















