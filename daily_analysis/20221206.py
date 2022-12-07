import sys
import os
sys.path.append("c:/Users/48944/finance/struct")
from price_struct import *


stock_files = os.listdir("c:/Users/48944/finance/kdata60")
infos = [pd.read_pickle(f"./kdata60/{i}") for i in stock_files]
close_df = pd.concat([i["close"] for i in infos], axis=1)
close_df.columns = [i[: -4] for i in stock_files]

amount_df = pd.concat([i["amount"] for i in infos], axis=1)
amount_df.columns = [i[: -4] for i in stock_files]

# 1. 对比PCA hs300-fit/total-fit 得到主特征的有效性
check_price(close_df)

sp = price_ts(close_df)
amount = amount_ts(amount_df)

sp.ts.info.iloc[0]
spd = sp.end(["year", "month", "day"])
amtd = amount.end(["year", "month", "day"])

rtd = spd.ret()

corr = dict()
rety = dict()
for i, j in rtd.gp1("year"):
    print(i)
    rety[i] = j
    corr[i] = j.proc1().corr()
    


hs300 = comp(pd.read_pickle("c:/Users/48944/finance/compdata/hs300.pkl"))
sz50 = comp(pd.read_pickle("c:/Users/48944/finance/compdata/sz50.pkl"))
zz500 = comp(pd.read_pickle("c:/Users/48944/finance/compdata/zz500.pkl"))

code = hs300.get("20211206")["code"]
stocks_count = pd.concat([pd.Series(i.index) for i in corr.values()]).value_counts()
common_stocks = stocks_count[stocks_count == 10]. index[stocks_count[stocks_count == 10]. index.isin(code)]
self_abs = pd.Series({i:j.loc[common_stocks, common_stocks].abs().mean().mean() for i, j in corr.items()})
diff_abs = pd.DataFrame()
for i in self_abs.keys():
    for j in self_abs.keys():
        diff_abs.loc[i, j] = (corr[i].loc[common_stocks, common_stocks] - \
                              corr[j].loc[common_stocks, common_stocks]).abs().mean().m


## 走势的延续性


from importlib import reload
import test
reload(test)

from test import pca_factory
i1 = 4
i2 = 1
pf300 = pca_factory(rtd, hs300)
pf500 = pca_factory(rtd, zz500)
pf_total = pca_factory(rtd)

pf300.yearly(4, 1)
pf500.yearly(4, 1)
pf_total.yearly(4, 1)

pf300.pf_record[2013]

pf300.idx1_record[2014]
pf500.idx1_record[2014]
pf_total.idx1_record[2014]




