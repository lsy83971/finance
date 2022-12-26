import sys
import os
sys.path.append("c:/Users/48944/finance/pylib")
from factor.ts import *
from stats.pca_comp import pca_comp
from sklearn.metrics import roc_auc_score
import tools.pd_tools

stock_files = os.listdir("c:/Users/48944/finance/kdata60")
infos = [pd.read_pickle(f"c:/Users/48944/finance/kdata60/{i}") for i in stock_files]
close_df = pd.concat([i["close"] for i in infos], axis=1)
close_df.columns = [i[: -4] for i in stock_files]
check_price(close_df)

amount_df = pd.concat([i["amount"] for i in infos], axis=1)
amount_df.columns = [i[: -4] for i in stock_files]

sp = price_ts(close_df)
amount = amount_ts(amount_df)
spd = sp.end(["year", "month", "day"])
amtd = amount.end(["year", "month", "day"])

rtd = spd.ret()
hs300 = comp(pd.read_pickle("c:/Users/48944/finance/compdata/hs300.pkl"))
sz50 = comp(pd.read_pickle("c:/Users/48944/finance/compdata/sz50.pkl"))
zz500 = comp(pd.read_pickle("c:/Users/48944/finance/compdata/zz500.pkl"))

