from sklearn.metrics import roc_auc_score
import sys
import os
sys.path.append("c:/Users/48944/finance/")
sys.path.append("c:/Users/48944/finance/tools")
from tools.price_struct import *
from factor.pca_comp import pca_comp
import pd_tools


stock_files = os.listdir("c:/Users/48944/finance/kdata60")
infos = [pd.read_pickle(f"../kdata60/{i}") for i in stock_files]
close_df = pd.concat([i["close"] for i in infos], axis=1)
close_df.columns = [i[: -4] for i in stock_files]

amount_df = pd.concat([i["amount"] for i in infos], axis=1)
amount_df.columns = [i[: -4] for i in stock_files]



# 1. 对比PCA hs300-fit/total-fit 得到主特征的有效性
check_price(close_df)

sp = price_ts(close_df)
amount = amount_ts(amount_df)
spd = sp.end(["year", "month", "day"])
amtd = amount.end(["year", "month", "day"])

rtd = spd.ret()
hs300 = comp(pd.read_pickle("c:/Users/48944/finance/compdata/hs300.pkl"))
sz50 = comp(pd.read_pickle("c:/Users/48944/finance/compdata/sz50.pkl"))
zz500 = comp(pd.read_pickle("c:/Users/48944/finance/compdata/zz500.pkl"))

i1 = 3
i2 = 1
pf300 = pca_comp(rtd, hs300)
pf500 = pca_comp(rtd, zz500)
pf50 = pca_comp(rtd, sz50)
pf_total = pca_comp(rtd)
pf300.yearly(i1, i2)
pf500.yearly(i1, i2)
pf50.yearly(i1, i2)
pf_total.yearly(i1, i2)

pf_total.idx0_record.keys()
pf_total.idx1_record.keys()

#pf300.idx0_record[2013]. c1("hs300_")
#pf500.idx0_record[2013]. c1("zz500_")
#pf_total.idx0_record[2013]. c1("total_")
#pf50.idx0_record[2013]. c1("sz50_")
pf_sets = [pf50, pf300, pf500, pf_total]
pf_header = ["sz50_", "hs300_", "zz500_", "total_"]

for i, j in zip(pf_sets, pf_header):
    i.header = j

    
year = 2018
factor_mkt = pd.concat([i.idx0_record[year]. c1(i.header) for i in pf_sets], axis=1)
factor_mkt_valid = pd.concat([i.idx1_record[year]. c1(i.header) for i in pf_sets], axis=1)

factor_mkt_df = factor_mkt.shift(1).iloc[1:]
factor_mkt_df_valid = factor_mkt_valid.shift(1).iloc[1:]


stockd = rtd.df_on()
stockad = amtd.df_on()[stockd.columns]
stockd = stockd.clip(-0.15, 0.15)
tdr = decomp_raw(stockd.loc[factor_mkt.index], stockd.loc[factor_mkt.index]. mean(axis=1))
alpha = tdr.trans_alpha(stockd, stockd.mean(axis=1))

t_minus = dict()
for i in range(1, 6):
    t_minus[i] = alpha.shift(i).iloc[i:]

t_minus_amt = dict()
for i in range(1, 6):
    t_minus_amt[i] = stockad.shift(i).iloc[i:]

train_ts = factor_mkt_df.index
for i, j in t_minus.items():
    train_ts = train_ts.intersection(j.index)

valid_ts = factor_mkt_df_valid.index
for i, j in t_minus.items():
    valid_ts = valid_ts.intersection(j.index)

import numpy as np
x = np.zeros([train_ts.shape[0] * train_factor_stockd.shape[1], train_factor_mkt_df.shape[1]])
for i in range(12):
    i1 = np.zeros([train_ts.shape[0], train_factor_stockd.shape[1]])
    i1.T[:, :] = train_factor_mkt_df.iloc[:, i]
    x[:, i] = i1.flatten()
x = pd.DataFrame(x)

param_ret = pd.DataFrame({i: j.loc[train_ts]. values.flatten() for i, j in t_minus.items()})
param_amt = pd.DataFrame({i: j.loc[train_ts]. values.flatten() for i, j in t_minus_amt.items()})
param = pd.concat([param_ret, param_amt], axis=1)
param.columns = range(param.columns.shape[0])
st_train = std()
st_train.fit(param)
#param1 = st_train.trans(param).iloc[:, :1]
param1 = st_train.trans(param)

y = pd.Series(alpha.loc[train_ts].values.flatten()) > 0
gdt = _gd_t(min_cnt=100000, trace=5000, max_depth=3, max_nodes=8)
gdt.data(param1, x, y)
y1 = gdt.y1

learning_rate = 0.1
lvs_cluster = list()
y1_gd = list()
        
for i in range(10):
    gn = _gd_node(gdt)
    gn.rec_split()
    gn.calc_t_gd()
    lvs = plgd_sta_leaves(gn.leaves)
    lvs_cluster.append(lvs)
    y1 += gn.gd_t_param * learning_rate
    y1_gd.append(gn.gd_t_param)
    y1.gd(y)
    print(roc_auc_score(y, y1))


x_valid = np.zeros([valid_ts.shape[0] * alpha.shape[1], valid_factor_mkt_df.shape[1]])
for i in range(12):
    i1 = np.zeros([valid_ts.shape[0], alpha.shape[1]])
    i1.T[:, :] = valid_factor_mkt_df.iloc[:, i]
    x_valid[:, i] = i1.flatten()
x_valid = pd.DataFrame(x_valid)


param_valid = pd.DataFrame({i: j.loc[valid_ts]. values.flatten() for i, j in t_minus.items()})
param1_valid = st_train.trans(param_valid).iloc[:, :1]
param1_valid = st_train.trans(param_valid)
y_valid = pd.Series(alpha.loc[valid_ts].values.flatten()) > 0





gdt_valid = _gd_t(min_cnt=100000, trace=5000, max_depth=3, max_nodes=8)
gdt_valid.data(param1_valid, x_valid, y_valid)
y1_valid = gdt_valid.y1


for i in range(10):
    y1_valid += lvs_cluster[i]. calc_param(x_valid, param1_valid)*learning_rate
    print(roc_auc_score(y_valid, y1_valid))


import sklearn.linear_model
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(param1_valid, y_valid)
lr.fit(param1, y)
pb1 = pd.Series(lr.predict_proba(param1_valid)[:, 1], index=y1_valid.index)
pb0 = pd.Series(lr.predict_proba(param1)[:, 1], index=y1.index)
roc_auc_score(y_valid, pb1)
roc_auc_score(y, pb0)




