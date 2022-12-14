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

sp.ts.info.iloc[0]
spd = sp.end(["year", "month", "day"])
amtd = amount.end(["year", "month", "day"])
rtd = spd.ret()
hs300 = comp(pd.read_pickle("c:/Users/48944/finance/compdata/hs300.pkl"))
sz50 = comp(pd.read_pickle("c:/Users/48944/finance/compdata/sz50.pkl"))
zz500 = comp(pd.read_pickle("c:/Users/48944/finance/compdata/zz500.pkl"))

os.chdir("c:/Users/48944/finance/daily_analysis/")

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


factor_stockd = rtd.df_on().shift(1).iloc[1:]
target_stockd = rtd.df_on()



train_ts = factor_mkt_df.index.intersection(factor_stockd.index).intersection(target_stockd.index)
valid_ts = factor_mkt_df_valid.index.intersection(factor_stockd.index).intersection(target_stockd.index)


sb = train_factor_stockd.mean(axis=1)
g1 = (sb * train_factor_stockd.T).sum(axis=1)
g2 = ((sb ** 2).sum()*(sb ** 2).sum())**(1 / 2)
#g2 = ((sb ** 2).sum()*(train_factor_stockd ** 2).sum())**(1 / 2)

g3 = g1 / g2
g3.quantile([i / 10 for i in range(11)])


train_target_stockd

train_factor_stockd = factor_stockd.loc[train_ts]

mean_ret = train_factor_stockd.mean(axis=1)
beta = (sb * train_factor_stockd.T).sum(axis=1) / (sb ** 2).sum()
alpha = train_factor_stockd - mean_ret.values[:, None] * beta.values[None, :]

g1 = (alpha.shift(1)[1:] * alpha[1:]).sum(axis=1)
g2 =((alpha.shift(1)[1:] ** 2).sum(axis=1)*(alpha[1:] ** 2).sum(axis=1))**(1 / 2)

cor = (g1 / g2)
cor.quantile([i / 10 for i in range(11)])





train_target_stockd = target_stockd.loc[train_ts]
train_factor_mkt_df = factor_mkt_df.loc[train_ts]

train_factor_stockd_valid = factor_stockd.loc[valid_ts]
train_target_stockd_valid = target_stockd.loc[valid_ts]
valid_factor_mkt_df = factor_mkt_df_valid.loc[valid_ts]


train_target_stockd = train_target_stockd.clip( -0.15, 0.15)
train_factor_stockd = train_factor_stockd.clip( -0.15, 0.15)

train_factor_stockd_valid = train_factor_stockd_valid.clip( -0.15, 0.15)
train_target_stockd_valid = train_target_stockd_valid.clip( -0.15, 0.15)

train_factor_stockd = (train_factor_stockd.T - train_factor_stockd.mean(axis=1)).T
train_target_stockd = (train_target_stockd.T - train_target_stockd.mean(axis=1)).T

train_factor_stockd_valid = (train_factor_stockd_valid.T - train_factor_stockd_valid.mean(axis=1)).T
train_target_stockd_valid = (train_target_stockd_valid.T - train_target_stockd_valid.mean(axis=1)).T



sb = (train_factor_stockd * train_target_stockd).sum(axis=1) / ((train_factor_stockd ** 2).sum(axis=1)*(train_target_stockd ** 2).sum(axis=1))**(1 / 2)
sb.quantile([i / 10 for i in range(11)])

import numpy as np
x = np.zeros([train_factor_stockd.shape[0] * train_factor_stockd.shape[1], train_factor_mkt_df.shape[1]])
for i in range(12):
    i1 = np.zeros_like(train_target_stockd)
    i1.T[:, :] = train_factor_mkt_df.iloc[:, i]
    x[:, i] = i1.flatten()
x = pd.DataFrame(x)
param = pd.DataFrame(train_factor_stockd.values.flatten())
param1 = (param - param.mean()) / param.std()
y = pd.Series(train_target_stockd.values.flatten()) > 0


gdt = _gd_t(min_cnt=100000, trace=5000, max_depth=3, max_nodes=8)
gdt.data(param1, x, y)
y1 = gdt.y1


learning_rate = 0.1
lvs_cluster = list()
        
for i in range(10):
    gn1 = _gd_node(gdt)
    gn1.rec_split()
    self = gdt
    mask = gn.leaves[0]. mask
    a, b=gdt.calc_gd(gn.leaves[0]. mask)

    
    gn.calc_t_gd()
    gn1.calc_t_gd()    

    lvs = plgd_sta_leaves(gn.leaves)
    lvs_cluster.append(lvs)
    y1 += gn.gd_t_param * learning_rate
    y1.gd(y)
    print(i)    
    print(roc_auc_score(y, y1))
    print(roc_auc_score(y, y1))






