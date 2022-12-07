import sys
import os
sys.path.append("c:/Users/48944/finance/")
from tools.price_struct import *
from factor.pca_comp import pca_comp


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



pf_total.idx0_record[2013]



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



train_factor_stockd = factor_stockd.loc[train_ts]
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


g12 = (train_factor_stockd * train_target_stockd).sum(axis=1)
g11 = (train_factor_stockd * train_factor_stockd).sum(axis=1)
g22 = (train_target_stockd * train_target_stockd).sum(axis=1)

g12_valid = (train_factor_stockd_valid * train_target_stockd_valid).sum(axis=1)
g11_valid = (train_factor_stockd_valid * train_factor_stockd_valid).sum(axis=1)
g22_valid = (train_target_stockd_valid * train_target_stockd_valid).sum(axis=1)


corr_ts = g12 / (g11 * g22)**(1 / 2)
corr_ts_valid = g12_valid / (g11_valid * g22_valid)**(1 / 2)
#corr_ts.quantile([i / 10 for i in range(11)])



import lightgbm as lgb


x = factor_mkt_df
y = corr_ts

tt_cut = int(x.shape[0] * 0.6)
tt_id1 = x.index[:tt_cut]
tt_id2 = x.index[tt_cut:]


x1 = x.loc[tt_id1]
y1 = y.loc[tt_id1]

x2 = x.loc[tt_id2]
y2 = y.loc[tt_id2]



lgbs1 = lgb.Dataset(x1, y1 > 0)
lgbs2 = lgb.Dataset(x2, y2 > 0)



params = {
            'boosting_type': 'gbdt',
            'boosting': 'dart',
            'objective': 'binary',
            'metric': 'auc',
 
            'learning_rate': 0.01,
            'num_leaves':25,
            'max_depth':3,
 
            'max_bin':10,
            'min_data_in_leaf':8,
 
            'feature_fraction': 0.6,
            'bagging_fraction': 1,
            'bagging_freq':0,
 
            'lambda_l1': 0,
            'lambda_l2': 0,
            'min_split_gain': 0
}

gbm = lgb.train(params,                     # 参数字典
                lgbs1,                      # 训练集
                num_boost_round=300,        # 迭代次数
                valid_sets=lgbs2,           # 验证集
                early_stopping_rounds=20)   # 早停系数

score = gbm.predict(valid_factor_mkt_df, gbm.best_iteration)
score = pd.Series(score, index=valid_factor_mkt_df.index)
from sklearn.metrics import roc_auc_score

roc_auc_score((corr_ts_valid > 0), score)
pd.Series(gbm.feature_importance(), index=x1.columns).sort_values()

pd.concat([corr_ts_valid, score], axis=1).corr()


