import sys
sys.path.append("c:/Users/48944/finance/pylib")
from tools.pd_tools import pds, inter_all
from env.stockd import * 


###############################################
## 1d variable
## zs
zs_ret.index_cut(8)
zs_amt.index_cut(8)

zd = pds(zs_ret.df_on(), name="zd")
zad = pds(zs_amt.df_on(), name="zad")

zad_avg10 = zad.sum_range(10).div2(10)
zad_avg10_divzad = (zad_avg10.div1(zad)).fillna1(1).shift1(1).clip1(0.2, 5)
zad_avg5 = zad.sum_range(5).div2(5)
zad_avg5_divzad = (zad_avg5.div1(zad)).fillna1(1).shift1(1).clip1(0.2, 5)

zd_sumrange1 = zd.shift1(1)
zd_sumrange5 = zd.sum_range(5).div2(5)
zd_sumrange10 = zd.sum_range(10).div2(10)

zs_params = [
    zad_avg10_divzad,
    zad_avg5_divzad,
    zd_sumrange1, 
    zd_sumrange5, 
    zd_sumrange10
    ]


zs_param = pd.concat([i.c1(i.name + "_") for i in zs_params], axis=1).sort_index().loc[inter_all(zs_params)]
zs_target = zd

#################################################
## 2d variable

rtd.index_cut(8)
amtd.index_cut(8)

rdf = pds(rtd.df_on(), name="rdf").clip1( -0.15, 0.15)
adf = pds(amtd.info[rdf.columns], "adf")
adf_avg10 = adf.sum_range(10).div2(10)
adf_avg10_divadf = (adf.div1(adf_avg10)).fillna1(1).shift1(1).clip1(0.1, 10)

adf_avg5 = adf.sum_range(5).div2(5)
adf_avg5_divadf = (adf.div1(adf_avg5)).fillna1(1).shift1(1).clip1(0.1, 10)

rdf_sumrange1 = rdf.sum_range(1)
rdf_sumrange5 = rdf.sum_range(5).div2(5)
rdf_sumrange10 = rdf.sum_range(10).div2(10)

# rdf_sumgeo1 = rdf.sum_geo(0.7).shift1(1)
# rdf_sumgeo2 = rdf.sum_geo(0.8).shift1(1)
# rdf_sumgeo3 = rdf.sum_geo(0.9).shift1(1)

params =\
[
    adf_avg10_divadf,
    adf_avg5_divadf,
    rdf_sumrange1, 
    rdf_sumrange5,
    rdf_sumrange10,
    rdf_sumgeo1,
    rdf_sumgeo2,
    rdf_sumgeo3,  
]

y = rdf
# alpha beta

class ic:
    def __init__(self, factors, y):
        ts = inter_all(factors + [y])
        y = y.loc1(ts)
        y1 = y.sub(y.mean(axis=1), axis=0)
        g22 = (y1 ** 2).sum(axis=1)
        res = dict()
        for i in factors:
            name = i.name
            i = i.loc1(ts)
            i = i.sub(i.mean(axis=1), axis=0)
            g12 = (i * y1).sum(axis=1)
            g11 = (i ** 2).sum(axis=1)
            res[name] = g12 / ((g11 * g22)**(1 / 2))
            print(name)
            print(i.isnull().sum().sum())
        self.ic = pd.DataFrame(res)
        
ic_rs = ic(params, y)
ic_rs.ic.mean() / ic_rs.ic.std()
ic_rs.ic.mean()
ic_rs.ic.abs().mean()
ic_df = ic_rs.ic

#########################################

ts_total = inter_all([zs_param, ic_df])
split1 = int(ts_total.shape[0] * 0.7)
split2 = int(ts_total.shape[0] * 0.85)
train_ts = ts_total[:split1]
valid_ts = ts_total[split1:split2]
test_ts = ts_total[split2:]

idx_num = 5

ic_x_train = zs_param.loc[train_ts]
ic_y_train = ic_df.loc[train_ts]. iloc[:, idx_num]

ic_x_valid = zs_param.loc[valid_ts]
ic_y_valid = ic_df.loc[valid_ts]. iloc[:, idx_num]

ic_x_test = zs_param.loc[test_ts]
ic_y_test = ic_df.loc[test_ts]. iloc[:, idx_num]

import lightgbm as lgb
lgbs1 = lgb.Dataset(ic_x_train, ic_y_train > 0)
lgbs2 = lgb.Dataset(ic_x_valid, ic_y_valid > 0)

params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
 
            'learning_rate': 0.1,
            'num_leaves':4,
            'max_depth':2,
 
            'max_bin':5,
            'min_data_in_leaf':100,
            #'feature_pre_filter': False, 
            #'feature_fraction': 0.8,
            #'bagging_fraction': 0.8,
            #'bagging_freq':1,
 
            'lambda_l1': 10,
            'lambda_l2': 10,
            'min_split_gain': 1
}

gbm = lgb.train(params,                     # 参数字典
                lgbs1,                      # 训练集
                num_boost_round=300,        # 迭代次数
                valid_sets=lgbs2,           # 验证集
                early_stopping_rounds=10)   # 早停系数

score = gbm.predict(ic_x_train, num_iteration=gbm.best_iteration)
score = pd.Series(score, index=ic_x_train.index)
from sklearn.metrics import roc_auc_score
roc_auc_score((ic_y_train > 0), score)

score = gbm.predict(ic_x_valid, num_iteration=gbm.best_iteration)
score = pd.Series(score, index=ic_x_valid.index)
from sklearn.metrics import roc_auc_score
roc_auc_score((ic_y_valid > 0), score)

score = gbm.predict(ic_x_test, num_iteration=gbm.best_iteration)
score = pd.Series(score, index=ic_x_test.index)
from sklearn.metrics import roc_auc_score
roc_auc_score((ic_y_test > 0), score)

score = gbm.predict(ic_x_test, num_iteration=gbm.best_iteration)
score = pd.Series(score, index=ic_x_test.index)
test_score_df = pd.concat([ic_y_test, score], axis=1)
test_score_df["bin"] = pd.qcut(test_score_df.iloc[:, 1], q=20, duplicates="drop")
test_score_df.groupby("bin").apply(lambda x:pd.Series({"mean": x.iloc[:, 0]. mean(), "cnt": x.shape[0]}))


score = gbm.predict(ic_x_valid, num_iteration=gbm.best_iteration)
score = pd.Series(score, index=ic_x_valid.index)
valid_score_df = pd.concat([ic_y_valid, score], axis=1)
valid_score_df["bin"] = pd.qcut(valid_score_df.iloc[:, 1], q=20, duplicates="drop")
valid_score_df.groupby("bin").apply(lambda x:pd.Series({"mean": x.iloc[:, 0]. mean(), "cnt": x.shape[0]}))


score = gbm.predict(ic_x_train, num_iteration=gbm.best_iteration)
score = pd.Series(score, index=ic_x_train.index)
train_score_df = pd.concat([ic_y_train, score], axis=1)
train_score_df["bin"] = pd.qcut(train_score_df.iloc[:, 1], q=10, duplicates="drop")
train_score_df.groupby("bin").apply(lambda x:pd.Series({"mean": x.iloc[:, 0]. mean(), "cnt": x.shape[0]}))



importance = pd.Series(gbm.feature_importance(), index=gbm.feature_name()).sort_values()

# zd_sz.399959                  1
# zd_sz.399376                  2
# zd_sz.399368                  4
# zd_sz.399608                  5
# zd_sz.399905                  5
zs_cn.loc[importance.index[importance > 0]. str[3:]]


