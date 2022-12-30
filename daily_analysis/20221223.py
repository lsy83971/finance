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
    zd_sumrange10]


zs_param = pd.concat([i.c1(i.name + "_") for i in zs_params], axis=1).sort_index().loc[inter_all(zs_params)]
zs_target = zd

#################################################
## 2d variable

rtd.index_cut(8)
amtd.index_cut(8)

rdf = pds(rtd.df_on(), name="rdf")
adf = pds(amtd.info[rdf.columns], "adf")
adf_avg10 = adf.sum_range(10).div2(10)
adf_avg10_divadf = (adf.div1(adf_avg10)).fillna1(1).shift1(1).clip1(0.1, 10)

adf_avg5 = adf.sum_range(5).div2(5)
adf_avg5_divadf = (adf.div1(adf_avg5)).fillna1(1).shift1(1).clip1(0.1, 10)

rdf_sumrange1 = rdf.sum_range(1)
rdf_sumrange5 = rdf.sum_range(5).div2(5)
rdf_sumrange10 = rdf.sum_range(10).div2(10)

params =\
[
    adf_avg10_divadf,
    adf_avg5_divadf,
    rdf_sumrange1, 
    rdf_sumrange5,
    rdf_sumrange10,
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
ic_df = ic_rs.ic

#########################################


train_d1param_l = \
[
 
]

test_d1param_l = \
[

]

zs_param

train_d1param = pds(pd.concat(train_d1param_l, axis=1), name="train").shift1(1)
test_d1param = pds(pd.concat(test_d1param_l, axis=1), name="test").shift1(1)

train_ts = inter_all([train_d1param, ic_df])
test_ts = inter_all([test_d1param, ic_df])

ic_y_train = ic_df.loc[train_ts]
ic_y_test = ic_df.loc[test_ts]

ic_x_train = train_d1param.loc[train_ts]
ic_x_test = test_d1param.loc[test_ts]

ic_y_train_idx = ic_y_train.iloc[:, 3]
ic_y_test_idx = ic_y_test.iloc[:, 3]



import lightgbm as lgb
lgbs1 = lgb.Dataset(ic_x_train, ic_y_train_idx > 0)
lgbs2 = lgb.Dataset(ic_x_test, ic_y_test_idx > 0)

params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
 
            'learning_rate': 0.1,
            'num_leaves':4,
            'max_depth':2,
 
            'max_bin':5,
            'min_data_in_leaf':50,
            #'feature_pre_filter': False, 
            #'feature_fraction': 0.8,
            #'bagging_fraction': 0.8,
            #'bagging_freq':1,
 
            'lambda_l1': 10,
            'lambda_l2': 10,
            'min_split_gain': 0.5
}

gbm = lgb.train(params,                     # 参数字典
                lgbs1,                      # 训练集
                num_boost_round=300,        # 迭代次数
                valid_sets=lgbs2,           # 验证集
                early_stopping_rounds=10)   # 早停系数

score = gbm.predict(ic_x_train, num_iteration=gbm.best_iteration)
score = pd.Series(score, index=ic_x_train.index)
from sklearn.metrics import roc_auc_score
roc_auc_score((ic_y_train_idx > 0), score)

score = gbm.predict(ic_x_test, num_iteration=gbm.best_iteration)
score = pd.Series(score, index=ic_x_test.index)
from sklearn.metrics import roc_auc_score
roc_auc_score((ic_y_test_idx > 0), score)

test_score_df = pd.concat([ic_y_test_idx, score], axis=1)
test_score_df["bin"] = pd.qcut(test_score_df.iloc[:, 1], q=200, duplicates="drop")
test_score_df.groupby("bin").apply(lambda x:x.iloc[:, 0]. mean())
test_score_df.groupby("bin").apply(lambda x:x.shape[0])


pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 20)
pd.Series(dir(gbm))

# threshold            3.592479
# decision_type              <=

gbm.trees_to_dataframe()[["split_feature", "threshold", "decision_type"]]
gbm.trees_to_dataframe()

# 计算IC
# 将y分解成为alpha+beta
# alpha和beta分别进行预测
# alpha
# 1.量价指标
# beta
# 1.lgb进行预测 政策面-基本面-技术面
# alpha+beta
# 1.真实价位回归模型 股票有一个隐藏真实价位(波动率更小)
# 股价会回归这个真实价位

##################################

pf300 = pca_comp(rtd, hs300)
pf500 = pca_comp(rtd, zz500)
pf50 = pca_comp(rtd, sz50)
pf_total = pca_comp(rtd)

i1 = 3
i2 = 1
pf300.yearly(i1, i2)
pf500.yearly(i1, i2)
pf50.yearly(i1, i2)
pf_total.yearly(i1, i2)

train_d1param_l = \
[
 pf300.idx0_record[2018]. c1("hs300_"), 
 pf500.idx0_record[2018]. c1("zz500_"),
 pf50.idx0_record[2018]. c1("sz50_"), 
 pf_total.idx0_record[2018]. c1("total_"), 
]

test_d1param_l = \
[
 pf300.idx1_record[2018]. c1("hs300_"), 
 pf500.idx1_record[2018]. c1("zz500_"), 
 pf50.idx1_record[2018]. c1("sz50_"), 
 pf_total.idx1_record[2018]. c1("total_"), 
]

train_d1param = pds(pd.concat(train_d1param_l, axis=1), name="train").shift1(1)
test_d1param = pds(pd.concat(test_d1param_l, axis=1), name="test").shift1(1)

train_ts = inter_all([train_d1param, ic_df])
test_ts = inter_all([test_d1param, ic_df])

ic_y_train = ic_df.loc[train_ts]
ic_y_test = ic_df.loc[test_ts]

ic_x_train = train_d1param.loc[train_ts]
ic_x_test = test_d1param.loc[test_ts]

ic_y_train_idx = ic_y_train.iloc[:, 2]
ic_y_test_idx = ic_y_test.iloc[:, 2]

import lightgbm as lgb
lgbs1 = lgb.Dataset(ic_x_train, ic_y_train_idx > 0)
lgbs2 = lgb.Dataset(ic_x_test, ic_y_test_idx > 0)

params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
 
            'learning_rate': 0.1,
            'num_leaves':4,
            'max_depth':2,
 
            'max_bin':5,
            'min_data_in_leaf':50,
            #'feature_pre_filter': False, 
            #'feature_fraction': 0.8,
            #'bagging_fraction': 0.8,
            #'bagging_freq':1,
 
            'lambda_l1': 10,
            'lambda_l2': 10,
            'min_split_gain': 0.5
}

gbm = lgb.train(params,                     # 参数字典
                lgbs1,                      # 训练集
                num_boost_round=300,        # 迭代次数
                valid_sets=lgbs2,           # 验证集
                early_stopping_rounds=10)   # 早停系数

score = gbm.predict(ic_x_train, num_iteration=gbm.best_iteration)
score = pd.Series(score, index=ic_x_train.index)
from sklearn.metrics import roc_auc_score
roc_auc_score((ic_y_train_idx > 0), score)

score = gbm.predict(ic_x_test, num_iteration=gbm.best_iteration)
score = pd.Series(score, index=ic_x_test.index)
from sklearn.metrics import roc_auc_score
roc_auc_score((ic_y_test_idx > 0), score)

test_score_df = pd.concat([ic_y_test_idx, score], axis=1)
test_score_df["bin"] = pd.qcut(test_score_df.iloc[:, 1], q=200, duplicates="drop")
test_score_df.groupby("bin").apply(lambda x:x.iloc[:, 0]. mean())
test_score_df.groupby("bin").apply(lambda x:x.shape[0])

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 20)
pd.Series(dir(gbm))

# threshold            3.592479
# decision_type              <=

gbm.trees_to_dataframe()[["split_feature", "threshold", "decision_type"]]
gbm.trees_to_dataframe()

# 计算IC

# 将y分解成为alpha+beta
# alpha和beta分别进行预测

# alpha
# 1.量价指标

# beta
# 1.lgb进行预测 政策面-基本面-技术面

# alpha+beta
# 1.真实价位回归模型 股票有一个隐藏真实价位(波动率更小)
#   股价会回归这个真实价位









