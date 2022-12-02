# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso
import re
from price_struct import *

from importlib import reload
import price_struct
reload(price_struct)
from price_struct import *


stock_files = os.listdir("./kdata60")


dfs = [pd.read_pickle(f"./kdata60/{i}") for i in stock_files]
amount_df = pd.concat([i["amount"] for i in dfs], axis=1)
amount_df.columns = [i[: -4] for i in stock_files]

close_df = pd.concat([i["close"] for i in dfs], axis=1)
close_df.columns = [i[: -4] for i in stock_files]

amt60 = amount_ts(amount_df)
amtd = ats.end(["year", "month", "day"])


basic_info = pd.read_pickle("./stockdata/stocks_basic_info.pkl")
zs_info = basic_info[basic_info["type"] == '2']

cn_name = basic_info.set_index("code")["code_name"]
hs300 = comp(pd.read_pickle("./compdata/hs300.pkl"))

zs_files = [i for i in os.listdir("./kdata60zs") if '.pkl' in i]
close_zs = [pd.read_pickle(f"./kdata60zs/{i}")["close"] for i in zs_files]
close_zs_df = pd.concat(close_zs, axis=1)
close_zs_df.columns = [i[: -4] for i in zs_files]

check_price(close_df)
check_price(close_zs_df)
# 1. 对比PCA hs300-fit/total-fit 得到主特征的有效性
from decompse import factor

sp60 = price_ts(close_df)
ret60 = sp60.ret()
spd = sp60.end(["year", "month", "day"])
retd = spd.ret()
retd_df = retd.df_on()

zs_priced = price_ts(close_zs_df)
zs_retd = zs_priced.ret()
zs_retd_df = zs_retd.df_on()




### 找出一个有效的beta
basis = ["sz.399905", "sz.399300", "sh.000016"]
basis = ["sz.399300"]
basis = ["sh.000001", "sz.399001", "sz.399905", "sz.399101", "sz.399102"]
basis = ["sh.000001", "sz.399001", "sz.399101", "sz.399102"]
basis = ["sh.000001", "sz.399101"]

for y_gp, y_zs in zip(retd.gp1l("year", [4, 1]), zs_retd.gp1l("year", [4, 1])):
    break


df_gp = y_gp[1][0]. df_on()
df_gp1 = y_gp[1][1]. df_on()





idx = df_gp1.columns & df_gp.columns
df_gp = df_gp[idx]
df_gp1 = df_gp1[idx]



df_zs = y_zs[1][0]. df_on()[basis]
df_zs1 = y_zs[1][1]. df_on()[basis]

mask = amtd.info > 0


ff = factor(df_zs)
ff.factor_mask(df_gp, mask)

compm1 = ((ff.comp1 * ff.mask.values)**2).mean()
compm2 = ((ff.comp2 * ff.mask.values)**2).mean()
res = compm2 / (compm1 + compm2)

df_zs1.index = df_gp1.index
ff.trans(df_zs1)
compm2_1 = ff.trans_comp * mask.loc[ff.trans_comp.index, ff.trans_comp.columns]
res1 = ((df_gp1 - compm2_1)**2).mean() / (df_gp1**2).mean()

codes = hs300.get(201701030)["code"]

res[res.index & codes]. mean()
res1[res1.index & codes]. mean()



pca0 = cls_pca(df_gp[df_gp.columns & codes])


comp1 = df_gp1[df_gp1.columns & codes]@pca0.comp
comp0 = df_gp[df_gp.columns & codes]@pca0.comp

df_gp300 = df_gp[df_gp.columns & codes]
df_gp1_300 = df_gp1[df_gp1.columns & codes]

i = 6
ff1 = factor(comp0.iloc[:, :i])
ff1.factor_mask(df_gp300, mask)


ff1.trans(comp0.iloc[:, :i])
compm3_0 = ff1.trans_comp * mask.loc[ff1.trans_comp.index, ff1.trans_comp.columns]
res0 = ((df_gp300 - compm3_0)**2).mean() / (df_gp300**2).mean()
ff1.trans(comp1.iloc[:, :i])
compm3_1 = ff1.trans_comp * mask.loc[ff1.trans_comp.index, ff1.trans_comp.columns]
res1 = ((df_gp1_300 - compm3_1)**2).mean() / (df_gp1_300**2).mean()
print(res0.mean())
print(res1.mean())



pd.set_option("display.max_rows", 500)
zs_info[["code", "code_name"]]. head(500)

zs_info[zs_info["code_name"]. str.contains("深证")]
zs_info[zs_info["code_name"]. str.contains("中小")]
zs_info[zs_info["code_name"]. str.contains("创业")]
