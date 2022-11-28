import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso

files = pd.Series(os.listdir("./kdata60"))

df = pd.DataFrame()

for i in files:
    code = i[:9]
    print(code)
    df1 = pd.read_pickle(f"./kdata60/{i}")
    tmp_o = df1["open"]. astype(float)
    ret_60 = (tmp_o / tmp_o.shift(1) - 1)[1:]
    df[code] = ret_60

df_na_sum = df.isnull(). sum()
df1 = df[df_na_sum.index[df_na_sum == 0]]



df2 = df1.applymap(lambda x:max(min(x, 0.1), -0.1))
df3 = df2 / df2.std()


df_corr1 = df2.iloc[:5000]. corr()
df_corr2 = df2.iloc[5000:]. corr()



#df_corr2.mean().mean()

info = pd.read_pickle("./stockdata/stocks_basic_info.pkl")
code_dict = info.set_index("code")["code_name"]
 
pca = PCA(n_components=20)
s1 = pca.fit(df3)
sb = pd.DataFrame(s1.components_, columns=df3.columns).T
sb["cn_name"] = pd.Series(sb.index).apply(lambda x:code_dict[x]).values

sb0 = sb[[0, "cn_name"]]. sort_values(0)
sb1 = sb[[1, "cn_name"]]. sort_values(1)
sb2 = sb[[2, "cn_name"]]. sort_values(2)
sb3 = sb[[3, "cn_name"]]. sort_values(3)

sb0.head(50)
sb0.tail(50)

sb1.head(50)
sb1.tail(50)

sb2.head(50)
sb2.tail(50)

sb3.head(50)
sb3.tail(50)

pd.Series(pca.explained_variance_ratio_)

df3_corr = df3.corr()

sb1_idx = sb1.head(10).index
sb1_tail_idx = sb1.tail(10).index


gg = df3@sb.iloc[:, :20]
gg.shape

lrg = LinearRegression(fit_intercept=False)

ls = dict()
for i in df3.columns:
    print(i)
    lrg.fit(gg, df3[i])
    ls[i] = lrg.coef_. tolist()

gg1 = pd.DataFrame(ls).T


gg1[1].loc[sb1.index]
gg.std()


pd.Series(pca.explained_variance_ratio_)


gg3 = (sb.iloc[:, :20] * gg.std())

(gg3.mean())
 
gg3
gg4 = gg3.copy()
gg4["cn_name"] = pd.Series(gg4.index).apply(lambda x:code_dict[x]).values


gg4[[0, "cn_name"]]. sort_values(0)
gg4[[1, "cn_name"]]. sort_values(1)
gg4[[2, "cn_name"]]. sort_values(2)


comp_ret = df3@sb.iloc[:, :20]
import matplotlib.pyplot as plt
plt.plot(comp_ret[0]. values. cumsum())

plt.plot(comp_ret[7]. values. cumsum())
plt.show()


comp_ret_new = comp_ret.copy()
comp_ret_new["new0"] = comp_ret[0] * (comp_ret[0] > 0)
comp_ret_new["new1"] = comp_ret[0] * (comp_ret[0] < 0)

comp_ret_new.corr()

comp_ret_new1 = comp_ret_new.iloc[:, 1:]
comp_ret_new2 = comp_ret_new1 / comp_ret_new1.std()

ls1 = dict()
for i in df3.columns:
    print(i)
    lrg.fit(comp_ret_new2, df3[i])
    ls1[i] = lrg.coef_. tolist()
    




df3




comp_ret_std = comp_ret / comp_ret.std()

comp_ret_std
ls_tmp = list()
for i in range( -2, 3):
    _tmp = comp_ret_std.shift(i)
    _tmp.columns = pd.Series(_tmp.columns).astype(str) + "_" + str(i)
    ls_tmp.append(_tmp)

ts2 = pd.concat(ls_tmp, axis=1).iloc[2: -2]

lrg = LinearRegression(fit_intercept=False)

df4 = df3.iloc[2: -2]

ss2 = list()
for i in df4.columns:
    lrg.fit(ts2, df4[i])
    ss1 = pd.DataFrame(np.array(lrg.coef_).reshape(5, 20))
    ss2.append(ss1)

ss3 = np.array(ss2)
pd.DataFrame(ss3[:, :, 0]).abs().max(axis=0)
pd.DataFrame(ss3[:, :, 0]).abs().mean(axis=0)
pd.DataFrame(ss3[:, :, 1]).abs().max(axis=0)
pd.DataFrame(ss3[:, :, 2]).abs().max(axis=0)
pd.DataFrame(ss3[:, :, 3]).abs().max(axis=0)
pd.DataFrame(ss3[:, :, 3]).abs().mean(axis=0)
pd.DataFrame(ss3[:, :, 4]).max(axis=0)
pd.DataFrame(ss3[:, :, 5]).max(axis=0)
pd.DataFrame(ss3[:, :, 6]).max(axis=0)
pd.DataFrame(ss3[:, :, 7]).max(axis=0)





