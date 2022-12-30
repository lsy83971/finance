import sys
import os
sys.path.append("c:/Users/48944/finance/pylib")
from factor.ts import *
from stats.pca_comp import pca_comp
from sklearn.metrics import roc_auc_score
import tools.pd_tools
pd.set_option("display.max_rows", 600)

stock_files = os.listdir("c:/Users/48944/finance/kdata60")
zs_files = os.listdir("c:/Users/48944/finance/kdatazs")

##############
# 1.basic info

basic_info = pd.read_pickle("c:/Users/48944/finance/stockdata/stocks_basic_info.pkl").set_index("code")
basic_cn = basic_info["code_name"]
stock_basic_info = basic_info.loc[[i[: -4] for i in stock_files]]
zs_basic_info = basic_info.loc[[i[: -4] for i in zs_files]]
stock_cn = basic_cn.loc[[i[: -4] for i in stock_files]]
zs_cn = basic_cn.loc[[i[: -4] for i in zs_files]]

###############
# 2. amount_df,close_df

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

##################
# 3. components stocks

hs300 = comp(pd.read_pickle("c:/Users/48944/finance/compdata/hs300.pkl"))
sz50 = comp(pd.read_pickle("c:/Users/48944/finance/compdata/sz50.pkl"))
zz500 = comp(pd.read_pickle("c:/Users/48944/finance/compdata/zz500.pkl"))

##################
# 4. components

zs_selected = \
["sz.399377", #巨潮小盘价值指数  
"sz.399376", #巨潮小盘成长指数  
"sh.000056", #上证国企  
"sz.399374", #巨潮中盘成长指数  
"sz.399373", #巨潮大盘价值指数  
"sz.399372", #巨潮大盘成长指数  
"sz.399006", #创业板指数(价格)
"sz.399606", #创业板指数(收益)
"sz.399101", #中小企业综合指数
"sz.399905", #中证500指数
"sh.000300", #沪深300指数
"sz.399951", #沪深300银行指数
"sz.399437", #国证证券龙头指数
"sz.399805", #中证互联网金融指数
"sh.000928", #中证能源指数
"sh.000950", #沪深300基建主题指数
"sz.399911", #沪深300可选消费指数
"sz.399912", #沪深300主要消费指数
"sh.000957", #沪深300运输指数
"sh.000857", #中证500医药卫生指数
"sh.000913", #沪深300医药卫生指数
"sz.399608", #国证科技100指数
"sz.399368", #国证航天军工指数
"sz.399959", #中证中航军工主题指数
"sz.399967", #中证军工指数
"sz.399233", #制造业指数
]
# zs_basic_info.iloc[0]
# zs_basic_info.sort_values("ipoDate")
zs_info = [pd.read_pickle("c:/Users/48944/finance/kdatazs/" + i + ".pkl") for i in zs_selected]
zs_close_df = pd.concat([i["close"] for i in zs_info], axis=1)
zs_amt_df = pd.concat([i["amount"] for i in zs_info], axis=1)

zs_close_df.columns = zs_selected
zs_amt_df.columns = zs_selected

zs_sp = price_ts(zs_close_df)
zs_amt = amount_ts(zs_amt_df)
zs_ret = zs_sp.ret()





