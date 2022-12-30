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
["sz.399377", #�޳�С�̼�ֵָ��  
"sz.399376", #�޳�С�̳ɳ�ָ��  
"sh.000056", #��֤����  
"sz.399374", #�޳����̳ɳ�ָ��  
"sz.399373", #�޳����̼�ֵָ��  
"sz.399372", #�޳����̳ɳ�ָ��  
"sz.399006", #��ҵ��ָ��(�۸�)
"sz.399606", #��ҵ��ָ��(����)
"sz.399101", #��С��ҵ�ۺ�ָ��
"sz.399905", #��֤500ָ��
"sh.000300", #����300ָ��
"sz.399951", #����300����ָ��
"sz.399437", #��֤֤ȯ��ͷָ��
"sz.399805", #��֤����������ָ��
"sh.000928", #��֤��Դָ��
"sh.000950", #����300��������ָ��
"sz.399911", #����300��ѡ����ָ��
"sz.399912", #����300��Ҫ����ָ��
"sh.000957", #����300����ָ��
"sh.000857", #��֤500ҽҩ����ָ��
"sh.000913", #����300ҽҩ����ָ��
"sz.399608", #��֤�Ƽ�100ָ��
"sz.399368", #��֤�������ָ��
"sz.399959", #��֤�к���������ָ��
"sz.399967", #��֤����ָ��
"sz.399233", #����ҵָ��
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





