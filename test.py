import pandas as pd
import baostock as bs
pd.set_option("display.max_rows", 500)
bs.login()

## 1.1 ��ȡ���й�Ʊ
date = "2022-05-27"
stock_df = bs.query_all_stock(date).get_data()

## 1.2 ��ȡhs300��Ʊ�ɷ�
stock_300 = bs.query_hs300_stocks("2021-11-01").get_data()



## 2.��ȡK������

code = "sh.600000"
# data_fields = """date,open,high,low,close,preclose,volume,
# amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ, psTTM,pcfNcfTTM,isST"""
data_fields = "date,time,code,open,high,low,close,volume,amount,adjustflag"
start_date = "2022-05-21"
end_date = "2022-05-28"

## http://baostock.com/baostock/index.php/A%E8%82%A1K%E7%BA%BF%E6%95%B0%E6%8D%AE
## ����ָ��
## ��������	��������	˵��
## date	��������������	��ʽ��YYYY-MM-DD
## code	֤ȯ����	��ʽ��sh.600000��sh���Ϻ���sz������
## open	���̼۸�	���ȣ�С�����4λ����λ�������Ԫ
## high	��߼�	���ȣ�С�����4λ����λ�������Ԫ
## low	��ͼ�	���ȣ�С�����4λ����λ�������Ԫ
## close	�����̼�	���ȣ�С�����4λ����λ�������Ԫ
## preclose	�������̼�	���ȣ�С�����4λ����λ�������Ԫ
## volume	�ɽ�����	��λ����
## amount	�ɽ����	���ȣ�С�����4λ����λ�������Ԫ
## adjustflag	��Ȩ״̬	����Ȩ��ǰ��Ȩ����Ȩ
## turn	������	���ȣ�С�����6λ����λ��%
## tradestatus	����״̬	1���������� 0��ͣ��
## pctChg	�ǵ������ٷֱȣ�	���ȣ�С�����6λ
## peTTM	������ӯ��	���ȣ�С�����6λ
## psTTM	����������	���ȣ�С�����6λ
## pcfNcfTTM	����������	���ȣ�С�����6λ
## pbMRQ	�о���	���ȣ�С�����6λ
## isST	�Ƿ�ST	1�ǣ�0��

# ������ָ�꣺date,time,code,open,high,low,close,volume,amount,adjustflag
# ������ָ�꣺date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg

## adjustflag����Ȩ���ͣ�Ĭ�ϲ���Ȩ��3��1����Ȩ��2��ǰ��Ȩ��
## ��֧�ַ����ߡ����ߡ����ߡ�����ǰ��Ȩ��
## BaoStock�ṩ�����ǵ�����Ȩ�㷨��Ȩ����adjustflag = "2"
## http://baostock.com/baostock/index.php/%E5%A4%8D%E6%9D%83%E5%9B%A0%E5%AD%90%E7%AE%80%E4%BB%8B

kdata_df = bs.query_history_k_data_plus(code,
                                        data_fields,
                                        start_date=start_date,
                                        end_date=end_date,
                                        frequency='5',
                                        adjustflag='2').get_data()


df3 = bs.query_history_k_data_plus("sh.600000", "date,open,close,adjustflag,tradestatus",
                                   "2008-01-01", "2022-11-20",
                                   frequency="d", adjustflag='3').get_data()
df2 = bs.query_history_k_data_plus("sh.600000", "date,open,close,adjustflag,tradestatus",
                                   "2008-01-01", "2022-11-20",
                                   frequency="d", adjustflag='2').get_data()
df1 = bs.query_history_k_data_plus("sh.600000", "date,open,close,adjustflag,tradestatus",
                                   "2008-01-01", "2022-11-20",
                                   frequency="d", adjustflag='1').get_data()




class stock_dataset:
    def __init__(self, stocks):
        self.stocks = stocks

    def data(t1, t2):
        
        









df2["open"] = df2["open"]. astype(float).round(3)
df2["close"] = df2["close"]. astype(float).round(3)
df3["open"] = df3["open"]. astype(float)
df3["close"] = df3["close"]. astype(float)

df2["tradestatus"]. value_counts()



def type_trans(df):
    for i in ["open", "close", "high", "low"]:
        if i in df.columns:
            df[i] = df[i]. astype(float)

    for i in ["tradestatus"]:
        if i in df.columns:
            df[i] = df[i]. astype(int)
            
    for i in ["date"]:
        if i in df.columns:
            df[i] = pd.to_datetime(df[i])



stock_300 = bs.query_hs300_stocks("2016-01-01").get_data()
data_start = "2016-03-01"
data_end = "2021-01-01"

import numpy as np
np.array([df1.values, df1.values])


_np_list = list()
_np_stock = list()
for i in stock_300["code"]:
    print(i)
    df1 = bs.query_history_k_data_plus(i, "date,open,close,tradestatus",
                                       "2016-01-01", "2022-01-01",
                                       frequency="d", adjustflag='1').get_data()
    type_trans(df1)
    if (df1.shape[0] == 1461):
        _np_stock.append(i)
        _np_list.append(df1.values)

k_day_info = np.array(_np_list)

###########################

on_mask = k_day_info[:, :, 3]
ret = (k_day_info[:, :, 2] / k_day_info[:, :, 1] - 1).astype(np.float)
ret_mask = (ret * on_mask).astype(np.float)
pd_ret_mask = pd.DataFrame(ret_mask, index=_np_stock)
pd_ret_mask.loc["sh.600518"]


cov = ret_mask@(ret_mask.T)
diag = (np.diag(cov))**(1 / 2)
corr = ((cov / diag).T / diag).T
corr_inv = np.linalg.inv(corr)

np.diag(corr_inv).max()
pd_s_inv = pd.Series(np.diag(corr_inv), index=_np_stock).sort_values()
pd_on_mask = pd.DataFrame(on_mask, index=_np_stock)
pd_on_mask.sum(axis=1).sort_values()

pd_df_corr_inv = pd.DataFrame(corr_inv, index=_np_stock, columns=_np_stock)
pd_s_diag = pd.Series(diag, index=_np_stock)


###############
##
code = "sh.600066"
code = "sh.600998"
code = "sh.600315"
(pd_ret_mask == 0).sum(axis=1)[pd_s_inv.index]. head(50)
############



coef = pd_df_corr_inv[code] / (pd_df_corr_inv.loc[code, code])
rsd = coef@pd_ret_mask
ret_shift_20 = pd.concat([rsd.shift(i) for i in range(20)], axis=1).iloc[20:]
ret_shift_20.corr()

ret_shift_20[ - 200:- 100:].corr()
ret_shift_20[ - 300:- 200:].corr()
ret_shift_20[ - 400:- 300:].corr()
ret_shift_20[ - 500:- 400:].corr()
ret_shift_20[ - 600:- 500:].corr()
ret_shift_20[ - 700:- 600:].corr()
ret_shift_20[ - 800:- 700:].corr()
ret_shift_20[ - 900:- 800:].corr()
ret_shift_20[ - 1000:- 900:].corr()
ret_shift_20[ - 1100:- 1000:].corr()
ret_shift_20[ - 1200:- 1100:].corr()

-0.009894 / 0.3


a = np.random.rand(1400)
a1 = np.random.rand(1400)
pd.DataFrame(np.array([a, a1])).T.corr()







































