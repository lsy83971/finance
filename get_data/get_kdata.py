# -*- coding: utf-8 -*-
import sys
sys.path.append("c:/Users/48944/finance/")
from bslib import *

if __name__ == "__main__":
    #stocks_basic_info = get_stock_basic(2010, 2022)
    #stocks_basic_info.to_pickle("./stockdata/stocks_basic_info.pkl")

    lst = sys.argv[1:]
    mode = lst[0]
    d1 = pd.to_datetime(lst[1]).strftime("%Y-%m-%d")
    d2 = pd.to_datetime(lst[2]).strftime("%Y-%m-%d")
    #d1 = "2013-01-01"
    #d2 = "2022-11-23"
    freq = lst[3]
    path = lst[4]
    # kdata60zs
    # kdata60
    
    stocks_basic_info = pd.read_pickle("c:/Users/48944/finance/stockdata/stocks_basic_info.pkl")
    stocks_basic_info_gp = stocks_basic_info[stocks_basic_info["type"] == "1"]
    stocks_basic_info_zs = stocks_basic_info[stocks_basic_info["type"] == "2"]
    
    if mode == "1":
        gp_on = stocks_basic_info_gp[stocks_basic_info_gp["outDate"] == ""]
        his = [i[: -4] for i in os.listdir(f"c:/Users/48944/finance/{path}")]
        gp_on1 = gp_on[~gp_on["code"]. isin(his)]
        get_k_info(gp_on1, begin_date=d1, date_end=d2, freq=freq, path=path)

    if mode == "2":
        zs_on = stocks_basic_info_zs[stocks_basic_info_zs["outDate"] == ""]
        his = [i[: -4] for i in os.listdir(f"c:/Users/48944/finance/{path}")]
        zs_on1 = zs_on[~zs_on["code"]. isin(his)]
        get_k_info(zs_on1, begin_date=d1, date_end=d2, freq=freq, path=path)


#从这些指数中选择可以的

## 一定程度受市场环境影响
## 同时还具有个性化趋势


## 两者以何种方式叠加
## 这个问题可以研究下
## 股民/机构/国家队 等等这些都是市场参与者

## 以投资方式而言 可以以理性投资者 投机者 来区分这些参与者
## 从投资时长来看 可以分为长期投资者 短期投资者

## 这些对于市场会有多大影响？？？

## 参与者机构
## 参与者散户
## 参与者长期投资？？？

## 市场给予的一个波动
## 自身利好给予的一个波动
## 竞价上会得到体现？？？

## 卖方压力








