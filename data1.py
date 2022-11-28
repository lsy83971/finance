import pandas as pd
import baostock as bs
from datetime import datetime, timedelta

_is_login = False

def login_logout(f):
    def f1(*args, **kwargs):
        global _is_login
        if _is_login:
            res = f(*args, **kwargs)
            return res
        else:
            bs.login()
            _is_login = True            
            res = f(*args, **kwargs)
            bs.logout()
            _is_login = False            
            return res
    return f1
    
@login_logout
def get_trade_dt(t1, t2):
    t1 = pd.to_datetime(t1).strftime("%Y-%m-%d")
    t2 = pd.to_datetime(t2).strftime("%Y-%m-%d")
    ts = bs.query_trade_dates(t1, t2).get_data()
    return ts[ts["is_trading_day"] == "1"]["calendar_date"]

@login_logout
def get_annual_stocks(y1, y2):
    y1 = int(y1)
    y2 = int(y2)
    res = dict()
    for y in range(y1, y2 + 1):
        d = get_trade_dt(datetime(y, 1, 1), datetime(y, 2, 1)).iloc[0]
        res[y] = bs.query_all_stock(d).get_data()
    return res


@login_logout
def get_stock_basic(y1, y2):
    stocks = get_annual_stocks(y1, y2)
    codes = pd.concat(stocks.values())["code"]. drop_duplicates()
    ls = list()
    for i in codes:
        print(i)
        ls.append(bs.query_stock_basic(code=i).get_data())
    return pd.concat(ls)


def get_k_info(info, begin_date="2013-01-01", date_end=None):
    t_dict = {
        "1030": 0,
        "1130": 1,
        "1400": 2,
        "1500": 3,         
    }
    if date_end is None:
        date_end = (datetime.now() - timedelta(1)).strftime("%Y-%m-%d")
    ts = get_trade_dt(begin_date, date_end)
    ts_int = pd.to_datetime(ts).dt.strftime("%Y%m%d").astype(int)
    ts_time = [j for i in ts_int for j in range(i * 10, i * 10 + 4)]
    bs.login()
    for i, j in info.iterrows():
        df = pd.DataFrame(index=ts_time, columns=["open", "close", "high", "low", "amount"])        
        code = j["code"]
        print(code)
        t1 = j["ipoDate"]
        t1 = max(t1, begin_date)
        t2 = date_end
        df1 = bs.query_history_k_data_plus(
            code, "time,open,close,high,low,amount",
            t1, t2, 
            frequency='60',
            adjustflag='1').get_data()
        if df1.shape[1] != 6:
            print("GG")
            continue
        
        print(df1.shape)        
        for k in ["open", "close", "high", "low", "amount"]:
            df1[k] = df1[k]. astype(float)
        df1["time"] = df1["time"]. str[:8]. astype(int)*10 + \
            df1["time"]. str[8:12]. apply(lambda x:t_dict.get(x, 0))
        df1 = df1.set_index("time")

        df.update(df1)
        df.fillna(method="ffill", inplace=True)
        df.to_pickle(f"c:/Users/48944/finance/kdata60/{code}.pkl")

if __name__ == "__main__":
    #stocks_basic_info = get_stock_basic(2010, 2022)
    #stocks_basic_info.to_pickle("./stockdata/stocks_basic_info.pkl")
    stocks_basic_info = pd.read_pickle("./stockdata/stocks_basic_info.pkl")
    stocks_basic_info_gp = stocks_basic_info[stocks_basic_info["type"] == "1"]
    stocks_basic_info_zs = stocks_basic_info[stocks_basic_info["type"] == "2"]
    gp_on = stocks_basic_info_gp[stocks_basic_info_gp["outDate"] == ""]
    his = [i[: -4] for i in os.listdir("./kdata60")]
    gp_on1 = gp_on[~gp_on["code"]. isin(his)]

    len(his)
    stocks_basic_info_gp.shape
    stocks_basic_info_gp
    get_k_info(gp_on1, begin_date="2013-01-01", date_end="2022-11-23")
        

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








