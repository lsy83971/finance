# -*- coding: utf-8 -*-
import pandas as pd
import baostock as bs
from datetime import datetime, timedelta
import sys

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


def get_k_info(info, begin_date="2013-01-01", date_end=None, freq='60', columns=["open", "close", "high", "low", "amount"], path="kdata60"):
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
    bs.login()
    if freq == '60':
        columns1 = ','. join(["time"] + columns)
        ts_time = [j for i in ts_int for j in range(i * 10, i * 10 + 4)]        
    if freq == 'd':
        columns1 = ','. join(["date"] + columns)
        ts_time = ts_int
    
    for i, j in info.iterrows():
        df = pd.DataFrame(index=ts_time, columns=columns)
        code = j["code"]
        print(code)
        t1 = j["ipoDate"]
        t1 = max(t1, begin_date)
        t2 = date_end
        df1 = bs.query_history_k_data_plus(
            code, columns1,
            t1, t2, 
            frequency=freq,
            adjustflag='1').get_data()
        
        if df1.shape[1] == 0:
            print("GG")
            continue
        
        print(df1.shape)        
        for k in ["open", "close", "high", "low", "amount"]:
            if k in df1:
                df1[k] = df1[k]. astype(float)

        if "time" in df1:
            df1["time"] = df1["time"]. str[:8]. astype(int)*10 + \
                df1["time"]. str[8:12]. apply(lambda x:t_dict.get(x, 0))
            df1 = df1.set_index("time")

        if "date" in df1:
            df1["date"] = pd.to_datetime(df1["date"]).dt.strftime("%Y%m%d").astype(int)
            df1 = df1.set_index("date")

        df.update(df1)
        for i in df.columns:
            if i == "amount":
                df[i].fillna(0, inplace=True)
            if i in ["open", "close", "high", "low"]:
                df[i].fillna(method="ffill", inplace=True)
                
        df.to_pickle(f"c:/Users/48944/finance/{path}/{code}.pkl")
