import pandas as pd
import numpy as np
import baostock as bs

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

def get_stocks_d(stocks, t1, t2):
    bs.login()
    ts = bs.query_trade_dates(t1, t2).get_data()
    ts = ts[ts["is_trading_day"] == "1"]
    assert ts.shape[0] > 0
    t1 = ts.iloc[0, 0]
    t2 = ts.iloc[ - 1, 0]
    err_stocks = list()
    start_date = dict()
    end_date = dict()
    _np_list = list()
    _np_stocks = list()
    for i in stocks:
        print(i)
        df1 = bs.query_history_k_data_plus(
            i, "date,open,close,tradestatus",
            t1, t2, 
            frequency="d",
            adjustflag='1').get_data()
        if (df1.shape[0] == 0):
            err_stocks.append(i)
            start_date[i] = None
            end_date[i] = None
            continue

        start_date[i] = df1["date"]. iloc[0]
        end_date[i] = df1["date"]. iloc[ - 1]
        if not (df1["date"]. iloc[0] == t1):
            err_stocks.append(i)
            continue
        if not (df1["date"]. iloc[ - 1] == t2):
            err_stocks.append(i)
            continue
        if not df1.shape[0] == ts.shape[0]:
            err_stocks.append(i)
            continue
        
        type_trans(df1)
        _np_list.append(df1.values)
        _np_stocks.append(i)

    k_day_info = np.array(_np_list)

    bs.logout()
    return {"data": k_day_info, "start": start_date, "end": end_date,
            "err": err_stocks, "t1": t1, "t2": t2, "stocks": _np_stocks, "ts": ts.iloc[:, 0]}

# res = get_stocks_d(stocks, t1, t2)
# self = stock_dataset(stocks, t1, t2)

class stock_dataset:
    def __init__(self, stocks, t1, t2):
        res = get_stocks_d(stocks, t1, t2)
        stocks = res["stocks"]
        ts = res["ts"]
        self.stocks = stocks
        self.ts = ts
        self.df_open = pd.DataFrame(res["data"][:, :, 1], columns=ts, index=stocks).astype(float)
        self.df_close = pd.DataFrame(res["data"][:, :, 2], columns=ts, index=stocks).astype(float)
        self.df_mask = pd.DataFrame(res["data"][:, :, 3], columns=ts, index=stocks).astype(bool)
        self.ret_d = (self.df_close / self.df_open - 1)

        cov = self.ret_d@self.ret_d.T
        diag = np.diag(cov)**(1 / 2)
        corr = (cov / diag).T / diag
        corr_inv = pd.DataFrame(np.linalg.inv(corr), index=stocks, columns=stocks)

        self.corr = corr
        self.corr_inv = corr_inv
        self.corr_inv_diag = pd.Series(np.diag(self.corr_inv), index=stocks)


class stock_dataset:
    def __init__(self, stocks, t1, t2):
        res = get_stocks_d(stocks, t1, t2)
        stocks = res["stocks"]
        ts = res["ts"]
        self.stocks = stocks
        self.ts = ts
        self.df_open = pd.DataFrame(res["data"][:, :, 1], columns=ts, index=stocks).astype(float)
        self.df_close = pd.DataFrame(res["data"][:, :, 2], columns=ts, index=stocks).astype(float)
        self.df_mask = pd.DataFrame(res["data"][:, :, 3], columns=ts, index=stocks).astype(bool)
        self.ret_d = (self.df_close / self.df_open - 1)

        cov = self.ret_d@self.ret_d.T
        diag = np.diag(cov)**(1 / 2)
        corr = (cov / diag).T / diag
        corr_inv = pd.DataFrame(np.linalg.inv(corr), index=stocks, columns=stocks)

        self.corr = corr
        self.corr_inv = corr_inv
        self.corr_inv_diag = pd.Series(np.diag(self.corr_inv), index=stocks)
        

