# -*- coding: utf-8 -*-
from bs_lib import *

if __name__ == "__main__":
    stocks_basic_info = get_stock_basic(2010, 2022)
    stocks_basic_info.to_pickle("./stockdata/stocks_basic_info.pkl")



