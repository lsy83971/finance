# -*- coding: utf-8 -*-
import sys
sys.path.append("c:/Users/48944/finance/")
from bslib import *
if __name__ == "__main__":
    stocks_basic_info = get_stock_basic(2010, 2022)
    stocks_basic_info.to_pickle("c:/Users/48944/finance/stockdata/stocks_basic_info.pkl")



