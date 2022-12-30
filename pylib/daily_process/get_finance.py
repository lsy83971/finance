# -*- coding: utf-8 -*-
import sys
sys.path.append("c:/Users/48944/finance/pylib")
from fetch.bsdata import *

stocks_basic_info = pd.read_pickle("c:/Users/48944/finance/stockdata/stocks_basic_info.pkl")
stocks_basic_info_gp = stocks_basic_info[stocks_basic_info["type"] == "1"]


fin_dir = "c:/Users/48944/finance/fin_data/"
files = os.listdir(fin_dir)

for i, j in stocks_basic_info_gp.iterrows():
    code = j["code"]
    if code + ".pkl" in files:
        continue
    y1 = int(j["ipoDate"][:4])
    df = get_finance_data(code, y1=y1, y2=2022)
    df.to_pickle(f"{fin_dir}{code}.pkl")
    print(f"{fin_dir}{code}.pkl")
    print(code)


    
