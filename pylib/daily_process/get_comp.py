import pandas as pd
import baostock as bs
import pickle
bs.login()
os.chdir("c:/Users/48944/finance/")

try:
    hs300_comp = pd.read_pickle("./compdata/hs300.pkl")
except:
    hs300_comp = dict()

try:
    zz500_comp = pd.read_pickle("./compdata/zz500.pkl")
except:
    zz500_comp = dict()

try:
    sz50_comp = pd.read_pickle("./compdata/zz500.pkl")
except:
    sz50_comp = dict()
    

from datetime import datetime
for i in range(2013, 2023):
    for j in range(1, 13):
        date = datetime(i, j, 1)
        print(date)
        if date > datetime.now():
            break
        date = date.strftime("%Y-%m-%d")
        if not (date in hs300_comp):
            hs300_comp[date] = bs.query_hs300_stocks(date=date).get_data()
        if not (date in zz500_comp):
            zz500_comp[date] = bs.query_zz500_stocks(date=date).get_data()
        if not (date in sz50_comp):
            sz50_comp[date] = bs.query_sz50_stocks(date=date).get_data()
            

with open("./compdata/hs300.pkl", "wb") as f:
    pickle.dump(hs300_comp, f)

with open("./compdata/sz50.pkl", "wb") as f:
    pickle.dump(sz50_comp, f)

with open("./compdata/zz500.pkl", "wb") as f:
    pickle.dump(zz500_comp, f)



