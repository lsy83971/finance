import sys
sys.path.append("c:/Users/48944/finance/pylib")
from tools.pd_tools import pds, inter_all
from env.stockd import *


hs300 = comp(pd.read_pickle("c:/Users/48944/finance/compdata/hs300.pkl"))
sz50 = comp(pd.read_pickle("c:/Users/48944/finance/compdata/sz50.pkl"))
zz500 = comp(pd.read_pickle("c:/Users/48944/finance/compdata/zz500.pkl"))
pf300 = pca_comp(rtd, hs300)
pf500 = pca_comp(rtd, zz500)
pf50 = pca_comp(rtd, sz50)
pf_total = pca_comp(rtd)

i1 = 3
i2 = 1
pf300.yearly(i1, i2)
pf500.yearly(i1, i2)
pf50.yearly(i1, i2)
pf_total.yearly(i1, i2)

rdf = pds(rtd.df_on(), name="rdf")
adf = pds(amtd.info[rtd_df.columns], "adf")
adf_avg10 = adf.sum_range(10).div2(10)
adf_avg10_divadf = (adf.div1(adf_avg10)).fillna1(1).shift1(1)

adf_avg5 = adf.sum_range(5).div2(5)
adf_avg5_divadf = (adf.div1(adf_avg5)).fillna1(1).shift1(1)

rdf_sumrange5 = rdf.sum_range(5).div2(5).shift1(1)
rdf_sumrange10 = rdf.sum_range(10).div2(5).shift1(1)

xs = [
]

params =\
[
    adf_avg10_divadf,
    adf_avg5_divadf,
    rdf_sumrange5,
    rdf_sumrange10,
]


ts = inter_all(factors)






