import statsmodels.tsa.api as smt
acf=smt.stattools.acf(df, nlags=5)
pacf=smt.stattools.pacf(df, nlags=5)

type(acf)
type(pacf)























