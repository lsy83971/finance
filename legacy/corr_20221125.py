from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso

alpha = 0.1
y = df3.iloc[:, 0]
x = df3.iloc[:, 1:]


self = decomp_y(alpha=alpha, x=x, y=y)
class decomp_y:
    def __init__(self, alpha, y, x):
        self.ls = Lasso(alpha=alpha, fit_intercept=False)
        self.y = y
        self.x = x
        self.ls.fit(x, y)

        self.coef = pd.Series(self.ls.coef_, index=x.columns)
        self.coef1 = self.coef[~(self.coef == 0)]

        self.r1 = self.x@self.coef
        self.r2 = self.y - self.r1
        self.dcp_r = (self.r1 ** 2).sum() / (self.y ** 2).sum()

i = 1

ps = dict()
for i in range(df3.shape[1]):
    print(i)
    ps[df3.columns[i]] = decomp_y(alpha=alpha, y=df3.iloc[:, i], x=df3.drop(df3.columns[i], axis=1))

        


