import pandas as pd
def c1(self, word: str):
    self.columns = word + pd.Series(self.columns).astype(str)
    return self
pd.DataFrame.c1 = c1


def shift1(self, i):
    assert isinstance(i, int)
    df = self.shift(i)
    if i > 0:
        return df.iloc[i:]
    if i < 0:
        return df.iloc[: -i]
pd.DataFrame.shift1 = shift1

def sum_range(self, i1, i2=None):
    if i2 is None:
        i2 = i1
        i1 = 1
    assert i2 >= i1
    df = self.cumsum()
    df1 = df.shift(i1).fillna(0) - df.shift(i2 + 1).fillna(0)
    return df1.iloc[(i2):]
pd.DataFrame.sum_range = sum_range

def sum_geo(self, rate):
    df = self.copy()
    for i in range(1, df.shape[0]):
        df.iloc[i] = df.iloc[i - 1] * rate + self.iloc[i] * (1 - rate)
    return df
pd.DataFrame.sum_geo = sum_geo


def inter1(self, df):
    idx1 = self.index
    idx2 = df.index
    idx = idx1.intersection(idx2)
    return self.loc[idx], df.loc[idx]
pd.DataFrame.inter1 = inter1

def add1(self, df):
    df1, df2=self.inter1(df)
    return df1 + df2
pd.DataFrame.add1 = add1


def div1(self, df):
    df1, df2=self.inter1(df)
    return df1 / df2
pd.DataFrame.div1 = div1

def inter_all(l):
    for j, i in enumerate(l):
        if j == 0:
            s = i.index
        else:
            s = s.intersection(i.index)
    return s

class pds(pd.DataFrame):
    def __init__(self, df, name):
        super().__init__(df)
        self.name = name

    def loc1(self, idx):
        return pds(self.loc[idx], self.name)
        
    def inter1(self, df):
        idx1 = self.index
        idx2 = df.index
        idx = idx1.intersection(idx2)
        return self.loc1(idx), df.loc1(idx)

    def add1(self, df):
        df1, df2=self.inter1(df)
        return pds(df1 + df2, name=df1.name + "_add_" + df2.name)

    def div1(self, df):
        df1, df2=self.inter1(df)
        return pds(df1 / df2, name=df1.name + "_div_" + df2.name)

    def div2(self, i):
        return pds(self / i, name=self.name)

    def sum_range(self, i1, i2=None):
        if i2 is None:
            i2 = i1
            i1 = 1
        assert i2 >= i1
        df = self.cumsum()
        df1 = df.shift(i1).fillna(0) - df.shift(i2 + 1).fillna(0)
        return pds(df1.iloc[i2:], name=self.name + "_sr" + str(i2))

    def sum_geo(self, rate):
        df = self.copy()
        for i in range(1, df.shape[0]):
            df.iloc[i] = df.iloc[i - 1] * rate + self.iloc[i] * (1 - rate)
        return pds(df, name=self.name + "_sg" + str(rate))

    def shift1(self, i):
        assert isinstance(i, int)
        df = self.shift(i)
        if i > 0:
            df = df.iloc[i:]
        if i < 0:
            df = df.iloc[: -i]
        return pds(df, name=self.name)

    def fillna1(self, i):
        return pds(self.fillna(i), name=self.name)


