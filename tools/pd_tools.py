import pandas as pd


def c1(self, word: str):
    self.columns = word + pd.Series(self.columns).astype(str)

pd.DataFrame.c1 = c1

