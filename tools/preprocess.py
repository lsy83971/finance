import numpy as np
import pandas as pd

def _t_clip(f):
    def f1(self, x, *args, **kwargs):
        if len(x.shape) > 0:
            x1 = pd.Series(np.ravel(x))
        else:
            x1 = x
        f(self, x=x1, *args, **kwargs)
        return self.trans(x=x)
    return f1

class clip():
    def trans(self, x):
        return x.clip(self.lower, self.upper)
    def fit(self, x):
        raise

class mad(clip):
    @_t_clip    
    def fit(self, x, n=3):
        med = x.quantile(0.5)
        dif_med = ((x - med).abs()).quantile(0.5)
        self.upper = med + dif_med * n
        self.lower = med - dif_med * n

class sigma3(clip):
    @_t_clip    
    def fit(self, x, n=3):
        mean = x.mean()
        std = x.std()
        self.upper = mean + n * std
        self.lower = mean - n * std

class percentile(clip):
    @_t_clip        
    def percentile(x, min= 0.025, max= 0.975):
        q = x.quantile([min, max])
        self.upper = q[ - 1]
        self.lower = q[0]

class std():
    @_t_clip
    def fit(self, x):
        self.mean = x.mean()
        self.std = x.std()

    def trans(self, x):
        return (x - self.mean) / self.std
