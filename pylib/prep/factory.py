from collections import OrderedDict
from prep.basic import * 

class pro_factory():
    """
    Example:
    pf=pro_factory(p1=sigma3(), p2=std(), p3=cls_pca(n_components=3))
    pf.fit(j0_on)

    This is used for data preprocess
    As input parameters, each one is a
    transformation.

    fit: use x to decide the coefficient of each transformation.
    trans: put x in the factory and get the result.
    """
    def __init__(self, **kwargs):
        self.kwargs = OrderedDict(kwargs)
        for i, j in self.kwargs.items():
            setattr(self, i, j)

    def fit(self, x):
        _x = x
        for i, j in self.kwargs.items():
            _x = j.fit(_x)

    def trans(self, x):
        _x = x
        for i, j in self.kwargs.items():
            _x = j.trans(_x)
        return _x



