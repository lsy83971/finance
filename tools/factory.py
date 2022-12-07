from collections import OrderedDict


# class pca_factory():
#     def __init__(self, p1clp=None, p2std=None, p3pca=None):
#         if p1clp is None:
#             p1clp = sigma3()
#         if p2std is None:
#             p2std = std()
#         if p3pca is None:
#             p3pca = cls_pca(n_components=3)
#         self.p1 = p1clp
#         self.p2 = p2std
#         self.p3 = p3pca

#     def fit(self, x):
#         df1 = self.p1.fit(x)
#         df2 = self.p2.fit(df1)
#         df3 = self.p3.fit(df2)

#     def trans(self, x):
#         df1 = self.p1.trans(x)
#         df2 = self.p2.trans(df1)
#         df3 = self.p3.trans(df2)
#         return df3



class pro_factory():
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



