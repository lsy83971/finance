import sys
sys.path.append("c:/Users/48944/finance/pylib")
from prep.factory import pro_factory, sigma3, std
from stats.pca import cls_pca

class pca_comp:
    def __init__(self, rt, _comp=None):
        """
        rt: ret
        _comp: hs300,zz500 etc.
        """
        self.rt = rt
        self._comp = _comp

    def yearly(self, i1, i2, _comp=None):
        """
        give the pca_comp trained for last i1 years and transformed for the next i2 years
        """
        self.pf_record = dict()
        self.idx0_record = dict()
        self.idx1_record = dict()
        
        if _comp is None:
            _comp = self._comp
            
        for i, j in self.rt.gp1l("year", [i1, i2]):
            j0_on = j[0]. df_on()
            j1_on = j[1]. df_on()            
            if _comp is not None:
                c1 = _comp.get(j[0]. info.index[0])["code"]
                c2 = _comp.get(j[0]. info.index[ - 1])["code"]
                j0_on = j0_on[j0_on.columns.intersection(c1).intersection(c2)]
                
            pf = pro_factory(p1=sigma3(), p2=std(), p3=cls_pca(n_components=3))
            pf.fit(j0_on)
            j1_res = pf.trans(j1_on)
            self.pf_record[i] = pf
            self.idx0_record[i] = pf.p3.index
            self.idx1_record[i] = j1_res

