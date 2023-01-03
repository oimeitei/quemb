import numpy

def rdm1_fullbasis(self, return_ao=True):

    C_mo = self.C

    nao, nmo = self.C.shape
    rdm1 = numpy.zeros((nao, nao))
    
    
    for fobjs in self.Fobjs:

        cind = [ fobjs.fsites[i] for i in fobjs.efac[1]]
        C_f = fobjs.TA @ fobjs.mo_coeffs

        Cf_S_Cl = C_f.T @ self.S @ self.W[:, cind]
        Cl_S_Cf = self.W[:, cind].T @ self.S @ C_f

        Proj_c = Cf_S_Cl @ Cl_S_Cf
        rdm1_ = Proj_c @ fobjs.__rdm1
        
        Cmo_S_Cf = self.C.T @ self.S @ C_f
        rdm1_ = numpy.einsum('ij,pi,qj->pq', rdm1_, Cmo_S_Cf, Cmo_S_Cf)

        rdm1 += rdm1_

    if return_ao:
        rdm1 = self.C @ rdm1 @ self.C.T

    return rdm1
