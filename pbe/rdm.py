import numpy, sys, scipy

def rdm1_fullbasis(self, return_ao=True, only_rdm1=False, only_rdm2=False, return_lo=False):

    C_mo = self.C.copy()
    if self.frozen_core:
        print('Try frozen_core=False')
        sys.exit()
        C_mo = self.C.copy()[:,self.ncore:]

    nao, nmo = C_mo.shape
    rdm1 = numpy.zeros((nao - self.ncore, nao - self.ncore))
    rdm2 = numpy.zeros((nao- self.ncore, nao- self.ncore, nao- self.ncore, nao- self.ncore))

    if return_lo:
        rdm1lo = numpy.zeros((nao - self.ncore, nao - self.ncore))
        rdm2lo = numpy.zeros((nao- self.ncore, nao- self.ncore, nao- self.ncore, nao- self.ncore))

    for fobjs in self.Fobjs:
        
        cind = [ fobjs.fsites[i] for i in fobjs.efac[1]]
        
        Pc_ = fobjs.TA.T @ self.S @ self.W[:, cind] @ self.W[:, cind].T @ self.S @ fobjs.TA
               
        if not only_rdm2:
            
            rdm1_eo = fobjs.mo_coeffs @ fobjs.__rdm1 @ fobjs.mo_coeffs.T
            rdm1_center = Pc_ @ rdm1_eo
            rdm1_lo = fobjs.TA @ rdm1_center @ fobjs.TA.T

            rdm1_mo = self.C.T @ self.S @ rdm1_lo @ self.S @ self.C
            rdm1 += rdm1_mo
            if return_lo: rdm1lo += rdm1_lo
            
        if not only_rdm1:

            rdm2s = numpy.einsum("ijkl,pi,qj,rk,sl->pqrs", fobjs.__rdm2,
                                 *([fobjs.mo_coeffs]*4),optimize=True)
            rdm2_lo = numpy.einsum('xi,ijkl,px,qj,rk,sl->pqrs',
                                   Pc_, rdm2s, fobjs.TA, fobjs.TA,
                                   fobjs.TA, fobjs.TA, optimize=True)
            CmoT_S = self.C.T @ self.S
            rdm2_mo = numpy.einsum("ijkl,pi,qj,rk,sl->pqrs",
                                   rdm2_lo, CmoT_S, CmoT_S,
                                   CmoT_S, CmoT_S, optimize=True)
            rdm2 += rdm2_mo
            if return_lo: rdm2lo += rdm2_lo

    if not only_rdm1:
        rdm2 = (rdm2 + rdm2.transpose(1,0,3,2))/2
        if return_ao: rdm2 = numpy.einsum('ijkl,pi,qj,rk,sl->pqrs', rdm2, *(4*[self.C]), optimize=True)
    if not only_rdm2:
        rdm1 = (rdm1 + rdm1.T)/2.
        if return_ao: rdm1 = self.C @ rdm1 @ self.C.T 

    if only_rdm1: return rdm1
    if only_rdm2: return rdm2

    if return_lo: return (rdm1, rdm2, rdm1lo, rdm2lo)
    
    return rdm1, rdm2

def get_rdm(self, approx_cumulant=False, use_full_rdm=False, return_ao=True):
    from pyscf import scf, ao2mo
    
            
    rdm1f, Kumul = self.rdm1_fullbasis()
    
    if not approx_cumulant:

        for fobjs in self.Fobjs:
            drdm1 = fobjs.__rdm1.copy()
            drdm1[numpy.diag_indices(fobjs.nsocc)] -= 2.
            dm_nc = numpy.einsum('ij,kl->ijkl', drdm1, drdm1, dtype=numpy.float64, optimize=True) - \
                0.5*numpy.einsum('ij,kl->iklj', drdm1, drdm1, dtype=numpy.float64,optimize=True)
            fobjs.__rdm2 -= dm_nc
        Kumul_T = self.rdm1_fullbasis(only_rdm2=True)
        if use_full_rdm:
            RDM2_full =  numpy.einsum('ij,kl->ijkl', rdm1f, rdm1f, dtype=numpy.float64, optimize=True) - \
                numpy.einsum('ij,kl->iklj', rdm1f, rdm1f, dtype=numpy.float64, optimize=True)*0.5
            RDM2_full += Kumul_T

    del_gamma = rdm1f - self.hf_dm        
    veff = scf.hf.get_veff(self.mol, rdm1f, hermi=0)
    Eh1 = numpy.einsum('ij,ij', self.hcore, rdm1f, optimize=True)
    EVeff = numpy.einsum('ij,ij',veff, rdm1f, optimize=True)

    Eh1_dg = numpy.einsum('ij,ij',self.hcore, del_gamma, optimize=True)
    Eveff_dg = numpy.einsum('ij,ij',self.hf_veff, del_gamma, optimize=True)
    
    eri = ao2mo.restore(1,self.mf._eri, self.mf.mo_coeff.shape[1])
    EKumul = numpy.einsum('pqrs,pqrs', eri,Kumul, optimize=True)
    EKumul_T = numpy.einsum('pqrs,pqrs', eri,Kumul_T, optimize=True)
    if use_full_rdm:
        E2 = numpy.einsum('pqrs,pqrs', eri,RDM2_full, optimize=True)
        print(E2*0.5)
    EKapprox = self.ebe_hf + Eh1_dg + Eveff_dg + EKumul/2. 
    EKtrue = Eh1 + EVeff/2. + EKumul_T/2. + self.enuc + self.E_core
    
    print(flush=True)    
    print('-----------------------------------------------------',
          flush=True)
    print(' BE ENERGIES with cumulant-based expression', flush=True)
    
    print('-----------------------------------------------------',
          flush=True)
    print(' E_BE = E_HF + Tr(F del g) + Tr(V K_approx)', flush=True)
    print(' E_HF            : {:>12.8f} Ha'.format(self.ebe_hf), flush=True)
    print(' Tr(F del g)     : {:>12.8f} Ha'.format(Eh1_dg+Eveff_dg), flush=True)
    print(' Tr(V K_aprrox)  : {:>12.8f} Ha'.format(EKumul/2.), flush=True)
    print(' E_BE            : {:>12.8f} Ha'.format(EKapprox), flush=True)
    print(' Ecorr BE        : {:>12.8f} Ha'.format(EKapprox-self.ebe_hf), flush=True)
    print(flush=True)
    print(' E_BE = Tr(F[g] g) + Tr(V K_true)', flush=True)
    print(' Tr(h1 g)        : {:>12.8f} Ha'.format(Eh1), flush=True)
    print(' Tr(Veff[g] g)   : {:>12.8f} Ha'.format(EVeff/2.), flush=True)
    print(' Tr(V K_true)    : {:>12.8f} Ha'.format(EKumul_T/2.), flush=True)
    print(' E_BE            : {:>12.8f} Ha'.format(EKtrue), flush=True)
    if use_full_rdm: print(' E(g+G)          : {:>12.8f} Ha'.format(Eh1 + 0.5*E2 + self.E_core + self.enuc),
                           flush=True)
    print(' Ecorr BE        : {:>12.8f} Ha'.format(EKtrue-self.ebe_hf), flush=True)
    print(flush=True)
    print(' True - approx   : {:>12.4e} Ha'.format(EKtrue-EKapprox))
    print('-----------------------------------------------------',
          flush=True)
    
    print(flush=True)

    if return_ao: return(rdm1f, RDM2_full)
    rdm1f, Kumul = self.rdm1_fullbasis(return_ao =False)    
    Kumul_T = self.rdm1_fullbasis(only_rdm2=True, return_ao=False)
    
    RDM2_full =  numpy.einsum('ij,kl->ijkl', rdm1f, rdm1f, optimize=True) - \
        numpy.einsum('ij,kl->iklj', rdm1f, rdm1f, optimize=True)*0.5
    RDM2_full += Kumul_T
    
    return(rdm1f, RDM2_full)
