import numpy, sys
from pyscf import ao2mo
def get_veff(eri_, dm, S, TA, hf_veff):
    import functools
    from pyscf import scf

    if dm.ndim==2:
        ST = numpy.dot(S, TA)
        P_ = functools.reduce(numpy.dot,(ST.T, dm, ST))
    
    P_ = numpy.asarray(P_.real, dtype=numpy.double)
    eri_ = numpy.asarray(eri_, dtype=numpy.double)
    vj, vk = scf.hf.dot_eri_dm(eri_, P_, hermi=1, with_j=True, with_k=True)
    Veff_ = vj - 0.5 * vk
    
    # remove core contribution from hf_veff
    if dm.ndim == 2:
        Veff = functools.reduce(numpy.dot,(TA.T, hf_veff, TA)) - Veff_
    
    return Veff

# create pyscf pbc scf object
def get_scfObj(h1, Eri, nocc, dm0=None, enuc=0.):
    # from 40-customizing_hamiltonian.py in pyscf examples
    from pyscf import gto, scf
        
    nao = h1.shape[0]


    S = numpy.eye(nao)
    mol = gto.M()

    mol.nelectron = nocc * 2
    mol.nao_nr = lambda *args : nao
    mol.energy_nuc = lambda *args : enuc
    mol.incore_anyway = True
    
    
    mf_ = scf.RHF(mol)
    mf_.get_hcore = lambda *args:h1
    mf_.get_ovlp = lambda *args: S
    mf_._eri = Eri
    mf_.incore_anyway = True
    mf_.max_cycle=50
    
    mf_.verbose=0
    
    if dm0 is None:
        mf_.kernel()
    else:
        mf_.kernel(dm0=dm0)
        
    
    if not mf_.converged:
        print(flush=True)
        print('WARNING!!! SCF not convereged - applying level_shift=0.2, diis_space=25 ',flush=True)
        print(flush=True)
        mf_.verbose=0
        mf_.level_shift=0.2
        mf_.diis_space=25    
        if dm0 is None:
            mf_.kernel()
        else:
            mf_.kernel(dm0=dm0)
        if not mf_.converged:
            print(flush=True)
            print('WARNING!!! SCF still not convereged!',flush=True)
            print(flush=True)
        else:
            print(flush=True)
            print('SCF Converged!',flush=True)
            print(flush=True)
                    
    return mf_


def get_eri(i_frag, Nao, symm = 8, ignore_symm = False, eri_file='eri_file.h5'):
    from pyscf import ao2mo, lib
    import h5py
    
    r = h5py.File(eri_file,'r')
    eri__ = numpy.array(r.get(i_frag))
    
    if not ignore_symm:
        lib.num_threads(1)
        eri__ = ao2mo.restore(symm, eri__, Nao)

    r.close()

    return eri__

def ncore_(z):
    
    if 1<= z<=2:
        nc = 0
    elif 2<=z<=5:
        nc=1
    elif 5<=z<=12:
        nc=1
    elif 12<=z<=30:
        nc=5
    elif 31<=z<=38:
        nc=9
    elif 39<=z<=48:
        nc=14
    else:
        print('Ncore not computed in helper.ncore(), add it yourself!',
              flush=True)
        print('exiting',flush=True)
        sys.exit()
        
    return nc


def get_core(mol):

    idx = []
    corelist = []
    Ncore = 0
    for ix, bas in enumerate(mol.aoslice_by_atom()):
        ncore = ncore_(mol.atom_charge(ix))
        corelist.append(ncore)
        Ncore += ncore
        idx.extend([k for k in range(bas[2]+ncore, bas[3])])

    return (Ncore,idx,corelist)



def be_energy(nfsites, h1, mo_coeffs, rdm1, rdm2s, eri_file='eri_file.h5'):
        
    rdm2s = numpy.einsum("ijkl,pi,qj,rk,sl->pqrs", 0.5*rdm2s,
                         *([mo_coeffs]*4),optimize=True)        
    
    e1 = 2.*numpy.einsum("ij,ij->i",h1[:nfsites], self._rdm1[:self.nfsites])
    ec = numpy.einsum("ij,ij->i",self.veff[:self.nfsites],self._rdm1[:self.nfsites])
    

    if self.TA.ndim == 3:
        jmax = self.TA[0].shape[1]
    else:
        jmax = self.TA.shape[1]
        
    if eri is None:
        r = h5py.File(eri_file,'r')
        eri = r[self.dname][()]
        r.close()

    
    
    e2 = numpy.zeros_like(e1)
    for i in range(self.nfsites):
        for j in range(jmax):
            ij = i*(i+1)//2+j if i > j else j*(j+1)//2+i
            Gij = rdm2s[i,j,:jmax,:jmax].copy()            
            Gij[numpy.diag_indices(jmax)] *= 0.5
            Gij += Gij.T            
            e2[i] += Gij[numpy.tril_indices(jmax)] @ eri[ij]

    e_ = e1+e2+ec        
    etmp = 0.
    for i in self.efac[1]:
        etmp += self.efac[0]*e_[i]        
    
    self.ebe = etmp
    return (e1+e2+ec)

    
