import numpy, sys, functools, h5py
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
    #mol.nao_nr = lambda *args : nao
    #mol.energy_nuc = lambda *args : enuc
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

def get_eri(i_frag, Nao, symm = 8, ignore_symm = False, eri_file='eri_file.h5', eri_files=None, unrestricted=False, spin_ind=None):
    from pyscf import ao2mo, lib
    import h5py
    
    if eri_files:
        r = h5py.File(eri_files[i_frag],'r')
    else:
        r = h5py.File(eri_file,'r')
    eri__ = numpy.array(r.get(i_frag))

    if not ignore_symm:
        lib.num_threads(1)
        if unrestricted:
            eri__ = ao2mo.restore(symm, eri__, Nao)
        else:
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
    elif 49<=z<=56:
        nc=18
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

    self.ebe += etmp
    return (e1+e2+ec)

    
def get_frag_energy_u(mo_coeffs, nsocc, nfsites, efac, TA, h1, hf_veff, rdm1, rdm2s, dname, 
                    eri_file='eri_file.h5',eri_files=None, gcores=None, frozen=False):

    rdm1s_rot = [mo_coeffs[s] @ rdm1[s] @ mo_coeffs[s].T for s in [0,1] ]# for unrestricted, removing factor * 0.5

    hf_1rdm = [numpy.dot(mo_coeffs[s][:,:nsocc[s]],
                       mo_coeffs[s][:,:nsocc[s]].conj().T) for s in [0,1]]

    delta_rdm1 = [2 * (rdm1s_rot[s] - hf_1rdm[s]) for s in [0,1]]

    veff0 = [functools.reduce(numpy.dot,(TA[s].T,hf_veff[s],TA[s])) for s in [0,1]]

    if frozen:
        for s in [0,1]:
            veff0[s] -= gcores[s]
            h1[s] -= gcores[s]

    e1 = [numpy.einsum("ij,ij->i",h1[s][:nfsites[s]], delta_rdm1[s][:nfsites[s]]) for s in [0,1]]
    ec = [numpy.einsum("ij,ij->i",veff0[s][:nfsites[s]], delta_rdm1[s][:nfsites[s]]) for s in [0,1]]
   
    jmax = [TA[0].shape[1],TA[1].shape[1]]

    if eri_files:
        r = h5py.File(eri_files[dname[0]],'r')
        Vs = r[dname[0]][()]
    else:
        r = h5py.File(eri_file,'r')
        Vs = [r[dname[0]][()],r[dname[1]][()],r[dname[2]][()]]
    r.close()

    rdm2s_k = [numpy.einsum("ijkl,pi,qj,rk,sl->pqrs", rdm2s[s],
                            *([mo_coeffs[s12[0]]]*2+[mo_coeffs[s12[1]]]*2), optimize=True)
                            for s,s12 in zip([0,1,2],[[0,0],[0,1],[1,1]])]

    # From Frankenstein!
    e2 = [numpy.zeros(h1[0].shape[0]),numpy.zeros(h1[1].shape[0])]

    def contract_2e(jmaxs, rdm2_, V_, s, sym):
        e2_ = numpy.zeros(nfsites[s])
        jmax1,jmax2 = [jmaxs]*2 if isinstance(jmaxs,int) else jmaxs
        for i in range(nfsites[s]):
            for j in range(jmax1):
                # By default, Vs is stored in the "compact" format. S$
                ij = i*(i+1)//2+j if i > j else j*(j+1)//2+i
                if sym in [4,2]:
                    Gij = rdm2_[i,j,:jmax2,:jmax2].copy()
                    Vij = V_[ij]
                else:
                    Gij = rdm2_[:jmax2,:jmax2,i,j].copy()
                    Vij = V_[:,ij]
                Gij[numpy.diag_indices(jmax2)] *= 0.5
                Gij += Gij.T
                e2_[i] += Gij[numpy.tril_indices(jmax2)] @ Vij
        e2_ *= 0.5

        return e2_

    # the first nf are frag sites
    e2ss = [0.,0.]
    e2os = [0.,0.]

    for s in [0,1]:
        e2ss[s] += contract_2e(jmax[s], rdm2s_k[2*s], Vs[s], s, sym=4)
    V = Vs[2]

    # ab
    e2os[0] += contract_2e(jmax, rdm2s_k[1], V, 0, sym=2)
    # ba
    e2os[1] += contract_2e(jmax[::-1], rdm2s_k[1], V, 1, sym=-2)

    e2 = sum(e2ss) + sum(e2os)
    #ending frankenstein

    e_ = e1+e2+ec        
    etmp = 0.
    e1_tmp = 0.
    e2_tmp = 0.
    ec_tmp = 0.

    for i in efac[0][1]:
        e2_tmp += efac[0][0]*e2[i]
        for s in [0,1]:
            etmp += efac[s][0]*e_[s][i]
            e1_tmp += efac[s][0]*e1[s][i]
            ec_tmp += efac[s][0]*ec[s][i]

    print("fragment number", dname[0].split('/')[0])
    print("e1_tmp",e1_tmp)
    print("e2_tmp",e2_tmp)
    print("ec_tmp",ec_tmp)
    print("sum e",e1_tmp+e2_tmp+ec_tmp)
    return [e1_tmp,e2_tmp,ec_tmp]



def get_frag_energy(mo_coeffs, nsocc, nfsites, efac, TA, h1, hf_veff, rdm1, rdm2s, dname, 
                    eri_file='eri_file.h5',eri_files=None):
    rdm1s_rot = mo_coeffs @ rdm1 @ mo_coeffs.T # for unrestricted, removing factor * 0.5

    hf_1rdm = numpy.dot(mo_coeffs[:,:nsocc],
                       mo_coeffs[:,:nsocc].conj().T)

    delta_rdm1 = 2 * (rdm1s_rot - hf_1rdm)

    veff0 = functools.reduce(numpy.dot,(TA.T,hf_veff,TA))

    e1 = numpy.einsum("ij,ij->i",h1[:nfsites], delta_rdm1[:nfsites])
    ec = numpy.einsum("ij,ij->i",veff0[:nfsites], delta_rdm1[:nfsites])

    if TA.ndim == 3:
        jmax = TA[0].shape[1]
    else:
        jmax = TA.shape[1]


    if eri_files:
        r = h5py.File(eri_files[dname],'r')
        eri = r[dname][()]
    else:
        r = h5py.File(eri_file,'r')
        eri = r[dname][()]
    r.close()

    rdm2s = numpy.einsum("ijkl,pi,qj,rk,sl->pqrs", 0.5*rdm2s,
                         *([mo_coeffs]*4),optimize=True)

    e2 = numpy.zeros_like(e1)
    for i in range(nfsites):
        for j in range(jmax):
            ij = i*(i+1)//2+j if i > j else j*(j+1)//2+i
            Gij = rdm2s[i,j,:jmax,:jmax].copy()            
            Gij[numpy.diag_indices(jmax)] *= 0.5
            Gij += Gij.T            
            e2[i] += Gij[numpy.tril_indices(jmax)] @ eri[ij]

    e_ = e1+e2+ec        
    etmp = 0.
    e1_tmp = 0.
    e2_tmp = 0.
    ec_tmp = 0.

    for i in efac[1]:
        etmp += efac[0]*e_[i]
        e1_tmp += efac[0]*e1[i] 
        e2_tmp += efac[0]*e2[i] 
        ec_tmp += efac[0]*ec[i] 

    print("e1_tmp",e1_tmp)
    print("e2_tmp",e2_tmp)
    print("ec_tmp",ec_tmp)
    print("sum e",e1_tmp+e2_tmp+ec_tmp)
    return [e1_tmp,e2_tmp,ec_tmp]
