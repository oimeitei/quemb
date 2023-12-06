from .solver import schmidt_decomposition, schmidt_decomp_rdm1, schmidt_decomp_svd, schmidt_decomp_rdm1_new
from .helper import *
import numpy,h5py
import functools,sys, math
from pyscf import ao2mo
from functools import reduce


class Frags:
    """
    Class for fragments.
    """
    
    def __init__(self, fsites, ifrag, edge=None, center=None,
                 edge_idx=None, center_idx=None, efac=None,
                 eri_file='eri_file.h5',unitcell_nkpt=1,
                 ewald_ek=None, centerf_idx=None, unitcell=1):
        """Constructor function for `Frags` class

        Parameters
        ----------
        fsites : list
            list of AOs in the fragment (i.e. pbe.fsites[i] or fragpart.fsites[i])
        ifrag : int
            fragment index (∈ [0, pbe.Nfrag])
        edge : list, optional
            list of lists of edge site AOs for each atom in the fragment, by default None
        center : list, optional
            list of fragment indices where edge site AOs are center site, by default None
        edge_idx : list, optional
            list of lists of indices for edge site AOs within the fragment, by default None
        center_idx : list, optional
            list of lists of indices within the fragment specified in `center` that points to the edge site AOs , by default None
        efac : list, optional
            weight used for energy contributions, by default None
        eri_file : str, optional
            two-electron integrals stored as h5py file, by default 'eri_file.h5'
        unitcell_nkpt : int, optional
            number of k-points in the unit cell; 1 for molecular calculations, by default 1
        centerf_idx : list, optional
            indices of the center site atoms in the fragment, by default None
        unitcell : int, optional
            number of unitcells used in building the fragment, by default 1
        """

        
        self.fsites = fsites
        self.unitcell=unitcell
        self.unitcell_nkpt=unitcell_nkpt
        self.nfsites = len(fsites)
        self.TA = None
        self.TA_lo_eo = None
        self.h1 = None
        self.ifrag = ifrag
        self.dname = 'f'+str(ifrag)
        self.nao = None
        self.mo_coeffs = None 
        self._mo_coeffs = None 
        self.nsocc = None
        self._mf = None
        self._mc = None
        
        # CCSD
        self.t1 = None
        self.t2 = None
        
        self.heff = None
        self.edge = edge
        self.center = center 
        self.edge_idx = edge_idx
        self.center_idx = center_idx
        self.centerf_idx = centerf_idx
        self.udim = None

        self._rdm1 = None
        self.__rdm1 = None
        self.__rdm2 = None
        self.rdm1 = None
        self.genvs = None
        self.ebe = 0.
        self.ebe_hf = 0.
        self.efac = efac
        self.ewald_ek=ewald_ek
        self.fock = None
        self.veff = None
        self.dm0 = None
        self.eri_file = eri_file

    def sd(self, lao, lmo, nocc, nkpt = None, s=None, mo_coeff= None,
           frag_type='autogen', cinv = '',
           cell=None, kpts = None, kmesh=None, rdm=None, mo_energy=None, h1=None):
        from pyscf import lib
        from pyscf.pbc.tools.k2gamma import mo_k2gamma
        from scipy import fft

        if lao.ndim==2:
            TA = schmidt_decomposition(lmo, nocc, self.fsites)
            self.C_lo_eo = TA
            TA = numpy.dot(lao,TA)

            
            self.nao = TA.shape[1]
            self.TA = TA
        else:
            print(' Only molecular BE is supported! ')
            sys.exit()

    def cons_h1(self, h1, stmp = None, wtmp = None):

        if h1.ndim == 2:            
            h1_tmp = functools.reduce(numpy.dot,
                                      (self.TA.T, h1, self.TA))
            
            self.h1 = h1_tmp
        
    def cons_fock(self, hf_veff, S, dm, eri_=None):

        if eri_ is None:
            eri_ = get_eri(self.dname, self.TA.shape[1], ignore_symm=True, eri_file=self.eri_file)
                
        veff_ = get_veff(eri_, dm, S, self.TA, hf_veff)
        self.veff = veff_.real        
        self.fock = self.h1 + veff_.real
        

    def get_nsocc(self, S, C, nocc,ncore=0, tdebug=None, wdebug=None, hf_dm = None):
        
        import scipy.linalg
        if self.TA.ndim ==2:
            C_ = functools.reduce(numpy.dot,(self.TA.T, S, C[:,ncore:ncore+nocc]))
            P_ = numpy.dot(C_, C_.T)
            nsocc_ = numpy.trace(P_)
            nsocc = int(numpy.round(nsocc_))
            
            try:
                mo_coeffs = scipy.linalg.svd(C_)[0]
            except:
                
                mo_coeffs = scipy.linalg.eigh(C_)[1][:,-nsocc:]
            
            self._mo_coeffs = mo_coeffs
            self.nsocc = nsocc
            
            return P_
            
            
    def scf(self, heff=None, fs=False, eri=None, dm0 = None):
        import copy
        if self._mf is not None: self._mf = None
        if self._mc is not None: self._mc = None
        if heff is None: heff = self.heff

        if eri is None:
            eri = get_eri(self.dname, self.nao, eri_file=self.eri_file)

        if dm0 is None:
            dm0 = numpy.dot( self._mo_coeffs[:,:self.nsocc],
                             self._mo_coeffs[:,:self.nsocc].conj().T) *2.
        
        mf_ = get_scfObj(self.fock + heff, eri, self.nsocc, dm0 = dm0)
        if not fs:
            self._mf = mf_
            self.mo_coeffs = mf_.mo_coeff.copy()
        else:
            self._mo_coeffs = mf_.mo_coeff.copy()
        mf_= None
        
    def update_heff(self,u, cout = None, return_heff=False, only_chem=False):
        
        heff_ = numpy.zeros_like(self.h1)

        if cout is None:
            cout = self.udim
        else:
            cout = cout
        
        for i,fi in enumerate(self.fsites):            
            if not any(i in sublist for sublist in self.edge_idx):    
                heff_[i,i] -= u[-1]

        if only_chem:
            self.heff = heff_
            if return_heff:
                if cout is None:
                    return heff_
                else:
                    return(cout, heff_)
            return cout

        for i in self.edge_idx:
            for j in range(len(i)):
                for k in range(len(i)):
                    if j>k :#or j==k:
                        continue                    
                    
                    heff_[i[j], i[k]] = u[cout]
                    heff_[i[k], i[j]] = u[cout]
                    
                    cout += 1
    
        
        self.heff = heff_
        if return_heff:
            if cout is None:
                return heff_
            else:
                return(cout, heff_)
        return cout

    def set_udim(self, cout):
        for i in self.edge_idx:
            for j in range(len(i)):
                for k in range(len(i)):
                    if j>k :
                        continue
                    cout += 1
        return cout
        

    def energy(self,rdm2s, eri=None, print_fragE=False):

        ## This function uses old energy expression and will be removed
        rdm2s = numpy.einsum("ijkl,pi,qj,rk,sl->pqrs", 0.5*rdm2s,
                             *([self.mo_coeffs]*4),optimize=True)        
        
        e1 = 2.*numpy.einsum("ij,ij->i",self.h1[:self.nfsites], self._rdm1[:self.nfsites])
        ec = numpy.einsum("ij,ij->i",self.veff[:self.nfsites],self._rdm1[:self.nfsites])
        

        if self.TA.ndim == 3:
            jmax = self.TA[0].shape[1]
        else:
            jmax = self.TA.shape[1]
            
        if eri is None:
            r = h5py.File(self.eri_file,'r')
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
        e1_ = 0.
        ec_ = 0.
        e2_ = 0.
        for i in self.efac[1]:
            etmp += self.efac[0]*e_[i]
            e1_ += self.efac[0] * e1[i]
            ec_ += self.efac[0] * ec[i]
            e2_ += self.efac[0] * e2[i]
            
        print('BE Energy Frag-{:>3}   {:>12.7f}  {:>12.7f}  {:>12.7f};   Total : {:>12.7f}'.
              format(self.dname, e1_, ec_, e2_, etmp))
        
        self.ebe = etmp
        return (e1+e2+ec)

    def energy_hf(self, rdm_hf=None, mo_coeffs = None, eri=None, return_e1=False):
        if mo_coeffs is None:
            mo_coeffs = self._mo_coeffs

        if rdm_hf is None:
            rdm_hf = numpy.dot(mo_coeffs[:,:self.nsocc],
                               mo_coeffs[:,:self.nsocc].conj().T)
        
        
        e1 = numpy.einsum('ij,ji->', self.h1, rdm_hf)
        
        
        e1 = 2.*numpy.einsum("ij,ij->i", self.h1[:self.nfsites],
                             rdm_hf[:self.nfsites])

        ec = numpy.einsum("ij,ij->i",self.veff[:self.nfsites],
                          rdm_hf[:self.nfsites])

        if self.TA.ndim == 3:
            jmax = self.TA[0].shape[1]
        else:
            jmax = self.TA.shape[1]
        if eri is None:
            r = h5py.File(self.eri_file,'r')
            eri = r[self.dname][()]

            r.close()

        
        e2 = numpy.zeros_like(e1)
        for i in range(self.nfsites):
            for j in range(jmax):
                ij = i*(i+1)//2+j if i > j else j*(j+1)//2+i
                Gij =  (2.*rdm_hf[i,j]*rdm_hf -
                        numpy.outer(rdm_hf[i], rdm_hf[j]))[:jmax,:jmax] 
                Gij[numpy.diag_indices(jmax)] *= 0.5
                Gij += Gij.T                
                e2[i] += Gij[numpy.tril_indices(jmax)] @ eri[ij]

        e_ = e1+e2+ec        
        etmp = 0.
        for i in self.efac[1]:
            etmp += self.efac[0]*e_[i]
                
        self.ebe_hf = etmp
        
        if return_e1:
            e_h1 = 0.
            e_coul = 0.
            for i in self.efac[1]:
                
                e_h1 += self.efac[0]*e1[i]
                e_coul += self.efac[0]*(e2[i]+ec[i])
            return(e_h1,e_coul, e1+e2+ec)
        
        return e1+e2+ec
    
