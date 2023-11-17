from .pfrag import Frags
from .helper import get_core
import numpy,functools,sys, pickle
from pyscf import lib
import h5py,os

from .lo import iao_tmp

class storePBE:
    def __init__(self, Nocc, hf_veff, hcore,
                 S, C, hf_dm, hf_etot, W, lmo_coeff,
                 enuc, ek,
                 E_core, C_core, P_core, core_veff, mo_energy):
        self.Nocc = Nocc
        self.hf_veff = hf_veff
        self.hcore = hcore
        self.S = S
        self.C = C
        self.hf_dm = hf_dm
        self.hf_etot = hf_etot
        self.W = W
        self.lmo_coeff = lmo_coeff
        self.enuc = enuc
        self.ek = ek
        self.E_core = E_core
        self.C_core = C_core
        self.P_core = P_core
        self.core_veff = core_veff
        self.mo_energy = mo_energy

        


class pbe:


    def __init__(self, mf, fobj, eri_file='eri_file.h5', exxdiv='ewald',
                 lo_method='lowdin',compute_hf=True, nkpt = None, kpoint = False,
                 super_cell=False, molecule=False,
                 kpts = None, cell=None, kmesh=None,
                 restart=False, save=False,
                 restart_file='storepbe.pk',
                 mo_energy = None, iao_wannier=True,
                 save_file='storepbe.pk',hci_pt=False,
                 hci_cutoff=0.001, ci_coeff_cutoff = None, select_cutoff=None,
                 debug00=False, debug001=False):
        """Constructor for pbe object

        Parameters
        ----------
        mf : pyscf.scf.SCF
          PySCF HF
        fobj : pbe.fragpart
          Fragment object containing sites, center, edges and indices
        lo_method: str
          Method for orbital localization. Supports 'lowdin', 'boys', and 'wannier'. Defaults to 'lowdin'
        save : bool
          Save intermediate objects for a restart. 
        restart : bool
          Restart. If set True, HF need not be repeated.
        """

        
        if restart:
            with open(restart_file, 'rb') as rfile:
                store_ = pickle.load(rfile)
                rfile.close()
            self.Nocc = store_.Nocc
            self.hf_veff = store_.hf_veff
            self.hcore = store_.hcore
            self.S = store_.S
            self.C = store_.C
            self.hf_dm = store_.hf_dm
            self.hf_etot = store_.hf_etot
            self.W = store_.W
            self.lmo_coeff = store_.lmo_coeff
            self.enuc = store_.enuc
            self.ek = store_.ek
            self.E_core = store_.E_core
            self.C_core = store_.C_core
            self.P_core = store_.P_core
            self.core_veff = store_.core_veff
            self.mo_energy = store_.mo_energy
        
        self.self_match = fobj.self_match
        self.frag_type=fobj.frag_type
        self.Nfrag = fobj.Nfrag 
        self.fsites = fobj.fsites
        self.edge = fobj.edge
        self.center = fobj.center
        self.edge_idx = fobj.edge_idx
        self.center_idx = fobj.center_idx
        self.centerf_idx = fobj.centerf_idx
        self.ebe_weight = fobj.ebe_weight
        self.be_type = fobj.be_type
        self.unitcell = fobj.unitcell
        self.mol = fobj.mol

        unitcell_nkpt = 1
        self.unitcell_nkpt = unitcell_nkpt
                    
        self.ebe_hf = 0.
        self.super_cell = super_cell
        
        self.kpoint = kpoint         
        self.kpts = kpts
        self.cell = cell
        self.kmesh = kmesh
        self.molecule = fobj.molecule

        # HCI parameters
        self.hci_cutoff = hci_cutoff
        self.ci_coeff_cutoff = ci_coeff_cutoff
        self.select_cutoff = select_cutoff
        self.hci_pt=hci_pt
       
        self.mf = mf # tmp
        if not restart:   
            self.mo_energy = mf.mo_energy
            
            self.mf = mf
            self.Nocc = mf.mol.nelectron//2 
            self.enuc = mf.energy_nuc()
            
            self.hcore = mf.get_hcore()
            self.S = mf.get_ovlp()
            self.C = numpy.array(mf.mo_coeff)            
            self.hf_dm = mf.make_rdm1()
            self.hf_veff = mf.get_veff()
            self.hf_etot = mf.e_tot
            self.W = None
            self.lmo_coeff = None
            self.cinv = None

        self.print_ini()
        self.Fobjs = []
        self.pot = initialize_pot(self.Nfrag, self.edge_idx)
        self.eri_file = eri_file
        self.ek=0.

        self.frozen_core = False if not fobj.frozen_core else True
        self.ncore = 0
        if not restart:
            self.E_core = 0
            self.C_core = None
            self.P_core = None
            self.core_veff = None
        
        if self.frozen_core:
            self.ncore = fobj.ncore
            self.no_core_idx = fobj.no_core_idx
            self.core_list = fobj.core_list
            
            if not restart:
                self.Nocc -=self.ncore                                
                self.hf_dm = 2.*numpy.dot(self.C[:,self.ncore:self.ncore+self.Nocc],
                                          self.C[:,self.ncore:self.ncore+self.Nocc].T)
                self.C_core = self.C[:,:self.ncore]
                self.P_core = numpy.dot(self.C_core, self.C_core.T)
                self.core_veff = mf.get_veff(dm = self.P_core*2.)
                self.E_core = numpy.einsum('ji,ji->',2.*self.hcore+self.core_veff, self.P_core)                
                self.hf_veff -= self.core_veff
                self.hcore += self.core_veff
        # fock
        self.FOCK = self.mf.get_fock(self.hcore, self.S, self.hf_veff, self.hf_dm)
        if not restart or debug00:
            self.localize(lo_method, mol=self.cell, valence_basis=fobj.valence_basis, valence_only=fobj.valence_only, iao_wannier=iao_wannier)
            if fobj.valence_only and lo_method=='iao':
                self.Ciao_pao = self.localize(lo_method, mol=self.cell, valence_basis=fobj.valence_basis,
                                              hstack=True,
                                              valence_only=False, nosave=True)
        if save:
            store_ = storePBE(self.Nocc, self.hf_veff, self.hcore,
                              self.S, self.C, self.hf_dm, self.hf_etot,
                              self.W, self.lmo_coeff, self.enuc, self.ek,
                              self.E_core, self.C_core, self.P_core, self.core_veff, self.mo_energy)

            with open(save_file, 'wb') as rfile:
                pickle.dump(store_, rfile, pickle.HIGHEST_PROTOCOL)
            rfile.close()
            
        if debug001:
            file_eri = h5py.File('eri_fullk3.h5','w')
            file_eri.create_dataset('erifull', data=mf._eri, dtype=numpy.complex128)
            file_eri.close()
            sys.exit()

        if debug00:
            
            r = h5py.File('eri_fullk3.h5','r')
            eri00 = r.get('erifull')
            r.close()
            self.mf=mf
           
        if not restart :            
            self.initialize(mf._eri,compute_hf)
            
        elif debug00:
            self.initialize(eri00,compute_hf)
        else:            
            self.initialize(None,compute_hf, restart=True)
        
        
    from ._opt import optimize
    from .optqn import get_be_error_jacobian,get_be_error_jacobian_selffrag
    from .lo import localize
    from .rdm import rdm1_fullbasis, get_rdm
    def print_ini(self):
        
        print('-----------------------------------------------------------',
                  flush=True)

        print('  MMM     MMM    OOOO    LL           BBBBBBB    EEEEEEE ',flush=True)
        print('  M MM   MM M   OO  OO   LL           BB     B   EE      ',flush=True)
        print('  M  MM MM  M  OO    OO  LL           BB     B   EE      ',flush=True)
        print('  M   MMM   M  OO    OO  LL     ===   BBBBBBB    EEEEEEE ',flush=True)
        print('  M         M  OO    OO  LL           BB     B   EE      ',flush=True)
        print('  M         M   OO  OO   LL           BB     B   EE      ',flush=True)
        print('  M         M    OOOO    LLLLLL       BBBBBBB    EEEEEEE ',flush=True)
                
        print(flush=True)
        print('            MOLECULAR BOOTSTRAP EMBEDDING',flush=True)            
        print('           BEn = ',self.be_type,flush=True)
        print('-----------------------------------------------------------',
                  flush=True)
        print(flush=True)
        

    def initialize(self, eri_,compute_hf, restart=False):
        from .helper import get_scfObj        
        import h5py
        from pyscf import ao2mo
            
        if compute_hf: E_hf = 0.
        EH1 = 0.
        ECOUL = 0.
        EF = 0.
        
        # from here remove ncore from C
        if not restart:
            file_eri = h5py.File(self.eri_file,'w')
        lentmp = len(self.edge_idx)
        for I in range(self.Nfrag):
            
            if lentmp:
                fobjs_ = Frags(self.fsites[I], I, edge=self.edge[I],
                               eri_file=self.eri_file,
                               center=self.center[I], edge_idx=self.edge_idx[I],
                               center_idx=self.center_idx[I],efac=self.ebe_weight[I],
                               centerf_idx=self.centerf_idx[I], unitcell=self.unitcell,
                               unitcell_nkpt=self.unitcell_nkpt)
            else:
                fobjs_ = Frags(self.fsites[I],I,edge=[],center=[],
                               eri_file=self.eri_file,
                               edge_idx=[],center_idx=[],centerf_idx=[],
                               efac=self.ebe_weight[I], unitcell=self.unitcell,
                               unitcell_nkpt=self.unitcell_nkpt)
            fobjs_.sd(self.W, self.lmo_coeff, self.Nocc,
                      frag_type=self.frag_type)
                
            if not restart:
                eri = ao2mo.incore.full(eri_, fobjs_.TA, compact=True)                    
                if fobjs_.dname in eri:
                    del(file_eri[fobjs_.dname])
                
                file_eri.create_dataset(fobjs_.dname, data=eri)
            else:
                eri=None
                
            dm_init = fobjs_.get_nsocc(self.S, self.C, self.Nocc, ncore=self.ncore)
            
            fobjs_.cons_h1(self.hcore)
                       
            if not restart:
                eri = ao2mo.restore(8, eri, fobjs_.nao)
            
            fobjs_.cons_fock(self.hf_veff, self.S, self.hf_dm, eri_=eri)
                
            fobjs_.heff = numpy.zeros_like(fobjs_.h1)
            fobjs_.scf(fs=True, eri=eri)
            
            fobjs_.dm0 = numpy.dot( fobjs_._mo_coeffs[:,:fobjs_.nsocc],
                                    fobjs_._mo_coeffs[:,:fobjs_.nsocc].conj().T) *2.
                
            if compute_hf:
            
                eh1, ecoul, ef = fobjs_.energy_hf(return_e1=True)
                EH1 += eh1
                ECOUL += ecoul
                E_hf += fobjs_.ebe_hf

            self.Fobjs.append(fobjs_)
        if not restart:
            file_eri.close()
        
        if compute_hf:
            
            E_hf /= self.unitcell_nkpt                
            hf_err = self.hf_etot-(E_hf+self.enuc+self.E_core)
            
            self.ebe_hf = E_hf+self.enuc+self.E_core-self.ek
            print('HF-in-HF error                 :  {:>.4e} Ha'.
                  format(hf_err), flush=True)
            if abs(hf_err)>1.e-5:
                print('WARNING!!! Large HF-in-HF energy error')
                       
            print(flush=True)
            
        couti = 0
        for fobj in self.Fobjs:
            fobj.udim = couti
            couti = fobj.set_udim(couti)
                        
    def oneshot(self, solver='MP2',nproc=1, ompnum=4):
        from .solver import be_func
        from .be_parallel import be_func_parallel


        if nproc == 1:
            E = be_func(None, self.Fobjs, self.Nocc, solver, self.enuc,
                        ek = self.ek, kp=self.nkpt,
                        hci_cutoff=self.hci_cutoff,
                        ci_coeff_cutoff = self.ci_coeff_cutoff,
                        select_cutoff = self.select_cutoff,
                        nproc=ompnum,
                        ereturn=True, eeval=True)
        else:
            E = be_func_parallel(None, self.Fobjs, self.Nocc, solver, self.enuc,
                                 ek = self.ek, kp=self.nkpt,
                                 hci_cutoff=self.hci_cutoff,
                                 ci_coeff_cutoff = self.ci_coeff_cutoff,
                                 select_cutoff = self.select_cutoff,
                                 ereturn=True, eeval=True,
                                 nproc=nproc, ompnum=ompnum)

        print(flush=True)
        print('-----------------------------------------------------',
                  flush=True)
        print('             One Shot BE ', flush=True)
        print('             Solver : ',solver,flush=True)
        print('-----------------------------------------------------',
                  flush=True)
        print(flush=True)

        print('BE energy per unit cell        : {:>12.8f} Ha'.format(E+self.E_core),
              flush=True)
        print('BE Ecorr  per unit cell        : {:>12.8f} Ha'.format(E+self.E_core-self.ebe_hf),
              flush=True)
        print(flush=True)
        print('-----------------------------------------------------',
                  flush=True)

    def update_fock(self, heff=None):

        if heff is None:
            for fobj in self.Fobjs:
                fobj.fock += fobj.heff
        else:
            for idx, fobj in self.Fobjs:
                fobj.fock += heff[idx]

    def write_heff(self, heff_file='bepotfile.h5'):
        filepot = h5py.File(heff_file, 'w')
        for fobj in self.Fobjs:
            print(fobj.heff.shape, fobj.dname, flush=True)
            filepot.create_dataset(fobj.dname, data=fobj.heff)
        filepot.close()

    def read_heff(self, heff_file='bepotfile.h5'):
        filepot = h5py.File(heff_file, 'r')
        for fobj in self.Fobjs:
            fobj.heff = filepot.get(fobj.dname)
        filepot.close()
        
        
        
def initialize_pot(Nfrag, edge_idx):
    pot_=[]
    
    if not len(edge_idx) == 0:
        for I in range(Nfrag):
            for i in edge_idx[I]:
                for j in range(len(i)):
                    for k in range(len(i)):
                        if j>k:
                            continue
                        pot_.append(0.)
    
    pot_.append(0.)
    return pot_
