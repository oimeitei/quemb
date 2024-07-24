from .pfrag import Frags
from .helper import get_core
import numpy,functools,sys, pickle
from pyscf import lib
import h5py,os,time,pbe_var

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
                 kpts = None, cell=None,
                 kmesh=None, cderi=None,
                 restart=False, save=False,
                 restart_file='storepbe.pk',
                 mo_energy = None, iao_wannier=True,
                 save_file='storepbe.pk',hci_pt=False,
                 nproc=1, ompnum=4,
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
        
        self.unrestricted = False

        self.nproc = nproc
        self.ompnum = ompnum
        
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
        self.ebe_tot = 0.
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
        self.cderi = cderi
        self.ek=0.

        # set scratch dir in pbe_var
        jobid=''
        if pbe_var.CREATE_SCRATCH_DIR:
            try:
                jobid = str(os.environ['SLURM_JOB_ID'])
            except:
                jobid = ''
        if not pbe_var.SCRATCH=='': 
            self.scratch_dir = pbe_var.SCRATCH+str(jobid)
            os.system('mkdir '+self.scratch_dir)
        else:
            self.scratch_dir = None
        if jobid == '':
            self.eri_file = pbe_var.SCRATCH+eri_file
        else:
            self.eri_file = self.scratch_dir+'/'+eri_file
            
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
        time_pre_fock = time.time()
        self.FOCK = self.mf.get_fock(self.hcore, self.S, self.hf_veff, self.hf_dm)
        time_post_fock = time.time()
        print("Time to get full-system Fock matrix: ", time_post_fock - time_pre_fock)
        if not restart or debug00:
            self.localize(lo_method, mol=self.cell, valence_basis=fobj.valence_basis, valence_only=fobj.valence_only, iao_wannier=iao_wannier)
            if fobj.valence_only and lo_method=='iao':
                self.Ciao_pao = self.localize(lo_method, mol=self.cell, valence_basis=fobj.valence_basis,
                                              hstack=True,
                                              valence_only=False, nosave=True)
            time_post_lo = time.time()
            print("Time to localize:" , time_post_lo - time_post_fock)
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
            time_pre_hfinit = time.time()
            self.initialize(mf._eri,compute_hf)
            time_post_hfinit = time.time()
            print("Time to initialize HF: ",time_post_hfinit - time_pre_hfinit)
            
        elif debug00:
            self.initialize(eri00,compute_hf)
        else:            
            self.initialize(None,compute_hf, restart=True)
        
        
    from ._opt import optimize
    from .optqn import get_be_error_jacobian,get_be_error_jacobian_selffrag
    from .lo import localize
    from .rdm import rdm1_fullbasis, compute_energy_full
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
        print('            BEn = ',self.be_type,flush=True)
        print('-----------------------------------------------------------',
                  flush=True)
        print(flush=True)
        

    def initialize(self, eri_,compute_hf, restart=False):
        from .helper import get_scfObj        
        import h5py
        from pyscf import ao2mo
        from multiprocessing import Pool
        
        if compute_hf: E_hf = 0.
        
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
                
            self.Fobjs.append(fobjs_)
                
        if not restart:
            # ERI Transform Decision Tree
            # Do we have full (ij|kl)?
            #   Yes -- ao2mo, incore version
            #   No  -- Do we have (ij|P) from density fitting?
            #            Yes -- ao2mo, outcore version, using saved (ij|P)
            assert (not eri_ is None) or (hasattr(self.mf, 'with_df')), "Input mean-field object is missing ERI (mf._eri) or DF (mf.with_df) object. You may want to ensure that incore_anyway was set for non-DF calculations."
            if not eri_ is None: # incore ao2mo using saved eri from mean-field calculation
                for I in range(self.Nfrag):
                    eri = ao2mo.incore.full(eri_, self.Fobjs[I].TA, compact=True)
                    file_eri.create_dataset(self.Fobjs[I].dname, data=eri)
            elif hasattr(self.mf, 'with_df') and not self.mf.with_df is None:
                # pyscf.ao2mo uses DF object in an outcore fashion using (ij|P) in pyscf temp directory
                for I in range(self.Nfrag):
                    eri = self.mf.with_df.ao2mo(self.Fobjs[I].TA, compact=True)
                    file_eri.create_dataset(self.Fobjs[I].dname, data=eri)
        else:
            eri=None
        
        for fobjs_ in self.Fobjs:
            eri = numpy.array(file_eri.get(fobjs_.dname))
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

        if not restart:
            file_eri.close()
        
        if compute_hf:
                        
            self.ebe_hf = E_hf+self.enuc+self.E_core
            hf_err = self.hf_etot - self.ebe_hf
            print('HF-in-HF error                 :  {:>.4e} Ha'.
                  format(hf_err), flush=True)
            if abs(hf_err)>1.e-5:
                print('WARNING!!! Large HF-in-HF energy error')
                       
            print(flush=True)
            
        couti = 0
        for fobj in self.Fobjs:
            fobj.udim = couti
            couti = fobj.set_udim(couti)
                        
    def oneshot(self, solver='MP2', nproc=1, ompnum=4, calc_frag_energy=False, clean_eri=False):
        from .solver import be_func
        from .be_parallel import be_func_parallel

        print("Calculating Energy by Fragment? ", calc_frag_energy)
        if nproc == 1:
            rets  = be_func(None, self.Fobjs, self.Nocc, solver, self.enuc, hf_veff=self.hf_veff,
                        hci_cutoff=self.hci_cutoff,
                        ci_coeff_cutoff = self.ci_coeff_cutoff,
                        select_cutoff = self.select_cutoff,
                        nproc=ompnum, frag_energy=calc_frag_energy,
                        ereturn=True, eeval=True)
        else:
            rets  = be_func_parallel(None, self.Fobjs, self.Nocc, solver, self.enuc, hf_veff=self.hf_veff,
                                 hci_cutoff=self.hci_cutoff,
                                 ci_coeff_cutoff = self.ci_coeff_cutoff,
                                 select_cutoff = self.select_cutoff,
                                 ereturn=True, eeval=True, frag_energy=calc_frag_energy,
                                 nproc=nproc, ompnum=ompnum)

        print('-----------------------------------------------------',
                  flush=True)
        print('             One Shot BE ', flush=True)
        print('             Solver : ',solver,flush=True)
        print('-----------------------------------------------------',
                  flush=True)
        print(flush=True)
        if calc_frag_energy:
            print("Final Tr(F del g) is         : {:>12.8f} Ha".format(rets[1][0]+rets[1][2]), flush=True)
            print("Final Tr(V K_approx) is      : {:>12.8f} Ha".format(rets[1][1]), flush=True)
            print("Final e_corr is              : {:>12.8f} Ha".format(rets[0]), flush=True)

            self.ebe_tot = rets[0]

        if not calc_frag_energy:
            self.compute_energy_full(approx_cumulant=True, return_rdm=False)

        if clean_eri == True:
            try:
                os.remove(self.eri_file)
                os.rmdir(self.scratch_dir)
            except:
                print("Scratch directory not removed")

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

def eritransform_parallel_mol(eri_, mcoeff):
    eri_t = ao2mo.incore.full(eri_, mcoeff, compact=True)
    return eri_t

def eritransform_parallel_mol_cd(atom, basis, C_ao_emb, cderi):
    from pyscf import gto, df
    
    mol_ = gto.M(atom=atom, basis=basis)
    mydf = df.DF(mol_)
    mydf._cderi = cderi
    eri = mydf.ao2mo(C_ao_emb, compact=True)
    return eri

def parallel_fock_wrapper(dname, nao, dm, S, TA, hf_veff, eri_file):
    from .helper import get_veff, get_eri

    eri_ = get_eri(dname, nao, eri_file=eri_file, ignore_symm=True)
    veff0, veff_ = get_veff(eri_, dm, S, TA, hf_veff, return_veff0 = True)

    return veff0, veff_


def parallel_scf_wrapper(dname, nao, nocc, h1,  dm_init, eri_file):
    from .helper import get_eri, get_scfObj
    eri = get_eri(dname, nao, eri_file=eri_file)
    mf_ = get_scfObj(h1, eri, nocc, dm_init)
    
    return mf_.mo_coeff
