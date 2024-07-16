"""
Bootstrap Embedding Calculation with an
Unrestricted Hartree-Fock Bath

Reference
  Tran, H.; Ye, H.; Van Voorhis, T.
  J. Chem. Phys. 153, 214101 (2020)

TODO
  Add oneshot and optimize later
"""

import numpy
from .lo import iao_tmp
from .pbe import pbe
from .pfrag import Frags
import h5py, os, time, pbe_var, sys

class ube(pbe):
    def __init__(
        self,
        mf,
        fobj,
        eri_file="eri_file.h5",
        exxdiv="ewald",
        lo_method="lowdin",
        compute_hf=True,
        nkpt=None,
        kpoint=False,
        super_cell=False,
        molecule=False,
        kpts=None,
        cell=None,
        kmesh=None,
        restart=False,
        save=False,
        restart_file="storepbe.pk",
        mo_energy=None,
        iao_wannier=True,
        save_file="storepbe.pk",
        hci_pt=False,
        hci_cutoff=0.001,
        ci_coeff_cutoff=None,
        select_cutoff=None,
        debug00=False,
        debug001=False,
    ):
        self.unrestricted = True

        self.self_match = fobj.self_match
        self.frag_type = fobj.frag_type
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

        self.ebe_hf = 0.0
        self.ebe_tot = 0.0
        self.super_cell = super_cell

        self.kpoint = kpoint
        self.kpts = None
        self.cell = cell
        self.kmesh = kmesh
        self.molecule = fobj.molecule

        # HCI parameters
        self.hci_cutoff = hci_cutoff
        self.ci_coeff_cutoff = ci_coeff_cutoff
        self.select_cutoff = select_cutoff
        self.hci_pt = hci_pt

        self.mo_energy = mf.mo_energy

        self.mf = mf
        self.Nocc = [mf.mol.nelec[0],mf.mol.nelec[1]]
        self.enuc = mf.energy_nuc()

        self.hcore = mf.get_hcore()
        self.S = mf.get_ovlp()
        #print("mf.mo_coeff", mf.mo_coeff)
        self.C = [numpy.array(mf.mo_coeff[0]),numpy.array(mf.mo_coeff[1])]
        self.hf_dm = [mf.make_rdm1()[0],mf.make_rdm1()[1]]
        self.hf_veff = [mf.get_veff()[0],mf.get_veff()[1]]

        self.hf_etot = mf.e_tot
        self.W = None
        self.lmo_coeff = None
        self.cinv = None

        self.print_ini()

        self.Fobjs_a = []
        self.Fobjs_b = []

        self.pot = initialize_pot(self.Nfrag, self.edge_idx)

        self.eri_file = eri_file
        self.ek = 0.0
        self.frozen_core = False if not fobj.frozen_core else True
        self.ncore = 0
        self.E_core = 0
        self.C_core = None
        self.P_core = None
        self.core_veff = None

        self.uhf_full_e = mf.e_tot

        if self.frozen_core:
            self.ncore = fobj.ncore
            print("ncore", self.ncore)
            self.no_core_idx = fobj.no_core_idx
            print("no_core_idx", self.no_core_idx)
            self.core_list = fobj.core_list
            print("core_list", self.core_list)
            print("full Nocc with core", self.Nocc)
            self.Nocc[0] -= self.ncore
            self.Nocc[1] -= self.ncore
            print("Nocc without core", self.Nocc)
            self.hf_dm = [numpy.dot(self.C[s][:,self.ncore:self.ncore+self.Nocc[s]],
                                        self.C[s][:,self.ncore:self.ncore+self.Nocc[s]].T) for s in [0,1]]
            self.C_core = [self.C[s][:, :self.ncore] for s in [0,1]]
            self.P_core = [numpy.dot(self.C_core[s], self.C_core[s].T) for s in [0,1]]
            self.core_veff = 1.*mf.get_veff(dm = self.P_core)

            self.E_core = sum([numpy.einsum("ji,ji->", 2*self.hcore+self.core_veff[s], self.P_core[s]) for s in [0,1]]) * 0.5

        # iao ignored for now
        self.C_a = numpy.array(mf.mo_coeff[0])
        self.C_b = numpy.array(mf.mo_coeff[1])
        del self.C

        self.localize(
            lo_method,
            mol=self.cell,
            valence_basis=fobj.valence_basis,
            valence_only=fobj.valence_only,
            iao_wannier=False,
        )

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

        """
        if pbe_var.CREATE_SCRATCH_DIR:
            try:
                jobid = str(os.environ['SLURM_JOB_ID'])
            except:
                jobid = ''
        if not pbe_var.SCRATCH=='':
            self.scratch_dir = pbe_var.SCRATCH+str(jobid)
            os.system('mkdir '+self.scratch_dir)
        if jobid == '':
            self.eri_file = pbe_var.SCRATCH+eri_file
        else:
            self.eri_file = pbe_var.SCRATCH+str(jobid)+'/'+eri_file
        """
        self.initialize(mf._eri, compute_hf)

    def initialize(self, eri_, compute_hf):
        import h5py
        from pyscf import ao2mo

        if compute_hf:
            E_hf = 0.0
        EH1 = 0.0
        ECOUL = 0.0
        EF = 0.0

        file_eri = h5py.File(self.eri_file, "w")
        lentmp = len(self.edge_idx)

        # alpha
        for I in range(self.Nfrag):
            if lentmp:
                fobjs_a = Frags(
                    self.fsites[I],
                    I,
                    edge=self.edge[I],
                    eri_file=self.eri_file,
                    center=self.center[I],
                    edge_idx=self.edge_idx[I],
                    center_idx=self.center_idx[I],
                    efac=self.ebe_weight[I],
                    centerf_idx=self.centerf_idx[I],
                    unitcell=self.unitcell,
                    unitcell_nkpt=self.unitcell_nkpt,
                    unrestricted=True
                )
            else:
                fobjs_a = Frags(
                    self.fsites[I],
                    I,
                    edge=[],
                    center=[],
                    eri_file=self.eri_file,
                    edge_idx=[],
                    center_idx=[],
                    centerf_idx=[],
                    efac=self.ebe_weight[I],
                    unitcell=self.unitcell,
                    unitcell_nkpt=self.unitcell_nkpt,
                    unrestricted=True
                )
            self.Fobjs_a.append(fobjs_a)
        # beta
        for I in range(self.Nfrag):
            if lentmp:
                fobjs_b = Frags(
                    self.fsites[I],
                    I,
                    edge=self.edge[I],
                    eri_file=self.eri_file,
                    center=self.center[I],
                    edge_idx=self.edge_idx[I],
                    center_idx=self.center_idx[I],
                    efac=self.ebe_weight[I],
                    centerf_idx=self.centerf_idx[I],
                    unitcell=self.unitcell,
                    unitcell_nkpt=self.unitcell_nkpt,
                    unrestricted=True
                )
            else:
                fobjs_b = Frags(
                    self.fsites[I],
                    I,
                    edge=[],
                    center=[],
                    eri_file=self.eri_file,
                    edge_idx=[],
                    center_idx=[],
                    centerf_idx=[],
                    efac=self.ebe_weight[I],
                    unitcell=self.unitcell,
                    unitcell_nkpt=self.unitcell_nkpt,
                    unrestricted=True
                )
            self.Fobjs_b.append(fobjs_b)

        orb_count_a = []
        orb_count_b = []

        all_noccs = []

        for I in range(self.Nfrag):
            fobj_a = self.Fobjs_a[I]
            fobj_b = self.Fobjs_b[I]

            if self.frozen_core:
                fobj_a.core_veff = self.core_veff[0]
                fobj_b.core_veff = self.core_veff[1]
                #print("fobj_a.core_veff", fobj_a.core_veff.shape, fobj_a.core_veff)
                #print("fobj_b.core_veff", fobj_a.core_veff.shape, fobj_b.core_veff)
                #print("self.W[0]", self.W[0].shape, self.W[0])
                #print("self.W[1]", self.W[1].shape, self.W[1])
                #print("self.lmo_coeff_a", self.lmo_coeff_a.shape, self.lmo_coeff_a)
                #print("self.lmo_coeff_b", self.lmo_coeff_b.shape, self.lmo_coeff_b)
                orb_count_a.append(fobj_a.sd(self.W[0], self.lmo_coeff_a, self.Nocc[0], frag_type=self.frag_type, return_orb_count=True))
                orb_count_b.append(fobj_b.sd(self.W[1], self.lmo_coeff_b, self.Nocc[1], frag_type=self.frag_type, return_orb_count=True))
            else:
                fobj_a.core_veff = None
                fobj_b.core_veff = None
                orb_count_a.append(fobj_a.sd(self.W, self.lmo_coeff_a, self.Nocc[0], frag_type=self.frag_type, return_orb_count=True))
                orb_count_b.append(fobj_b.sd(self.W, self.lmo_coeff_b, self.Nocc[1], frag_type=self.frag_type, return_orb_count=True))

            all_noccs.append(self.Nocc)

            a_TA = fobj_a.TA.shape
            b_TA = fobj_b.TA.shape
            
            """ 
            # Adjust different alpha and beta space sizes: not necessary yet without optimization steps
            if fobj_a.TA.shape[1] > fobj_b.TA.shape[1]: #different alpha and beta space size
                fobj_b.sd(self.W[0], self.lmo_coeff_b, self.Nocc[1], frag_type=self.frag_type, norb=fobj_a.TA.shape[1])
            elif fobj_b.TA.shape[1] > fobj_a.TA.shape[1]:
                fobj_a.sd(self.W[1], self.lmo_coeff_a, self.Nocc[0], frag_type=self.frag_type, norb=fobj_b.TA.shape[1])
            """

            if eri_ is None and not self.mf.with_df is None:
                # NOT IMPLEMENTED
                eri_a = ao2mo.kernel(
                    self.mf.mol, fobj_a.TA, compact=True)
                # for density-fitted integrals; if mf is provided, pyscf.ao2mo uses DF object in an outcore fashion
                eri_b = ao2mo.kernel(
                    self.mf.mol, fobj_b.TA, compact=True)
            else:
                eri_a = ao2mo.incore.full(
                    eri_, fobj_a.TA, compact=True)  # otherwise, do an incore ao2mo
                eri_b = ao2mo.incore.full(
                    eri_, fobj_b.TA, compact=True)

                Csd_A = fobj_a.TA # may have to add in nibath here
                Csd_B = fobj_b.TA

                eri_ab = ao2mo.incore.general(eri_, (Csd_A,Csd_A,Csd_B,Csd_B), compact=True)
                # Save cross-eri term here 

            #fobj_a.dname = fobj_b.dname: the object is (eri_a, eri_b, eri_ab)
            file_eri.create_dataset(fobj_a.dname[0], data=eri_a)
            file_eri.create_dataset(fobj_a.dname[1], data=eri_b)
            file_eri.create_dataset(fobj_a.dname[2], data=eri_ab)

            sab = self.C_a @ self.S @ self.C_b
            #sab = new_mo_coeff[0] @ fobj_a._mf.get_ovlp() @ new_mo_coeff[1]
            #print("self.C_a", self.C_a)
            dm_init = fobj_a.get_nsocc(self.S, self.C_a, self.Nocc[0], ncore=self.ncore, Sab=sab)

            fobj_a.cons_h1(self.hcore)
            eri_a = ao2mo.restore(8, eri_a, fobj_a.nao)
            fobj_a.cons_fock(self.hf_veff[0], self.S, self.hf_dm[0] * 2.0, eri_=eri_a)

            fobj_a.hf_veff = self.hf_veff[0]
            fobj_a.heff = numpy.zeros_like(fobj_a.h1)
            fobj_a.scf(fs=True, eri=eri_a)

            fobj_a.dm0 = numpy.dot(
                fobj_a._mo_coeffs[:, : fobj_a.nsocc], fobj_a._mo_coeffs[:, : fobj_a.nsocc].conj().T
            )

            if compute_hf:
                eh1_a, ecoul_a, ef_a = fobj_a.energy_hf(return_e1=True, unrestricted=True, spin_ind=0)
                EH1 += eh1_a
                ECOUL += ecoul_a
                E_hf += fobj_a.ebe_hf

            #print("self.C_b", self.C_b)
            dm_init = fobj_b.get_nsocc(self.S, self.C_b, self.Nocc[1], ncore=self.ncore, Sab=sab)

            fobj_b.cons_h1(self.hcore)
            eri_b = ao2mo.restore(8, eri_b, fobj_b.nao)
            fobj_b.cons_fock(self.hf_veff[1], self.S, self.hf_dm[1] * 2.0, eri_=eri_b)
            fobj_b.hf_veff = self.hf_veff[1]
            fobj_b.heff = numpy.zeros_like(fobj_b.h1)
            fobj_b.scf(fs=True, eri=eri_b)

            fobj_b.dm0 = numpy.dot(
                fobj_b._mo_coeffs[:, : fobj_b.nsocc], fobj_b._mo_coeffs[:, : fobj_b.nsocc].conj().T
            )

            if compute_hf:
                eh1_b, ecoul_b, ef_b = fobj_b.energy_hf(return_e1=True, unrestricted=True, spin_ind=1)
                EH1 += eh1_b
                ECOUL += ecoul_b
                E_hf += fobj_b.ebe_hf
        file_eri.close()

        print("Number of Orbitals per Fragment:", flush=True)
        print("____________________________________________________________________", flush=True)
        print("| Fragment |    Nocc   | Fragment Orbs | Bath Orbs | Schmidt Space |", flush=True)
        print("____________________________________________________________________", flush=True)
        for I in range(self.Nfrag):
            print('|    {:>2}    | ({:>3},{:>3}) |   ({:>3},{:>3})   | ({:>3},{:>3}) |   ({:>3},{:>3})   |'.format(I, all_noccs[I][0],all_noccs[I][1], 
                                                                                    orb_count_a[I][0], orb_count_b[I][0],
                                                                                    orb_count_a[I][1], orb_count_b[I][1],
                                                                                    orb_count_a[I][0]+orb_count_a[I][1],
                                                                                    orb_count_b[I][0]+orb_count_b[I][1]),
                                                                                    flush=True)
        print("____________________________________________________________________", flush=True)
        if compute_hf:
            E_hf /= self.unitcell_nkpt
            hf_err = self.hf_etot - (E_hf + self.enuc + self.E_core)

            self.ebe_hf = E_hf + self.enuc + self.E_core - self.ek
            print("HF-in-HF error                 :  {:>.4e} Ha".format(hf_err), flush=True)
            if abs(hf_err) > 1.0e-5:
                print("WARNING!!! Large HF-in-HF energy error")
                print("eh1 ", EH1)
                print("ecoul ", ECOUL)

            print(flush=True)

        couti = 0
        for fobj in self.Fobjs_a:
            fobj.udim = couti
            couti = fobj.set_udim(couti)

        couti = 0
        for fobj in self.Fobjs_b:
            fobj.udim = couti
            couti = fobj.set_udim(couti)

    def oneshot(self, solver="MP2", nproc=1, ompnum=4, calc_frag_energy=False, clean_eri=False):
        # TODO
        # Not ready; Can be used once fobj.energy gets cumulant expression
        # and unrestricted keyword multiplies the RDMs appropriately (see: energy_hf)
        from .solver import be_func, be_func_u
        from .be_parallel import be_func_parallel_u
        #return NotImplementedError
        print("started oneshot", calc_frag_energy)
        if nproc == 1:
            E, E_comp  = be_func_u(None,
                        zip(self.Fobjs_a, self.Fobjs_b), #self.Fobjs_a, self.Fobjs_b),
                        self.Nocc,
                        solver,
                        self.enuc,
                        hf_veff=self.hf_veff,
                        nproc=ompnum,
                        eeval=True,
                        ereturn=True,
                        relax_density=False,
                        frag_energy=calc_frag_energy,
                        frozen=self.frozen_core)
        else:
            E, E_comp = be_func_parallel_u(None,
                        zip(self.Fobjs_a, self.Fobjs_b), #self.Fobjs_a, self.Fobjs_b),
                        self.Nocc,
                        solver,
                        self.enuc,
                        hf_veff=self.hf_veff,
                        eeval=True,
                        ereturn=True,
                        relax_density=False,
                        frag_energy=calc_frag_energy,
                        frozen=self.frozen_core,
                        nproc=nproc, 
                        ompnum=ompnum)

        print("-----------------------------------------------------", flush=True)
        print("             One Shot BE ", flush=True)
        print("             Solver : ", solver, flush=True)
        print("-----------------------------------------------------", flush=True)
        print(flush=True)

        self.ebe_tot = E + self.uhf_full_e
        print("Total Energy : {:>12.8f} Ha".format((self.ebe_tot), flush=True))
        print("Corr  Energy : {:>12.8f} Ha".format((E), flush=True))
        
        if clean_eri == True:
            try:
                os.remove(self.eri_file)
                os.rmdir(self.scratch_dir)
            except:
                print("Scratch directory not removed")

def initialize_pot(Nfrag, edge_idx):
    pot_ = []

    if not len(edge_idx) == 0:
        for I in range(Nfrag):
            for i in edge_idx[I]:
                for j in range(len(i)):
                    for k in range(len(i)):
                        if j > k:
                            continue
                        pot_.append(0.0)

    pot_.append(0.0)  # alpha
    pot_.append(0.0)  # beta
    return pot_
