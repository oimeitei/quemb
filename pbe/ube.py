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
        self.Nocc_a = mf.mol.nelec[0]
        self.Nocc_b = mf.mol.nelec[1]
        self.enuc = mf.energy_nuc()

        self.hcore = mf.get_hcore()
        self.S = mf.get_ovlp()
        self.C_a = numpy.array(mf.mo_coeff[0])
        self.C_b = numpy.array(mf.mo_coeff[1])
        self.hf_dm_a = mf.make_rdm1()[0]
        self.hf_dm_b = mf.make_rdm1()[1]
        self.hf_veff_a = mf.get_veff()[0]
        self.hf_veff_b = mf.get_veff()[1]
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

        # placeholder for frzcore
        # if self.frozen_core:
        #     self.ncore = fobj.ncore
        #     self.no_core_idx = fobj.no_core_idx
        #     self.core_list = fobj.core_list
        #     self.Nocc -=self.ncore
        #     self.hf_dm = 2.*numpy.dot(self.C[:,self.ncore:self.ncore+self.Nocc],
        #                                 self.C[:,self.ncore:self.ncore+self.Nocc].T)
        #     self.C_core = self.C[:,:self.ncore]
        #     self.P_core = numpy.dot(self.C_core, self.C_core.T)
        #     self.core_veff = mf.get_veff(dm = self.P_core*2.)
        #     self.E_core = numpy.einsum('ji,ji->',2.*self.hcore+self.core_veff, self.P_core)
        #     self.hf_veff -= self.core_veff
        #     self.hcore += self.core_veff

        # iao ignored for now
        self.C = self.C_a
        self.localize(
            lo_method,
            mol=self.cell,
            valence_basis=fobj.valence_basis,
            valence_only=fobj.valence_only,
            iao_wannier=False,
        )
        self.lmo_coeff_a = self.lmo_coeff
        if not self.frozen_core:
            self.lmo_coeff_b = self.W.T @ self.S @ self.C_b
        else:
            self.lmo_coeff_b = self.W.T @ self.S @ self.C_b[:, self.ncore :]
        del self.C
        del self.lmo_coeff

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
                fobjs_ = Frags(
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
                )
            else:
                fobjs_ = Frags(
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
                )
            fobjs_.dname += "_a"
            fobjs_.sd(self.W, self.lmo_coeff_a, self.Nocc_a, frag_type=self.frag_type)

            if eri_ is None and not self.mf.with_df is None:
                eri = ao2mo.kernel(
                    self.mf.mol, fobjs_.TA, compact=True
                )  # for density-fitted integrals; if mf is provided, pyscf.ao2mo uses DF object in an outcore fashion
            else:
                eri = ao2mo.incore.full(
                    eri_, fobjs_.TA, compact=True
                )  # otherwise, do an incore ao2mo

            file_eri.create_dataset(fobjs_.dname, data=eri)
            dm_init = fobjs_.get_nsocc(self.S, self.C_a, self.Nocc_a, ncore=self.ncore)
            fobjs_.cons_h1(self.hcore)
            eri = ao2mo.restore(8, eri, fobjs_.nao)
            fobjs_.cons_fock(self.hf_veff_a, self.S, self.hf_dm_a * 2.0, eri_=eri)
            fobjs_.heff = numpy.zeros_like(fobjs_.h1)
            fobjs_.scf(fs=True, eri=eri)
            fobjs_.dm0 = numpy.dot(
                fobjs_._mo_coeffs[:, : fobjs_.nsocc], fobjs_._mo_coeffs[:, : fobjs_.nsocc].conj().T
            )

            if compute_hf:
                eh1, ecoul, ef = fobjs_.energy_hf(return_e1=True, unrestricted=True)
                EH1 += eh1
                ECOUL += ecoul
                E_hf += fobjs_.ebe_hf

            self.Fobjs_a.append(fobjs_)
        # beta
        for I in range(self.Nfrag):
            if lentmp:
                fobjs_ = Frags(
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
                )
            else:
                fobjs_ = Frags(
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
                )
            fobjs_.dname += "_b"
            fobjs_.sd(self.W, self.lmo_coeff_b, self.Nocc_b, frag_type=self.frag_type)

            if eri_ is None and not self.mf.with_df is None:
                eri = ao2mo.kernel(
                    self.mf.mol, fobjs_.TA, compact=True
                )  # for density-fitted integrals; if mf is provided, pyscf.ao2mo uses DF object in an outcore fashion
            else:
                eri = ao2mo.incore.full(
                    eri_, fobjs_.TA, compact=True
                )  # otherwise, do an incore ao2mo

            file_eri.create_dataset(fobjs_.dname, data=eri)
            dm_init = fobjs_.get_nsocc(self.S, self.C_b, self.Nocc_b, ncore=self.ncore)
            fobjs_.cons_h1(self.hcore)
            eri = ao2mo.restore(8, eri, fobjs_.nao)
            fobjs_.cons_fock(self.hf_veff_b, self.S, self.hf_dm_b * 2.0, eri_=eri)
            fobjs_.heff = numpy.zeros_like(fobjs_.h1)
            fobjs_.scf(fs=True, eri=eri)
            fobjs_.dm0 = numpy.dot(
                fobjs_._mo_coeffs[:, : fobjs_.nsocc], fobjs_._mo_coeffs[:, : fobjs_.nsocc].conj().T
            )

            if compute_hf:
                eh1, ecoul, ef = fobjs_.energy_hf(return_e1=True, unrestricted=True)
                EH1 += eh1
                ECOUL += ecoul
                E_hf += fobjs_.ebe_hf

            self.Fobjs_b.append(fobjs_)
        file_eri.close()

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

    def oneshot(self, solver="MP2", nproc=1, ompnum=4):
        # TODO
        # Not ready; Can be used once fobj.energy gets cumulant expression
        # and unrestricted keyword multiplies the RDMs appropriately (see: energy_hf)
        from .solver import be_func
        from .be_parallel import be_func_parallel
        return NotImplementedError
        if nproc == 1:
            E_a = be_func(
                None,
                self.Fobjs_a,
                self.Nocc_a,
                solver,
                self.enuc,
                hci_cutoff=self.hci_cutoff,
                ci_coeff_cutoff=self.ci_coeff_cutoff,
                select_cutoff=self.select_cutoff,
                nproc=ompnum,
                ereturn=True,
                eeval=True,
            )
            E_b = be_func(
                None,
                self.Fobjs_b,
                self.Nocc_b,
                solver,
                self.enuc,
                hci_cutoff=self.hci_cutoff,
                ci_coeff_cutoff=self.ci_coeff_cutoff,
                select_cutoff=self.select_cutoff,
                nproc=ompnum,
                ereturn=True,
                eeval=True,
            )
        else:
            E_a = be_func_parallel(
                None,
                self.Fobjs_a,
                self.Nocc_a,
                solver,
                self.enuc,
                hci_cutoff=self.hci_cutoff,
                ci_coeff_cutoff=self.ci_coeff_cutoff,
                select_cutoff=self.select_cutoff,
                ereturn=True,
                eeval=True,
                nproc=nproc,
                ompnum=ompnum,
            )
            E_b = be_func_parallel(
                None,
                self.Fobjs_b,
                self.Nocc_b,
                solver,
                self.enuc,
                hci_cutoff=self.hci_cutoff,
                ci_coeff_cutoff=self.ci_coeff_cutoff,
                select_cutoff=self.select_cutoff,
                ereturn=True,
                eeval=True,
                nproc=nproc,
                ompnum=ompnum,
            )

        print("-----------------------------------------------------", flush=True)
        print("             One Shot BE ", flush=True)
        print("             Solver : ", solver, flush=True)
        print("-----------------------------------------------------", flush=True)
        print(flush=True)

        print("Total Energy : {:>12.8f} Ha".format((E_a + E_b) / 2.0), flush=True)
        print("Corr  Energy : {:>12.8f} Ha".format((E_a + E_b) / 2.0 - self.ebe_hf), flush=True)


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
