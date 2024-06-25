"""
This is adapted from frankenstein (private) from Hongzhou Ye, Henry Tran, and Leah Weisburn
and pyscf: https://github.com/pyscf/pyscf , cc/uccsd.py

The original pyscf code has the following license:
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Timothy Berkelbach <tim.berkelbach@gmail.com>
#         Qiming Sun <osirpt.sun@gmail.com>
"""

from functools import reduce
import numpy

from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.cc import ccsd
from pyscf.ao2mo import _ao2mo
from pyscf.mp import ump2
from pyscf import scf
from pyscf import __config__

def make_eris_incore(mycc, Vss, Vos, mo_coeff=None, ao2mofn=None, frozen=False):
    eris = _ChemistsERIs()
    eris._common_init_(mycc, Vss, Vos, mo_coeff, frozen)
    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    nvira, nvirb = nmoa-nocca, nmob-noccb
    moa = eris.mo_coeff[0]
    mob = eris.mo_coeff[1]
    nmoa = moa.shape[1]
    nmob = mob.shape[1]
    if callable(ao2mofn):
        eri_aa = ao2mofn(moa).reshape([nmoa]*4)
        eri_bb = ao2mofn(mob).reshape([nmob]*4)
        eri_ab = ao2mofn((moa,moa,mob,mob))
    else:
        eri_aa = ao2mo.restore(1, ao2mo.full(mycc._scf._eri, moa), nmoa)
        eri_bb = ao2mo.restore(1, ao2mo.full(mycc._scf._eri, mob), nmob)
        eri_ab = ao2mo.general(mycc._scf._eri, (moa,moa,mob,mob), compact=False)
    eri_ba = eri_ab.reshape(nmoa,nmoa,nmob,nmob).transpose(2,3,0,1)

    eri_aa = eri_aa.reshape(nmoa,nmoa,nmoa,nmoa)
    eri_ab = eri_ab.reshape(nmoa,nmoa,nmob,nmob)
    eri_ba = eri_ba.reshape(nmob,nmob,nmoa,nmoa)
    eri_bb = eri_bb.reshape(nmob,nmob,nmob,nmob)
    eris.oooo = eri_aa[:nocca,:nocca,:nocca,:nocca].copy()
    eris.ovoo = eri_aa[:nocca,nocca:,:nocca,:nocca].copy()
    eris.ovov = eri_aa[:nocca,nocca:,:nocca,nocca:].copy()
    eris.oovv = eri_aa[:nocca,:nocca,nocca:,nocca:].copy()
    eris.ovvo = eri_aa[:nocca,nocca:,nocca:,:nocca].copy()
    eris.ovvv = eri_aa[:nocca,nocca:,nocca:,nocca:].copy()
    eris.vvvv = eri_aa[nocca:,nocca:,nocca:,nocca:].copy()

    eris.OOOO = eri_bb[:noccb,:noccb,:noccb,:noccb].copy()
    eris.OVOO = eri_bb[:noccb,noccb:,:noccb,:noccb].copy()
    eris.OVOV = eri_bb[:noccb,noccb:,:noccb,noccb:].copy()
    eris.OOVV = eri_bb[:noccb,:noccb,noccb:,noccb:].copy()
    eris.OVVO = eri_bb[:noccb,noccb:,noccb:,:noccb].copy()
    eris.OVVV = eri_bb[:noccb,noccb:,noccb:,noccb:].copy()
    eris.VVVV = eri_bb[noccb:,noccb:,noccb:,noccb:].copy()

    eris.ooOO = eri_ab[:nocca,:nocca,:noccb,:noccb].copy()
    eris.ovOO = eri_ab[:nocca,nocca:,:noccb,:noccb].copy()
    eris.ovOV = eri_ab[:nocca,nocca:,:noccb,noccb:].copy()
    eris.ooVV = eri_ab[:nocca,:nocca,noccb:,noccb:].copy()
    eris.ovVO = eri_ab[:nocca,nocca:,noccb:,:noccb].copy()
    eris.ovVV = eri_ab[:nocca,nocca:,noccb:,noccb:].copy()
    eris.vvVV = eri_ab[nocca:,nocca:,noccb:,noccb:].copy()

    #eris.OOoo = eri_ba[:noccb,:noccb,:nocca,:nocca].copy()
    eris.OVoo = eri_ba[:noccb,noccb:,:nocca,:nocca].copy()
    #eris.OVov = eri_ba[:noccb,noccb:,:nocca,nocca:].copy()
    eris.OOvv = eri_ba[:noccb,:noccb,nocca:,nocca:].copy()
    eris.OVvo = eri_ba[:noccb,noccb:,nocca:,:nocca].copy()
    eris.OVvv = eri_ba[:noccb,noccb:,nocca:,nocca:].copy()
    #eris.VVvv = eri_ba[noccb:,noccb:,nocca:,nocca:].copy()

    if not callable(ao2mofn):
        ovvv = eris.ovvv.reshape(nocca*nvira,nvira,nvira)
        eris.ovvv = lib.pack_tril(ovvv).reshape(nocca,nvira,nvira*(nvira+1)//2)
        eris.vvvv = ao2mo.restore(4, eris.vvvv, nvira)

        OVVV = eris.OVVV.reshape(noccb*nvirb,nvirb,nvirb)
        eris.OVVV = lib.pack_tril(OVVV).reshape(noccb,nvirb,nvirb*(nvirb+1)//2)
        eris.VVVV = ao2mo.restore(4, eris.VVVV, nvirb)

        ovVV = eris.ovVV.reshape(nocca*nvira,nvirb,nvirb)
        eris.ovVV = lib.pack_tril(ovVV).reshape(nocca,nvira,nvirb*(nvirb+1)//2)
        vvVV = eris.vvVV.reshape(nvira**2,nvirb**2)
        idxa = numpy.tril_indices(nvira)
        idxb = numpy.tril_indices(nvirb)
        eris.vvVV = lib.take_2d(vvVV, idxa[0]*nvira+idxa[1], idxb[0]*nvirb+idxb[1])

        OVvv = eris.OVvv.reshape(noccb*nvirb,nvira,nvira)
        eris.OVvv = lib.pack_tril(OVvv).reshape(noccb,nvirb,nvira*(nvira+1)//2)
    #print("Final made eris", eris)
    return eris

def _get_ovvv_base(ovvv, *slices):
    if ovvv.ndim == 3:
        ovw = np.asarray(ovvv[slices])
        nocc, nvir, nvir_pair = ovw.shape
        ovvv = lib.unpack_tril(ovw.reshape(nocc*nvir,nvir_pair))
        nvir1 = ovvv.shape[2]
        return ovvv.reshape(nocc,nvir,nvir1,nvir1)
    elif slices:
        return ovvv[slices]
    else:
        return ovvv

class _ChemistsERIs(ccsd._ChemistsERIs):
    def __init__(self, mol=None):
        ccsd._ChemistsERIs.__init__(self, mol)
        self.OOOO = None
        self.OVOO = None
        self.OVOV = None
        self.OOVV = None
        self.OVVO = None
        self.OVVV = None
        self.VVVV = None

        self.ooOO = None
        self.ovOO = None
        self.ovOV = None
        self.ooVV = None
        self.ovVO = None
        self.ovVV = None
        self.vvVV = None

        self.OVoo = None
        self.OOvv = None
        self.OVvo = None

    def _common_init_(self, mycc, Vss, Vos, mo_coeff=None, frozen=False):
        if mo_coeff is None:
            mo_coeff = mycc.mo_coeff
        mo_idx = mycc.get_frozen_mask()
        self.mo_coeff = mo_coeff = \
                (mo_coeff[0][:,mo_idx[0]], mo_coeff[1][:,mo_idx[1]])
        # Note: Recomputed fock matrix since SCF may not be fully converged.
        dm = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
        vhf = self.frank_get_veff(mycc, dm, Vss, Vos)
        fockao = self.frank_get_fock(mycc, vhf, frozen)
        self.focka = reduce(numpy.dot, (mo_coeff[0].conj().T, fockao[0], mo_coeff[0]))
        self.fockb = reduce(numpy.dot, (mo_coeff[1].conj().T, fockao[1], mo_coeff[1]))
        self.fock = (self.focka, self.fockb)
        #this energy is wrong
        self.e_hf = mycc._scf.energy_tot(dm=dm, h1e=[mycc._scf.h1[s]+mycc._scf.gcores_raw[s]-vhf[s] for s in [0,1]],  vhf=vhf)
        nocca, noccb = self.nocc = mycc.nocc
        self.mol = mycc.mol

        mo_ea = self.focka.diagonal().real
        mo_eb = self.fockb.diagonal().real
        self.mo_energy = (mo_ea, mo_eb)
        gap_a = abs(mo_ea[:nocca,None] - mo_ea[None,nocca:])
        gap_b = abs(mo_eb[:noccb,None] - mo_eb[None,noccb:])
        if gap_a.size > 0:
            gap_a = gap_a.min()
        else:
            gap_a = 1e9
        if gap_b.size > 0:
            gap_b = gap_b.min()
        else:
            gap_b = 1e9
        if gap_a < 1e-5 or gap_b < 1e-5:
            logger.warn(mycc, 'HOMO-LUMO gap (%s,%s) too small for UCCSD',
                        gap_a, gap_b)
        return self

    def get_ovvv(self, *slices):
        return _get_ovvv_base(self.ovvv, *slices)

    def get_ovVV(self, *slices):
        return _get_ovvv_base(self.ovVV, *slices)

    def get_OVvv(self, *slices):
        return _get_ovvv_base(self.OVvv, *slices)

    def get_OVVV(self, *slices):
        return _get_ovvv_base(self.OVVV, *slices)

    def _contract_VVVV_t2(self, mycc, t2, vvvv_or_direct=False, out=None, verbose=None):
        if isinstance(vvvv_or_direct, numpy.ndarray):
            vvvv = vvvv_or_direct
        elif vvvv_or_direct:
            vvvv = None
        else:
            vvvv = self.VVVV
        return ccsd._contract_vvvv_t2(mycc, self.mol, vvvv, t2, out, verbose)

    def _contract_vvVV_t2(self, mycc, t2, vvvv_or_direct=False, out=None, verbose=None):
        if isinstance(vvvv_or_direct, numpy.ndarray):
            vvvv = vvvv_or_direct
        elif vvvv_or_direct:
            vvvv = None
        else:
            vvvv = self.vvVV
        return ccsd._contract_vvvv_t2(mycc, self.mol, vvvv, t2, out, verbose)

    def frank_get_veff(self, mycc, dm, Vss, Vos):
        veffss = [numpy.einsum("pqrs,sr->pq", Vss[s], dm[s]) -
            numpy.einsum("psrq,sr->pq", Vss[s], dm[s]) for s in [0,1]]
        veffos = [numpy.einsum("pqrs,sr->pq", Vos, dm[1]),
            numpy.einsum("pqrs,qp->rs", Vos, dm[0])]
        veff = [veffss[s] + veffos[s] for s in [0,1]]

        return veff

    def frank_get_fock(self, mycc, vhf, frozen):
        if frozen==False:
            mycc._scf.full_gcore = None
            mycc._scf.full_hs = None
            fock = [mycc._scf.h1[s]+mycc._scf.gcores_raw[s] for s in [0,1]]
        else:
            mycc._scf.full_gcore = [mycc._scf.gcores_raw[s] - vhf[s] for s in [0,1]]
            mycc._scf.full_hs = [mycc._scf.h1[s] + mycc._scf.full_gcore[s] + mycc._scf.core_veffs[s] for s in [0,1]]
            fock = [mycc._scf.full_hs[s] + vhf[s] for s in [0,1]]
        return fock
