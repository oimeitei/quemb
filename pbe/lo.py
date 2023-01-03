'''
Authors: Henry Tran
         Oinam Meitei

func get_symm_mat_pow(), get_aoind_by_atom(), reorder_by_atom_()
copied from frankestein

'''

from pyscf import lib
import numpy,sys
from copy import deepcopy
from functools import reduce
# iao tmp


def iao_tmp(cell, C, nocc, S1, kpts, minbas='minao'):
    
    mango =0

def dot_gen(A, B, ovlp):
    return A.T @ B if ovlp is None else A.T @ ovlp @ B

def get_cano_orth_mat(A, thr=1.E-6, ovlp=None):
    S = dot_gen(A,A,ovlp)
    e, u = numpy.linalg.eigh(S)
    if thr > 0:
        idx_keep = e/e[-1] > thr
    else:
        idx_keep = list(range(e.shape[0]))
    U = u[:,idx_keep] * e[idx_keep]**-0.5

    return U

def cano_orth(A, thr=1.E-6, ovlp=None):
    """ Canonically orthogonalize columns of A
    """
    U = get_cano_orth_mat(A, thr, ovlp)

    return A @ U

def get_symm_orth_mat(A, thr=1.E-6, ovlp=None):
    S = dot_gen(A,A,ovlp)
    e, u = numpy.linalg.eigh(S)
    if int(numpy.sum(e < thr)) > 0:
        raise ValueError("Linear dependence is detected in the column space of A: smallest eigenvalue (%.3E) is less than thr (%.3E). Please use 'cano_orth' instead." % (numpy.min(e), thr))
    U = u @ numpy.diag(e**-0.5) @ u.T

    return U

def symm_orth(A, thr=1.E-6, ovlp=None):
    """ Symmetrically orthogonalize columns of A
    """
    U = get_symm_orth_mat(A, thr, ovlp)

    return A @ U


def remove_core_mo(Clo, Ccore, S, thr=0.5):
    assert(numpy.allclose(Clo.T@S@Clo,numpy.eye(Clo.shape[1])))
    assert(numpy.allclose(Ccore.T@S@Ccore,numpy.eye(Ccore.shape[1])))
    
    n,nlo = Clo.shape
    ncore = Ccore.shape[1]
    Pcore = Ccore@Ccore.T @ S
    Clo1 = (numpy.eye(n) - Pcore) @ Clo              
    pop = numpy.diag(Clo1.T @ S @ Clo1)
    idx_keep = numpy.where(pop>thr)[0]
    assert(len(idx_keep) == nlo-ncore)    
    Clo2 = symm_orth(Clo1[:,idx_keep], ovlp=S)

    return Clo2

 
def reorder_lo(C, S, idao_by_atom, atom_by_motif, motifname,
    ncore_by_motif, thresh=0.5, verbose=3):
    """ 
    TODO
    This function reorders the IAOs and PAOs so that the IAOs
    and PAOs for each atom are grouped together.
    """
    pop_by_ao = (Xinv @ C)**2
    reorder_idx_by_atom = []
    for idao in idao_by_atom:
        pop = numpy.sum(pop_by_ao[idao], axis = 0)

 
def get_xovlp(mol, basis='sto-3g'):
    """
    Gets set of valence orbitals based on smaller (should be minimal) basis
    inumpy.t:
        mol - pyscf mol object, just need it for the working basis
        basis - the IAO basis, Knizia recommended 'minao'
    returns:
        S12 - Overlap of two basis sets
        S22 - Overlap in new basis set
    """
    from pyscf.gto import intor_cross
    mol_alt = mol.copy()
    mol_alt.basis = basis
    mol_alt.build()

    S12 = intor_cross('int1e_ovlp', mol, mol_alt)
    S22 = mol_alt.intor('int1e_ovlp')

    return S12, S22

def get_iao(Co, S12, S1, S2 = None):
    """
    Args:
        Co: occupied coefficient matrix with core
        p: valence AO matrix in AO
        no: number of occ orbitals
        S12: ovlp between working basis and valence basis
             can be thought of as working basis in valence basis
        S1: ao ovlp matrix
        S2: valence AO ovlp
    """
    # define projection operators
    n = Co.shape[0]
    if S2 is None:
        S2 = S12.T @ numpy.linalg.inv(S1) @ S12
    P1 = numpy.linalg.inv(S1)
    P2 = numpy.linalg.inv(S2)
    
    # depolarized occ mo
    Cotil = P1 @ S12 @ P2 @ S12.T @ Co

    # repolarized valence AOs
    ptil = P1 @ S12
    Stil = Cotil.T @ S1 @ Cotil

    Po = Co @ Co.T
    Potil = Cotil @ numpy.linalg.inv(Stil) @ Cotil.T

    Ciao = (numpy.eye(n) - (Po + Potil - 2 * Po @ S1 @ Potil) @ S1) @ ptil
    Ciao = symm_orth(Ciao, ovlp = S1)
    
    # check span 
    rep_err = numpy.linalg.norm(Ciao @ Ciao.T @ S1 @ Po - Po)
    if rep_err > 1.E-10:
        raise RuntimeError
    return Ciao

def get_pao(Ciao, S, S12, S2, mol):
    """
    Args:
        Ciao: output of :func:`get_iao`
        S: ao ovlp matrix
        S12: valence orbitals projected into ao basis
        S2: valence ovlp matrix
        mol: pyscf mol instance
    Return:
        Cpao (orthogonalized)
    """
    n = Ciao.shape[0]
    s12 = numpy.linalg.inv(S) @ S12
    nonval = numpy.eye(n) - s12 @ s12.T # set of orbitals minus valence (orth in working basis)

    
    Piao = Ciao @ Ciao.T @ S # projector into IAOs
    Cpao = (numpy.eye(n) - Piao) @ nonval # project out IAOs from non-valence basis

    
    # begin canonical orthogonalization to get rid of redundant orbitals
    numpy.o0 = Cpao.shape[1]
    Cpao = cano_orth(Cpao, ovlp=S)
    numpy.o1 = Cpao.shape[1]
    
    return Cpao

def get_pao_native(Ciao, S, mol, valence_basis):
    """
    Args:
        Ciao: output of :func:`get_iao_native`
        S: ao ovlp matrix
        mol: mol object
        valence basis: basis used for valence orbitals
    Return:
        Cpao (symmetrically orthogonalized)
    """
    n = Ciao.shape[0]
    
    # Form a mol object with the valence basis for the ao_labels
    mol_alt = mol.copy()
    mol_alt.basis = valence_basis
    mol_alt.build()

    full_ao_labels = mol.ao_labels()
    valence_ao_labels = mol_alt.ao_labels()

    vir_idx = [idx for idx, label in enumerate(full_ao_labels) if (not label in valence_ao_labels)]

    Piao = Ciao @ Ciao.T @ S
    Cpao = (numpy.eye(n) - Piao)[:, vir_idx]

    try:
        Cpao = symm_orth(Cpao, ovlp=S)
    except:
        print("Symm orth PAO failed. Switch to cano orth", flush=True)
        npao0 = Cpao.shape[1]
        Cpao = cano_orth(Cpao, ovlp=S)
        npao1 = Cpao.shape[1]
        print("# of PAO: %d --> %d" % (npao0,npao1), flush=True)
        print("", flush=True)

    return Cpao

def get_loc(mol, C, method):
    if method.upper() == 'ER':
        from pyscf.lo import ER as Localizer
    elif method.upper() == 'PM':
        from pyscf.lo import PM as Localizer
    elif method.upper() == 'FB' or method.upper() == 'BOYS':
        from pyscf.lo import Boys as Localizer
    else:
        raise NotImplementedError('Localization scheme not understood')

    mlo = Localizer(mol, C)
    mlo.init_guess = None 
    C_ = mlo.kernel()

    return C_


def get_symm_mat_pow(A, p, check_symm=True, thresh=1.E-8):
    """A ** p where A is symmetric

    Note:
        For integer p, it calls numpy.linalg.matrix_power
    """
    if abs(int(p) - p) < thresh:
        return numpy.linalg.matrix_power(A, int(p))

    if check_symm:
        assert(numpy.linalg.norm(A-A.conj().T) < thresh)

    e, u = numpy.linalg.eigh(A)
    Ap = u @ numpy.diag(e**p) @ u.conj().T

    return Ap


def get_aoind_by_atom(mol, atomind_by_motif=None):
    import numpy as np
    natom = mol.natm
    aoslice_by_atom = mol.aoslice_by_atom()
    aoshift_by_atom = [0]+[aoslice_by_atom[ia][-1]
        for ia in range(natom)]
    # if motif info is provided, group lo by motif
    if atomind_by_motif is None:
    
        aoind_by_atom = [list(range(*aoshift_by_atom[ia:ia+2]))
            for ia in range(natom)]
    else:
    
        nmotif = len(atomind_by_motif)
        assert(
            set([ia for im in range(nmotif)
            for ia in atomind_by_motif[im]]) == set(range(natom))
        )
        aoind_by_atom = [[] for im in range(nmotif)]
        for im in range(nmotif):
            for ia in atomind_by_motif[im]:
                aoind_by_atom[im] += list(
                    range(*aoshift_by_atom[ia:ia+2]))
    
    return aoind_by_atom


def reorder_by_atom_(Clo, aoind_by_atom, S, thr=0.5):
    import numpy as np
    
    natom = len(aoind_by_atom)
    nlo = Clo.shape[1]
    X = get_symm_mat_pow(S, 0.5)
    
    Clo_soao = X @ Clo
    
    loind_reorder = []
    loind_by_atom = [None] * natom
    loshift = 0
    for ia in range(natom):
        ra = aoind_by_atom[ia]
        poplo_by_atom = np.sum(Clo_soao[ra]**2., axis=0)
        loind_a = np.where(poplo_by_atom>thr)[0].tolist()
        loind_reorder += loind_a
        nlo_a = len(loind_a)
        loind_by_atom[ia] = list(range(loshift,loshift+nlo_a))
        loshift += nlo_a
    if loind_reorder != list(range(nlo)):
        print('REORDERD')
        Clo_new = Clo[:,loind_reorder]
    else:
        Clo_new = Clo
    return Clo_new, loind_by_atom

class KMF:
    def __init__(self, cell, kpts = None, mo_coeff = None, mo_energy = None):
        self. cell = cell
        self.kpts = kpts
        self.mo_coeff = mo_coeff.copy()
        self.mo_energy = mo_energy
        self.mo_energy_kpts = mo_energy
        self.mo_coeff_kpts = mo_coeff.copy()

def localize(self, lo_method, mol=None, valence_basis='sto-3g', iao_wannier=True):
    from numpy.linalg import eigh
    from pyscf.lo.iao import iao
    import scipy.linalg,functools
    from  .helper import ncore_
    from .pbcgeom import sgeom
    
    if lo_method == 'lowdin':
        
        es_, vs_ = eigh(self.S)
        edx = es_ > 1.e-15                
        self.W = numpy.dot(vs_[:,edx]/numpy.sqrt(es_[edx]), vs_[:,edx].T)
        if self.frozen_core:
                
            P_core = numpy.eye(self.W.shape[0]) - numpy.dot(self.P_core, self.S)
            C_ = numpy.dot(P_core, self.W)

            # PYSCF has basis in 1s2s3s2p2p2p3p3p3p format
            # fix no_core_idx - use population for now
            #C_ = C_[:,self.no_core_idx]
            Cpop = functools.reduce(numpy.dot,
                                    (C_.T, self.S, C_))
            Cpop = numpy.diag(Cpop)
            no_core_idx = numpy.where(Cpop > 0.7)[0]
            C_ = C_[:,no_core_idx]

            S_ = functools.reduce(numpy.dot, (C_.T, self.S, C_))
            es_, vs_ = eigh(S_)
            s_ = numpy.sqrt(es_)
            s_ = numpy.diag(1.0/s_)
            W_ = functools.reduce(numpy.dot,
                                  (vs_, s_, vs_.T))
            self.W = numpy.dot(C_, W_)
                
            
        if not self.frozen_core:                 
            self.lmo_coeff = functools.reduce(numpy.dot,
                                              (self.W.T, self.S, self.C))
        else:            
            self.lmo_coeff = functools.reduce(numpy.dot,
                                              (self.W.T, self.S, self.C[:,self.ncore:]))            

    elif lo_method=='iao':
        
        from pyscf import lo
        import os, h5py

        # Things I recommend having as parameters
        loc_type = 'SO'
        val_basis = 'sto-3g'
        
        # Occupied mo_coeff (with core)
        Co = self.C[:,:self.Nocc]
        # Get necessary overlaps, second arg is IAO basis
        S12, S2 = get_xovlp(self.mol, basis=val_basis)
        # Use these to get IAOs
        Ciao = get_iao(Co, S12, self.S, S2 = S2)

        # Now get PAOs
        if loc_type.upper() != 'SO':
            Cpao = get_pao(Ciao, self.S, S12, S2, self.mol)
        elif loc_type.upper() == 'SO':
            Cpao = get_pao_native(Ciao, self.S, self.mol, valence_basis=val_basis)
        #else:
        #    raise NotImplementedError('Localization method', loc_type, 'not understood.')

        # rearrange by atom
        aoind_by_atom = get_aoind_by_atom(self.mol)
        Ciao, iaoind_by_atom = reorder_by_atom_(Ciao, aoind_by_atom, self.S)
        
        Cpao, paoind_by_atom = reorder_by_atom_(Cpao, aoind_by_atom, self.S)
        
        if self.frozen_core:
            # Remove core MOs
            Cc = self.C[:,:self.ncore] # Assumes core are first
            Ciao = remove_core_mo(Ciao, Cc, self.S)
                    
        # Localize orbitals beyond symm orth
        if loc_type.upper() != 'SO':
            Ciao = get_loc(self.mol, Ciao, loc_type)
            Cpao = get_loc(self.mol, Cpao, loc_type)
        
        #self.W = numpy.hstack([Ciao,  Cpao])
        # stack here
        shift = 0
        ncore = 0
        
        Wstack = numpy.zeros((Ciao.shape[0], Ciao.shape[1]+Cpao.shape[1])) #-self.ncore))
        if self.frozen_core:            
            for ix in range(self.mol.natm):
                nc = ncore_(self.mol.atom_charge(ix))
                ncore += nc
                niao = len(iaoind_by_atom[ix])
                iaoind_ix = [ i_ - ncore for i_ in iaoind_by_atom[ix][nc:]]
                Wstack[:, shift:shift+niao-nc] = Ciao[:, iaoind_ix]                
                shift += niao-nc
                npao = len(paoind_by_atom[ix])
                Wstack[:,shift:shift+npao] = Cpao[:, paoind_by_atom[ix]]                    
                shift += npao
        else:                    
            for ix in range(self.mol.natm):
                niao = len(iaoind_by_atom[ix])
                Wstack[:, shift:shift+niao] = Ciao[:, iaoind_by_atom[ix]]
                shift += niao
                npao = len(paoind_by_atom[ix])
                Wstack[:,shift:shift+npao] = Cpao[:, paoind_by_atom[ix]]
                shift += npao      
                
        self.W = Wstack            
        assert(numpy.allclose(self.W.T @ self.S @ self.W, numpy.eye(self.W.shape[1])))
        nmo = self.C.shape[1] - self.ncore
        nlo = self.W.shape[1]
        
        if nmo > nlo:
            Co_nocore = self.C[:,self.ncore:self.Nocc]
            Cv = self.C[:,self.Nocc:]
            # Ensure that the LOs span the occupied space
            assert(numpy.allclose(numpy.sum((self.W.T @ self.S @ Co_nocore)**2.),
                                  self.Nocc - self.ncore))
            # Find virtual orbitals that lie in the span of LOs
            u, l, vt = numpy.linalg.svd(self.W.T @ self.S @ Cv, full_matrices=False)
            nvlo = nlo - self.Nocc - self.ncore
            assert(numpy.allclose(numpy.sum(l[:nvlo]), nvlo))
            C_ = numpy.hstack([Co_nocore, Cv @ vt[:nvlo].T])
            self.lmo_coeff = self.W.T @ self.S @ C_
        else:
            self.lmo_coeff = self.W.T @ self.S @ self.C[:,self.ncore:]
        assert(numpy.allclose(self.lmo_coeff.T @ self.lmo_coeff, numpy.eye(self.lmo_coeff.shape[1])))            
    elif lo_method == 'boys':
        from pyscf.lo import Boys
        es_, vs_ = eigh(self.S)
        edx = es_ > 1.e-15                
        W_ = numpy.dot(vs_[:,edx]/numpy.sqrt(es_[edx]), vs_[:,edx].T)
        if self.frozen_core:                    
            P_core = numpy.eye(W_.shape[0]) - numpy.dot(self.P_core, self.S)
            C_ = numpy.dot(P_core, W_)
            Cpop = functools.reduce(numpy.dot,
                                    (C_.T, self.S, C_))
            Cpop = numpy.diag(Cpop)
            no_core_idx = numpy.where(Cpop > 0.55)[0]
            C_ = C_[:,no_core_idx]
            S_ = functools.reduce(numpy.dot, (C_.T, self.S, C_))
            es_, vs_ = eigh(S_)
            s_ = numpy.sqrt(es_)
            s_ = numpy.diag(1.0/s_)
            W_ = functools.reduce(numpy.dot,
                                  (vs_, s_, vs_.T))
            W_ = numpy.dot(C_, W_)            
        
        self.W = get_loc(self.mol, W_, 'BOYS')
        
        if not self.frozen_core:
            self.lmo_coeff = self.W.T @ self.S @ self.C
        else:                
            self.lmo_coeff = self.W.T @ self.S @ self.C[:,self.ncore:]   
        
    else:
        print('lo_method = ',lo_method,' not implemented!',flush=True)
        print('exiting',flush=True)
        sys.exit()
