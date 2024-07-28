import numpy,functools,sys, time,os
from molbe.external.ccsd_rdm import make_rdm1_ccsd_t1, make_rdm2_urlx

def be_func(pot, Fobjs, Nocc, solver, enuc, hf_veff=None,
            only_chem = False, nproc=4,hci_pt=False,
            hci_cutoff=0.001, ci_coeff_cutoff = None, select_cutoff=None,
            eeval=False, ereturn=False, frag_energy=False, relax_density=False,
            return_vec=False, ecore=0., ebe_hf=0., be_iter=None, use_cumulant=True):
    """
    Perform bootstrap embedding calculations for each fragment.

    This function computes the energy and/or error for each fragment in a molecular system using various quantum chemistry solvers.

    Parameters
    ----------
    pot : list
        List of potentials.
    Fobjs : list of MolBE.fragpart
        List of fragment objects.
    Nocc : int
        Number of occupied orbitals.
    solver : str
        Quantum chemistry solver to use ('MP2', 'CCSD', 'FCI', 'HCI', 'SHCI', 'SCI').
    enuc : float
        Nuclear energy.
    hf_veff : numpy.ndarray, optional
        Hartree-Fock effective potential. Defaults to None.
    only_chem : bool, optional
        Whether to only optimize the chemical potential. Defaults to False.
    nproc : int, optional
        Number of processors. Defaults to 4. This is only neccessary for 'SHCI' solver
    eeval : bool, optional
        Whether to evaluate the energy. Defaults to False.
    ereturn : bool, optional
        Whether to return the energy. Defaults to False.
    frag_energy : bool, optional
        Whether to calculate fragment energy. Defaults to False.
    relax_density : bool, optional
        Whether to relax the density. Defaults to False.
    return_vec : bool, optional
        Whether to return the error vector. Defaults to False.
    ecore : float, optional
        Core energy. Defaults to 0.
    ebe_hf : float, optional
        Hartree-Fock energy. Defaults to 0.
    be_iter : int or None, optional
        Iteration number. Defaults to None.
    use_cumulant : bool, optional
        Whether to use the cumulant-based energy expression. Defaults to True.

    Returns
    -------
    float or tuple
        Depending on the options, it returns the norm of the error vector, the energy, or a combination of these values.
    """
    from pyscf import fci
    import h5py,os
    from pyscf import ao2mo
    from .helper import get_frag_energy

    rdm_return = False
    if relax_density:
        rdm_return = True
    E = 0.
    if frag_energy or eeval:
        total_e = [0.,0.,0.]

    # Loop over each fragment and solve using the specified solver
    for fobj in Fobjs:
        # Update the effective Hamiltonian
        if not pot is None:
            heff_ = fobj.update_heff(pot, return_heff=True,
                                     only_chem=only_chem)
        else:
            heff_ = None
            
        # Compute the one-electron Hamiltonian
        h1_ = fobj.fock + fobj.heff
        # Perform SCF calculation
        fobj.scf()

        # Solve using the specified solver
        if solver=='MP2': 
            fobj._mc = solve_mp2(fobj._mf, mo_energy=fobj._mf.mo_energy)
        elif solver=='CCSD':
            if rdm_return:
                fobj.t1, fobj.t2, rdm1_tmp, rdm2s = solve_ccsd(fobj._mf,
                                                               mo_energy=fobj._mf.mo_energy,
                                                               relax=True, use_cumulant=use_cumulant,
                                                               rdm2_return=True,
                                                               rdm_return=True)
            else:
                fobj.t1, fobj.t2 = solve_ccsd(fobj._mf,
                                              mo_energy=fobj._mf.mo_energy,
                                              rdm_return=False)
                rdm1_tmp = make_rdm1_ccsd_t1(fobj.t1)
            

        elif solver=='FCI':
            from .helper import get_eri
            import scipy
            
            mc = fci.FCI(fobj._mf, fobj._mf.mo_coeff)
            efci, civec = mc.kernel()            
            rdm1_tmp = mc.make_rdm1(civec, mc.norb, mc.nelec)

        elif solver=='HCI':
            from pyscf import hci
            from .helper import get_eri
            # pilot pyscf.hci only in old versions
            
            nao, nmo = fobj._mf.mo_coeff.shape

            eri = ao2mo.kernel(fobj._mf._eri, fobj._mf.mo_coeff, aosym='s4',
                               compact=False).reshape(4*((nmo),))
            
            ci_ = hci.SCI(fobj._mf.mol)
            if select_cutoff is None and ci_coeff_cutoff is None:
                select_cutoff = hci_cutoff
                ci_coeff_cutoff = hci_cutoff
            elif select_cutoff is None or ci_coeff_cutoff is None:
                sys.exit()
            
            ci_.select_cutoff = select_cutoff
            ci_.ci_coeff_cutoff = ci_coeff_cutoff
            
            nelec = (fobj.nsocc, fobj.nsocc)            
            h1_ = fobj.fock+fobj.heff
            h1_ = functools.reduce(numpy.dot, (fobj._mf.mo_coeff.T, h1_, fobj._mf.mo_coeff))
            eci, civec = ci_.kernel(h1_, eri,  nmo, nelec)
            civec = numpy.asarray(civec)
            
            
            (rdm1a_, rdm1b_), (rdm2aa, rdm2ab, rdm2bb) = ci_.make_rdm12s(civec, nmo, nelec)
            rdm1_tmp = rdm1a_ + rdm1b_
            rdm2s = rdm2aa + rdm2ab + rdm2ab.transpose(2,3,0,1) + rdm2bb
            
        elif solver=='SHCI':
            from pyscf.shciscf import shci
        
            nao, nmo = fobj._mf.mo_coeff.shape
            
            nelec = (fobj.nsocc, fobj.nsocc)
            mch = shci.SHCISCF(fobj._mf, nmo, nelec, orbpath=fobj.dname)
            mch.fcisolver.mpiprefix = 'mpirun -np '+str(nproc)
            if hci_pt:
                mch.fcisolver.stochastic = False
                mch.fcisolver.epsilon2 = hci_cutoff 
            else:
                mch.fcisolver.stochastic = True # this is for PT and doesnt add PT to rdm
                mch.fcisolver.nPTiter = 0
            mch.fcisolver.sweep_iter = [0]
            mch.fcisolver.DoRDM = True
            mch.fcisolver.sweep_epsilon = [ hci_cutoff ]
            mch.fcisolver.scratchDirectory='/scratch/oimeitei/'+jobid+'/'+fobj.dname+jobid
            mch.mc1step()
            rdm1_tmp, rdm2s = mch.fcisolver.make_rdm12(0, nmo, nelec)
           
        elif solver == 'SCI':
            from pyscf import cornell_shci
            from pyscf import ao2mo, mcscf

            nao, nmo = fobj._mf.mo_coeff.shape
            nelec = (fobj.nsocc, fobj.nsocc)
            cas = mcscf.CASCI (fobj._mf, nmo, nelec)
            h1, ecore = cas.get_h1eff(mo_coeff=fobj._mf.mo_coeff)
            eri = ao2mo.kernel(fobj._mf._eri, fobj._mf.mo_coeff, aosym='s4', compact=False).reshape(4*((nmo),))

            ci = cornell_shci.SHCI()
            ci.runtimedir=fobj.dname
            ci.restart=True
            ci.config['var_only'] = True
            ci.config['eps_vars'] = [hci_cutoff]
            ci.config['get_1rdm_csv'] = True
            ci.config['get_2rdm_csv'] = True
            ci.kernel(h1, eri, nmo, nelec)
            rdm1_tmp, rdm2s = ci.make_rdm12(0,nmo,nelec)


        else:
            print('Solver not implemented',flush=True)
            print('exiting',flush=True)
            sys.exit()

        if solver=='MP2':
            rdm1_tmp = fobj._mc.make_rdm1()
        fobj.__rdm1 = rdm1_tmp.copy()
        fobj._rdm1 = functools.reduce(numpy.dot,
                                      (fobj.mo_coeffs,
                                       #fobj._mc.make_rdm1(),
                                       rdm1_tmp,
                                       fobj.mo_coeffs.T))*0.5

        if eeval or ereturn:
            if solver =='CCSD' and not rdm_return:
                with_dm1 = True
                if use_cumulant: with_dm1=False
                rdm2s = make_rdm2_urlx(fobj.t1, fobj.t2, with_dm1=with_dm1)
            elif solver == 'MP2':
                rdm2s = fobj._mc.make_rdm2()
            elif solver =='FCI':
                rdm2s = mc.make_rdm2(civec, mc.norb, mc.nelec)
                if use_cumulant:
                    hf_dm = numpy.zeros_like(rdm1_tmp)
                    hf_dm[numpy.diag_indices(fobj.nsocc)] += 2.
                    del_rdm1 = rdm1_tmp.copy()
                    del_rdm1[numpy.diag_indices(fobj.nsocc)] -= 2.
                    nc = numpy.einsum('ij,kl->ijkl',hf_dm, hf_dm) + \
                        numpy.einsum('ij,kl->ijkl',hf_dm, del_rdm1) + \
                        numpy.einsum('ij,kl->ijkl',del_rdm1, hf_dm)
                    nc -= (numpy.einsum('ij,kl->iklj',hf_dm, hf_dm) + \
                           numpy.einsum('ij,kl->iklj',hf_dm, del_rdm1) + \
                           numpy.einsum('ij,kl->iklj',del_rdm1, hf_dm))*0.5
                    rdm2s -= nc
            fobj.__rdm2 = rdm2s.copy()
            if frag_energy or eeval:
                # Find the energy of a given fragment, with the cumulant definition. 
                # Return [e1, e2, ec] as e_f and add to the running total_e.
                e_f = get_frag_energy(fobj._mo_coeffs, fobj.nsocc, fobj.nfsites,
                                      fobj.efac, fobj.TA, fobj.h1, hf_veff, rdm1_tmp,
                                      rdm2s, fobj.dname, eri_file=fobj.eri_file, veff0=fobj.veff0)
                total_e = [sum(x) for x in zip(total_e, e_f)]
                fobj.energy_hf()
                
    if frag_energy or eeval:
        Ecorr = sum(total_e)
    
    if frag_energy:
        return (Ecorr, total_e)
    
    ernorm, ervec = solve_error(Fobjs,Nocc, only_chem=only_chem)
    
    if return_vec:
        return (ernorm, ervec, [Ecorr, total_e])

    if eeval:
        print('Error in density matching      :   {:>2.4e}'.format(ernorm), flush=True)

    return ernorm

    
def solve_error(Fobjs, Nocc, only_chem=False):
    """
    Compute the error for self-consistent fragment density matrix matching.

    This function calculates the error in the one-particle density matrix for a given fragment,
    matching the density matrix elements of the edges and centers. It returns the norm of the error
    vector and the error vector itself.

    Parameters
    ----------
    Fobjs : list of MolBE.fragpart
        List of fragment objects.
    Nocc : int
        Number of occupied orbitals.

    Returns
    -------
    float
        Norm of the error vector.
    numpy.ndarray
        Error vector.
    """
    import math
    
    err_edge = []
    err_chempot = 0.

    if only_chem:
        for fobj in Fobjs:
            # Compute chemical potential error for each fragment
            for i in fobj.efac[1]:
                err_chempot += fobj._rdm1[i,i]
        err_chempot /= Fobjs[0].unitcell_nkpt
        err = err_chempot - Nocc
        
        return abs(err), numpy.asarray([err])

    # Compute edge and chemical potential errors
    for fobj in Fobjs:       
        #match rdm-edge
        for edge in fobj.edge_idx:

            for j_ in range(len(edge)):
                for k_ in range(len(edge)):
                    if j_>k_:
                        continue
                    err_edge.append(fobj._rdm1[edge[j_], edge[k_]])
        #chem potential        
        for i in fobj.efac[1]:
            err_chempot += fobj._rdm1[i,i]
            
    err_chempot /= Fobjs[0].unitcell_nkpt
    err_edge.append(err_chempot) # far-end edges are included as err_chempot

    # Compute center errors
    err_cen = []
    for findx, fobj in enumerate(Fobjs):  
        # Match RDM for centers
        for cindx, cens in enumerate(fobj.center_idx):            
            lenc = len(cens)
            for j_ in range(lenc):
                for k_ in range(lenc):
                    if j_>k_:
                        continue                    
                    err_cen.append(Fobjs[fobj.center[cindx]]._rdm1[cens[j_],
                                                                   cens[k_]])           
    
    err_cen.append(Nocc)
    err_edge = numpy.array(err_edge)
    err_cen = numpy.array(err_cen)  

    # Compute the error vector
    err_vec = err_edge - err_cen

    # Compute the norm of the error vector
    norm_ = numpy.mean(err_vec * err_vec)**0.5
    
    return norm_, err_vec

def solve_mp2(mf, frozen=None, mo_coeff=None, mo_occ=None, mo_energy=None):
    """
    Perform an MP2 (2nd order Moller-Plesset perturbation theory) calculation.

    This function sets up and runs an MP2 calculation using the provided mean-field object.
    It returns the MP2 object after the calculation.

    Parameters
    ----------
    mf : pyscf.scf.hf.RHF
        Mean-field object from PySCF.
    frozen : list or int, optional
        List of frozen orbitals or number of frozen core orbitals. Defaults to None.
    mo_coeff : numpy.ndarray, optional
        Molecular orbital coefficients. Defaults to None.
    mo_occ : numpy.ndarray, optional
        Molecular orbital occupations. Defaults to None.
    mo_energy : numpy.ndarray, optional
        Molecular orbital energies. Defaults to None.

    Returns
    -------
    pyscf.mp.mp2.MP2
        The MP2 object after running the calculation.
    """
    from pyscf import mp

    # Set default values for optional parameters
    if  mo_coeff is None: mo_coeff = mf.mo_coeff
    if  mo_energy is None: mo_energy = mf.mo_energy
    if  mo_occ is None: mo_occ = mf.mo_occ

    # Initialize the MP2 object
    pt__ = mp.MP2(mf, frozen=frozen, mo_coeff=mo_coeff, mo_occ=mo_occ)
    mf = None
    pt__.verbose=0

    # Run the MP2 calculation
    pt__.kernel(mo_energy=mo_energy)

    return pt__



def solve_ccsd(mf, frozen=None, mo_coeff=None,relax=False, use_cumulant=False, with_dm1=True,rdm2_return = False,
               mo_occ=None, mo_energy=None, rdm_return=False, verbose=0):
    """
    Solve the CCSD (Coupled Cluster with Single and Double excitations) equations.

    This function sets up and solves the CCSD equations using the provided mean-field object.
    It can return the CCSD amplitudes (t1, t2), the one- and two-particle density matrices, and the CCSD object.

    Parameters
    ----------
    mf : pyscf.scf.hf.RHF
        Mean-field object from PySCF.
    frozen : list or int, optional
        List of frozen orbitals or number of frozen core orbitals. Defaults to None.
    mo_coeff : numpy.ndarray, optional
        Molecular orbital coefficients. Defaults to None.
    relax : bool, optional
        Whether to use relaxed density matrices. Defaults to False.
    use_cumulant : bool, optional
        Whether to use cumulant-based energy expression. Defaults to False.
    with_dm1 : bool, optional
        Whether to include one-particle density matrix in the two-particle density matrix calculation. Defaults to True.
    rdm2_return : bool, optional
        Whether to return the two-particle density matrix. Defaults to False.
    mo_occ : numpy.ndarray, optional
        Molecular orbital occupations. Defaults to None.
    mo_energy : numpy.ndarray, optional
        Molecular orbital energies. Defaults to None.
    rdm_return : bool, optional
        Whether to return the one-particle density matrix. Defaults to False.
    verbose : int, optional
        Verbosity level. Defaults to 0.

    Returns
    -------
    tuple
        - t1 (numpy.ndarray): Single excitation amplitudes.
        - t2 (numpy.ndarray): Double excitation amplitudes.
        - rdm1a (numpy.ndarray, optional): One-particle density matrix (if rdm_return is True).
        - rdm2s (numpy.ndarray, optional): Two-particle density matrix (if rdm2_return is True and rdm_return is True).
        - cc__ (pyscf.cc.ccsd.CCSD, optional): CCSD object (if rdm_return is True and rdm2_return is False).
    """
    from pyscf import cc
    from pyscf.cc.ccsd_rdm import make_rdm2
    from molbe.external.rdm_ccsd import make_rdm1_ccsd_t1, make_rdm2_urlx

    # Set default values for optional parameters
    if  mo_coeff is None: mo_coeff = mf.mo_coeff
    if  mo_energy is None: mo_energy = mf.mo_energy
    if  mo_occ is None: mo_occ = mf.mo_occ

    # Initialize the CCSD object
    cc__ = cc.CCSD(mf, frozen=frozen, mo_coeff=mo_coeff,
                   mo_occ = mo_occ)
    cc__.verbose=0
    mf = None
    cc__.incore_complete=True

    # Prepare the integrals and Fock matrix
    eris = cc__.ao2mo()
    eris.mo_energy=mo_energy
    eris.fock = numpy.diag(mo_energy)

    # Solve the CCSD equations
    try:
        cc__.verbose=verbose
        cc__.kernel(eris=eris)
    except:
        print(flush=True)
        print('Exception in CCSD -> applying level_shift=0.2, diis_space=25',flush=True)
        print(flush=True)
        cc__.verbose=4
        cc__.diis_space=25
        cc__.level_shift=0.2
        cc__.kernel(eris=eris)

    # Extract the CCSD amplitudes
    t1 = cc__.t1
    t2 = cc__.t2

    # Compute and return the density matrices if requested
    if rdm_return:
        if not relax:
            l1 = numpy.zeros_like(t1)
            l2 = numpy.zeros_like(t2)
            rdm1a = cc.ccsd_rdm.make_rdm1(cc__, t1, t2,
                                          l1,l2)
        else:
            rdm1a = cc__.make_rdm1(with_frozen=False)

        if rdm2_return:
            if use_cumulant: with_dm1 = False
            rdm2s = make_rdm2(cc__, cc__.t1, cc__.t2, cc__.l1, cc__.l2, with_frozen=False, ao_repr=False, with_dm1=with_dm1)
            return(t1, t2, rdm1a, rdm2s)
        return(t1, t2, rdm1a, cc__)

    return (t1, t2)


def schmidt_decomposition(mo_coeff, nocc, Frag_sites, cinv = None, rdm=None):
    """
    Perform Schmidt decomposition on the molecular orbital coefficients.

    This function decomposes the molecular orbitals into fragment and environment parts
    using the Schmidt decomposition method. It computes the transformation matrix (TA)
    which includes both the fragment orbitals and the entangled bath.

    Parameters
    ----------
    mo_coeff : numpy.ndarray
        Molecular orbital coefficients.
    nocc : int
        Number of occupied orbitals.
    Frag_sites : list of int
        List of fragment sites (indices).
    cinv : numpy.ndarray, optional
        Inverse of the transformation matrix. Defaults to None.
    rdm : numpy.ndarray, optional
        Reduced density matrix. If not provided, it will be computed from the molecular orbitals. Defaults to None.

    Returns
    -------
    numpy.ndarray
        Transformation matrix (TA) including both fragment and entangled bath orbitals.
    """    
    import scipy.linalg
    import functools

    # Threshold for eigenvalue significance
    thres = 1.0e-10

    # Compute the reduced density matrix (RDM) if not provided
    if not mo_coeff is None:
        C = mo_coeff[:,:nocc]   
    if rdm is None:
        Dhf = numpy.dot(C, C.T)
        if not cinv is None:
            Dhf = functools.reduce(numpy.dot,
                                   (cinv, Dhf, cinv.conj().T))        
    else:
        Dhf = rdm

    # Total number of sites
    Tot_sites = Dhf.shape[0]

    # Identify environment sites (indices not in Frag_sites)
    Env_sites1 = numpy.array([i for i in range(Tot_sites)
                              if not i in Frag_sites])
    Env_sites = numpy.array([[i] for i in range(Tot_sites)
                             if not i in Frag_sites])
    Frag_sites1 = numpy.array([[i] for i in Frag_sites])

    # Compute the environment part of the density matrix
    Denv = Dhf[Env_sites, Env_sites.T]

    # Perform eigenvalue decomposition on the environment density matrix
    Eval, Evec = numpy.linalg.eigh(Denv)

    # Identify significant environment orbitals based on eigenvalue threshold
    Bidx = []
    for i in range(len(Eval)):
        if thres < numpy.abs(Eval[i]) < 1.0 - thres:         
            Bidx.append(i)

    # Initialize the transformation matrix (TA)
    TA = numpy.zeros([Tot_sites, len(Frag_sites) + len(Bidx)])
    TA[Frag_sites, :len(Frag_sites)] = numpy.eye(len(Frag_sites)) # Fragment part
    TA[Env_sites1,len(Frag_sites):] = Evec[:,Bidx]  # Environment part
    
    return TA
