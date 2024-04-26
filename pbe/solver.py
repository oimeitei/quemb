import numpy,functools,sys, time,os

# from frankestein
# start
def make_rdm1_ccsd_t1(t1):
    nocc, nvir = t1.shape
    nmo = nocc + nvir
    dm = numpy.zeros((nmo,nmo), dtype=t1.dtype)
    dm[:nocc,nocc:] = t1
    dm[nocc:,:nocc] = t1.T
    dm[numpy.diag_indices(nocc)] += 2.

    return dm

def make_rdm2_urlx(t1, t2, with_dm1=True):
    nocc, nvir = t1.shape
    nmo = nocc + nvir

    goovv = (numpy.einsum("ia,jb->ijab", t1, t1) + t2) * 0.5
    dovov = goovv.transpose(0,2,1,3) * 2 - goovv.transpose(1,2,0,3)

    dm2 = numpy.zeros([nmo,nmo,nmo,nmo], dtype=t1.dtype)

    dovov = numpy.asarray(dovov)
    dm2[:nocc,nocc:,:nocc,nocc:] = dovov
    dm2[:nocc,nocc:,:nocc,nocc:]+= dovov.transpose(2,3,0,1)
    dm2[nocc:,:nocc,nocc:,:nocc] = dm2[:nocc,nocc:,:nocc,nocc:].transpose(1,0,3,2).conj()
    dovov = None

    if with_dm1:
        dm1 = make_rdm1_ccsd_t1(t1)
        dm1[numpy.diag_indices(nocc)] -= 2
        
        for i in range(nocc):
            dm2[i,i,:,:] += dm1 * 2
            dm2[:,:,i,i] += dm1 * 2
            dm2[:,i,i,:] -= dm1
            dm2[i,:,:,i] -= dm1.T
        
        for i in range(nocc):
            for j in range(nocc):
                dm2[i,i,j,j] += 4
                dm2[i,j,j,i] -= 2

    return dm2  
# end


def be_func(pot, Fobjs, Nocc, solver, enuc, hf_veff=None,
            only_chem = False, nproc=4,hci_pt=False,
            hci_cutoff=0.001, ci_coeff_cutoff = None, select_cutoff=None,
            eeval=False, ereturn=False, frag_energy=False, ek = 0., kp=1.,relax_density=False,
            return_vec=False, ecore=0., ebe_hf=0., be_iter=None, writeh1=False, use_cumulant=True):
    from pyscf import fci
    import h5py,os
    from pyscf import ao2mo
    from .helper import get_frag_energy

    rdm_return = False
    if relax_density:
        rdm_return = True
    E = 0.
    if frag_energy:
        total_e = [0.,0.,0.]
    t1 = time.time()
    for fobj in Fobjs:
        
        if not pot is None:
            heff_ = fobj.update_heff(pot, return_heff=True,
                                     only_chem=only_chem)
        else:
            heff_ = None
        
        h1_ = fobj.fock + fobj.heff
        fobj.scf()
        if solver=='MP2': # here
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
            if not writeh1:
                mch.fcisolver.restart=True
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
            if frag_energy:
                # Find the energy of a given fragment, with the cumulant definition. 
                # Return [e1, e2, ec] as e_f and add to the running total_e.
                e_f = get_frag_energy(fobj._mo_coeffs, fobj.nsocc, fobj.efac, fobj.TA, fobj.h1, hf_veff, rdm1_tmp, rdm2s, fobj.dname, eri_file=fobj.eri_file)
                total_e = [sum(x) for x in zip(total_e, e_f)]
            if not frag_energy:
                E += fobj.ebe

    E /= Fobjs[0].unitcell_nkpt
    if frag_energy:
        E = sum(total_e)
        return (E, total_e)

    if ereturn:
        # this is really a waste of computation        
        return (E+enuc+ecore-ek)#/kp
    
    ernorm, ervec = solve_error(Fobjs,Nocc, only_chem=only_chem)
    if eeval:
        Ebe = (E+enuc+ecore-ek)#/kp
    if return_vec:
        return (ernorm, ervec, Ebe)

    if eeval:
        #print('BE energy per unit cell        : {:>12.8f} Ha'.format(Ebe), flush=True)
        #print('BE Ecorr  per unit cell        : {:>12.8f} Ha'.format(Ebe-ebe_hf), flush=True)
        print('Error in density matching      :   {:>2.4e}'.format(ernorm), flush=True)

    return ernorm

    
def solve_error(Fobjs, Nocc, only_chem=False):
    import math
    err_edge = []
    err_chempot = 0.

    if only_chem:
        for fobj in Fobjs:
            #chem potential        
            for i in fobj.efac[1]:
                err_chempot += fobj._rdm1[i,i]
        err = err_chempot - Nocc
        
        return abs(err), numpy.asarray([err])

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
    unitcell_nkpt = Fobjs[0].unitcell_nkpt
    
    err_chempot /= unitcell_nkpt
    err_edge.append(err_chempot) # far-end edges are included as err_chempot
    
    err_cen = []
    for findx, fobj in enumerate(Fobjs):        
        #match rdm-center
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
    
    err_vec = err_edge - err_cen
    norm_ = math.sqrt(numpy.sum(err_vec * err_vec))
    norm_ = numpy.mean(err_vec * err_vec)**0.5
    
    return norm_, err_vec


def solve_error_selffrag(Fobjs, Nocc):
    import math
    err_edge = [] 
    err_chempot = 0.


    fobj = Fobjs[0]

    tt_ = fobj._rdm1[:5,:5] - fobj._rdm1[5:10,5:10]
    
    # edge -> center <- edge
    for edge in fobj.edge_idx:
        for j_ in range(len(edge)):
            for k_ in range(len(edge)):
                if j_>k_:
                    continue    
                err_edge.append(fobj._rdm1[edge[j_], edge[k_]])
    
    
    for fidx,fval in enumerate(fobj.fsites):
        if not any(fidx in sublist for sublist in fobj.edge_idx): 
            err_chempot += fobj._rdm1[fidx,fidx]
    err_edge.append(err_chempot)
    
    err_cen_ = []
    
    #match rdm-center
    
    for j_ in fobj.center_idx:
        for k_ in fobj.center_idx:
            if j_>k_:
                continue
            err_cen_.append(fobj._rdm1[j_, k_])
            
            
    err_cen = [*err_cen_, *err_cen_]
    err_cen.append(2.5e0)
    
    
    err_edge = numpy.array(err_edge)
    err_cen = numpy.array(err_cen)
    
    err_vec = err_edge - err_cen
    norm_ = math.sqrt(numpy.sum(err_vec * err_vec))
    norm_ = numpy.mean(err_vec * err_vec)**0.5
    
    
    return norm_, err_vec


def solve_mp2(mf, frozen=None, mo_coeff=None, mo_occ=None, mo_energy=None):
    from pyscf import mp

    if  mo_coeff is None: mo_coeff = mf.mo_coeff
    if  mo_energy is None: mo_energy = mf.mo_energy
    if  mo_occ is None: mo_occ = mf.mo_occ

    pt__ = mp.MP2(mf, frozen=frozen, mo_coeff=mo_coeff, mo_occ=mo_occ)
    mf = None
    pt__.verbose=0
    pt__.kernel(mo_energy=mo_energy)

    return pt__



def solve_ccsd(mf, frozen=None, mo_coeff=None,relax=False, use_cumulant=False, with_dm1=True,rdm2_return = False,
               mo_occ=None, mo_energy=None, rdm_return=False, verbose=0):
    from pyscf import cc
    from pyscf.cc.ccsd_rdm import make_rdm2
    from pbe.external.rdm_ccsd import make_rdm1_ccsd_t1, make_rdm2_urlx

    if  mo_coeff is None: mo_coeff = mf.mo_coeff
    if  mo_energy is None: mo_energy = mf.mo_energy
    if  mo_occ is None: mo_occ = mf.mo_occ
   
    cc__ = cc.CCSD(mf, frozen=frozen, mo_coeff=mo_coeff,
                   mo_occ = mo_occ)
    cc__.verbose=0
    mf = None
    cc__.incore_complete=True

    eris = cc__.ao2mo()
    eris.mo_energy=mo_energy
    eris.fock = numpy.diag(mo_energy)

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

    #cc__.solve_lambda(eris=eris)
    t1 = cc__.t1
    t2 = cc__.t2
    if rdm_return:
        if not relax:
            l1 = numpy.zeros_like(t1)
            l2 = numpy.zeros_like(t2)
            rdm1a = cc.ccsd_rdm.make_rdm1(cc__, t1, t2,
                                          l1,l2)
        else:
            rdm1a = cc__.make_rdm1(with_frozen=False)
                        
        #cc__ = None
        if rdm2_return:
            if use_cumulant: with_dm1 = False
            rdm2s = make_rdm2(cc__, cc__.t1, cc__.t2, cc__.l1, cc__.l2, with_frozen=False, ao_repr=False, with_dm1=with_dm1)
            return(t1, t2, rdm1a, rdm2s)
        return(t1, t2, rdm1a, cc__)
    #cc__ = None
    return (t1, t2)

def pretty(dm):
    for i in dm:
        for j in i:
            print('{:>10.6f} '.format(j),end=' ')
        print()
    print()

def schmidt_decomposition(mo_coeff, nocc, Frag_sites, cinv = None, rdm=None,tmpa=None):
    import scipy.linalg
    import functools
    thres = 1.0e-10
    
    if not mo_coeff is None:
        C = mo_coeff[:,:nocc]   
    if rdm is None:
        Dhf = numpy.dot(C, C.T)
        if not cinv is None:
            Dhf = functools.reduce(numpy.dot,
                                   (cinv, Dhf, cinv.conj().T))        
    else:
        Dhf = rdm

    Tot_sites = Dhf.shape[0]        
    Env_sites1 = numpy.array([i for i in range(Tot_sites)
                              if not i in Frag_sites])
    Env_sites = numpy.array([[i] for i in range(Tot_sites)
                             if not i in Frag_sites])
    Frag_sites1 = numpy.array([[i] for i in Frag_sites])
    Denv = Dhf[Env_sites, Env_sites.T]
    Eval, Evec = numpy.linalg.eigh(Denv)
    
    Bidx = []
    for i in range(len(Eval)):
        if thres < numpy.abs(Eval[i]) < 1.0 - thres:         
            Bidx.append(i)

    TA = numpy.zeros([Tot_sites, len(Frag_sites) + len(Bidx)])
    TA[Frag_sites, :len(Frag_sites)] = numpy.eye(len(Frag_sites))
    TA[Env_sites1,len(Frag_sites):] = Evec[:,Bidx]
    
    return TA


def schmidt_decomp_rdm1_new(rdm, Frag_sites):
    import scipy.linalg
    import functools
    
    thres = 1.0e-9

    Tot_sites = rdm.shape[0]

    Fragsites = [i if i>=0 else Tot_sites+i for i in Frag_sites]
    
    Env_sites1 = numpy.array([i for i in range(Tot_sites)
                              if not i in Fragsites])
    Env_sites = numpy.array([[i] for i in range(Tot_sites)
                             if not i in Fragsites])
    Frag_sites1 = numpy.array([[i] for i in Fragsites])
    
    Denv_ = rdm[Env_sites1][:, Fragsites]    
    U, sigma, V = scipy.linalg.svd(Denv_, full_matrices=False)
    nbath = (sigma >= thres).sum()

    Denv = rdm[Env_sites, Env_sites.T]
    Eval, Evec = scipy.linalg.eigh(Denv)
    Bidx = []
    for i in range(len(Eval)):
        if thres < numpy.abs(Eval[i]) < 1.0 - thres:         
            Bidx.append(i)

    TA = numpy.zeros([Tot_sites, len(Frag_sites) + len(Bidx)], dtype=rdm.dtype)
    TA[Frag_sites, :len(Frag_sites)] = numpy.eye(len(Frag_sites))
    TA[Env_sites1,len(Frag_sites):] = Evec[:,Bidx]

    return TA

def schmidt_decomp_rdm1(rdm, Frag_sites):
    import scipy.linalg
    import functools
    thres = 1.0e-9

    Tot_sites = rdm.shape[0]        
    Env_sites1 = numpy.array([i for i in range(Tot_sites)
                              if not i in Frag_sites])
    Env_sites = numpy.array([[i] for i in range(Tot_sites)
                             if not i in Frag_sites])
    Frag_sites1 = numpy.array([[i] for i in Frag_sites])


    Denv = rdm[Env_sites, Env_sites.T]
    Eval, Evec = numpy.linalg.eigh(Denv)

    Eval_1 = numpy.abs(numpy.sqrt(numpy.abs(1.-Eval**2))-1.)

    Bidx = []    
    for i in range(len(Eval_1)):
        if Eval_1[i] > thres:
            Bidx.append(i)

    TA = numpy.zeros([Tot_sites, len(Frag_sites) + len(Bidx)], dtype=rdm.dtype)
    TA[Frag_sites, :len(Frag_sites)] = numpy.eye(len(Frag_sites))
    TA[Env_sites1,len(Frag_sites):] = Evec[:,Bidx]

    return TA


def schmidt_decomp_svd(rdm, Frag_sites):
    import scipy.linalg
    import functools
    
    thres = 1.0e-10
    Tot_sites = rdm.shape[0]     
    
    Fragsites = [i if i>=0 else Tot_sites+i for i in Frag_sites]   
    Env_sites1 = numpy.array([i for i in range(Tot_sites)
                              if not i in Fragsites])
    Denv = rdm[Env_sites1][:, Fragsites].copy()
    U, sigma, V = scipy.linalg.svd(Denv, full_matrices=False, lapack_driver='gesvd')
    
    nbath = (sigma >= thres).sum()
    nfs = len(Frag_sites)
    TA = numpy.zeros((Tot_sites, nfs + nbath), dtype=numpy.float64)
    TA[Frag_sites, :nfs] = numpy.eye(nfs, dtype=numpy.float64)
    TA[Env_sites1, nfs:] = U[:,:nbath]
    
    return TA
