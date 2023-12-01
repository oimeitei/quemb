from .solver import solve_error
from .solver import solve_mp2, solve_ccsd,make_rdm1_ccsd_t1
from .solver import make_rdm2_urlx
import functools, numpy, sys
from .helper import *

def run_solver(h1, dm0, dname, nao, nocc,               
               solver='MP2',eri_file='eri_file.h5',
               hci_cutoff=0.001, ci_coeff_cutoff = None, select_cutoff=None,
               ompnum=4, writeh1=False,
               eeval=True, return_rdm_ao=True, use_cumulant=True,relax_density=False):
    
    eri = get_eri(dname, nao, eri_file=eri_file)    
    mf_ = get_scfObj(h1, eri, nocc, dm0=dm0)
    rdm_return = False
    if relax_density:
        rdm_return = True

    if solver=='MP2': 
        mc_ = solve_mp2(mf_, mo_energy=mf_.mo_energy)
        rdm1_tmp = mc_.make_rdm1()
        
    elif solver=='CCSD':
        if not rdm_return:
            t1, t2 = solve_ccsd(mf_,
                                mo_energy=mf_.mo_energy,
                                rdm_return=False)
            rdm1_tmp = make_rdm1_ccsd_t1(t1)
        else:
            t1, t2, rdm1_tmp, rdm2s = solve_ccsd(mf_,
                                                 mo_energy=mf_.mo_energy,
                                                 rdm_return=True,
                                                 rdm2_return = True, use_cumulant=use_cumulant,
                                                 relax=True)
    elif solver=='FCI':
        from pyscf import fci
        
        mc_ = fci.FCI(mf_, mf_.mo_coeff)
        efci, civec = mc_.kernel()
        rdm1_tmp = mc_.make_rdm1(civec, mc_.norb, mc_.nelec)
    elif solver=='HCI':
        from pyscf import hci,ao2mo

        nao, nmo = mf_.mo_coeff.shape
        eri = ao2mo.kernel(mf_._eri, mf_.mo_coeff, aosym='s4',
                               compact=False).reshape(4*((nmo),))            
        ci_ = hci.SCI(mf_.mol)
        if select_cutoff is None and ci_coeff_cutoff is None:
            select_cutoff = hci_cutoff
            ci_coeff_cutoff = hci_cutoff
        elif select_cutoff is None or ci_coeff_cutoff is None:
            sys.exit()
        
        ci_.select_cutoff = select_cutoff
        ci_.ci_coeff_cutoff = ci_coeff_cutoff
        
        nelec = (nocc, nocc)            
        h1_ = functools.reduce(numpy.dot, (mf_.mo_coeff.T, h1, mf_.mo_coeff))
        eci, civec = ci_.kernel(h1_, eri,  nmo, nelec)
        civec = numpy.asarray(civec)
                
        (rdm1a_, rdm1b_), (rdm2aa, rdm2ab, rdm2bb) = ci_.make_rdm12s(civec, nmo, nelec)
        rdm1_tmp = rdm1a_ + rdm1b_
        rdm2s = rdm2aa + rdm2ab + rdm2ab.transpose(2,3,0,1) + rdm2bb        

    elif solver=='SHCI':
        from pyscf.shciscf import shci

        nao, nmo = mf_.mo_coeff.shape

        nelec = (nocc, nocc)
        mch = shci.SHCISCF(mf_, nmo, nelec, orbpath=dname)
        mch.fcisolver.mpiprefix = 'mpirun -np '+str(ompnum)
        mch.fcisolver.stochastic = True # this is for PT and doesnt add PT to rdm
        mch.fcisolver.nPTiter = 0
        mch.fcisolver.sweep_iter = [0]
        mch.fcisolver.DoRDM = True
        mch.fcisolver.sweep_epsilon = [ hci_cutoff ]
        mch.fcisolver.scratchDirectory='/scratch/oimeitei/'+dname
        if not writeh1:
            mch.fcisolver.restart=True
        mch.mc1step()
        rdm1_tmp, rdm2s = mch.fcisolver.make_rdm12(0, nmo, nelec)
    
    elif solver == 'SCI':
            from pyscf import cornell_shci
            from pyscf import ao2mo, mcscf

            nao, nmo = mf_.mo_coeff.shape
            nelec = (nocc, nocc)
            cas = mcscf.CASCI (mf_, nmo, nelec)
            h1, ecore = cas.get_h1eff(mo_coeff=mf_.mo_coeff)
            eri = ao2mo.kernel(mf_._eri, mf_.mo_coeff, aosym='s4', compact=False).reshape(4*((nmo),))

            ci = cornell_shci.SHCI()
            ci.runtimedir=dname
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
    rdm1 = functools.reduce(numpy.dot,
                            (mf_.mo_coeff,
                             rdm1_tmp,
                             mf_.mo_coeff.T))*0.5
    
    if eeval:
        if solver =='CCSD' and not rdm_return:
            with_dm1 = True
            if use_cumulant: with_dm1 = False
            rdm2s = make_rdm2_urlx(t1, t2, with_dm1 = with_dm1)
            
        elif solver == 'MP2':
            rdm2s = mc_.make_rdm2()
        elif solver == 'FCI':
            rdm2s = mc_.make_rdm2(civec, mc_.norb, mc_.nelec)
            if use_cumulant:
                hf_dm = numpy.zeros_like(rdm1_tmp)
                hf_dm[numpy.diag_indices(nocc)] += 2.
                del_rdm1 = rdm1_tmp.copy()
                del_rdm1[numpy.diag_indices(nocc)] -= 2.
                nc = numpy.einsum('ij,kl->ijkl',hf_dm, hf_dm) + \
                    numpy.einsum('ij,kl->ijkl',hf_dm, del_rdm1) + \
                    numpy.einsum('ij,kl->ijkl',del_rdm1, hf_dm)
                nc -= (numpy.einsum('ij,kl->iklj',hf_dm, hf_dm) + \
                       numpy.einsum('ij,kl->iklj',hf_dm, del_rdm1) + \
                       numpy.einsum('ij,kl->iklj',del_rdm1, hf_dm))*0.5
                rdm2s -= nc
    if return_rdm_ao:
        return(mf_.mo_coeff, rdm1, rdm2s, rdm1_tmp)
    
    return (mf_.mo_coeff, rdm1, rdm2s)
    

def be_func_parallel(pot, Fobjs, Nocc, solver, enuc,
                     nproc=1, ompnum=4,
                     only_chem=False,relax_density=False,use_cumulant=True,
                     eeval=False, ereturn=False, ek = 0., kp=1.,
                     hci_cutoff=0.001, ci_coeff_cutoff = None, select_cutoff=None,
                     return_vec=False, ecore=0., ebe_hf=0., be_iter=None, writeh1=False):

    from multiprocessing import Pool
    import os

    nfrag = len(Fobjs)
    if writeh1 and solver=='SCI':
        for nf in range(nfrag):
            dname = Fobjs[nf].dname
            os.system('mkdir '+dname)
            #os.system('mkdir /scratch/oimeitei/'+dname)
    os.system('export OMP_NUM_THREADS='+str(ompnum))
    nprocs = int(nproc/ompnum)

    if not pot is None:    
        for fobj in Fobjs:
            fobj.update_heff(pot, only_chem=only_chem)
        
    pool_ = Pool(nprocs)
    results = []    
    rdms = []
    
    for nf in range(nfrag):
        
        h1 = Fobjs[nf].fock + Fobjs[nf].heff

        dm0 = Fobjs[nf].dm0.copy()
        dname = Fobjs[nf].dname
        nao = Fobjs[nf].nao
        nocc = Fobjs[nf].nsocc

        result = pool_.apply_async(run_solver, [h1, dm0, dname, nao, nocc ,
                                                solver,Fobjs[nf].eri_file,
                                                hci_cutoff, ci_coeff_cutoff,select_cutoff,
                                                ompnum, writeh1, True, True, use_cumulant, relax_density])
        
        results.append(result)

    [rdms.append(result.get()) for result in results]
    pool_.close()

    Etot = 0.
    for idx, fobj in enumerate(Fobjs):
        fobj.mo_coeffs = rdms[idx][0]
        fobj._rdm1 = rdms[idx][1]
        fobj.__rdm1 = rdms[idx][3]
        Etot += fobj.ebe

    Etot /= Fobjs[0].unitcell_nkpt
    del rdms
    
    Ebe = Etot+enuc+ecore-ek

    if ereturn:        
        return Ebe
      
    ernorm, ervec = solve_error(Fobjs,Nocc, only_chem=only_chem)

    if return_vec:
        return (ernorm, ervec, Ebe)

    if eeval:
        print('Error in density matching      :   {:>2.4e}'.format(ernorm), flush=True)        
    return ernorm
