from .solver import solve_error
from .solver import solve_mp2, solve_ccsd, solve_uccsd
from .solver import make_rdm1_ccsd_t1, make_rdm1_uccsd, make_rdm2_urlx, make_rdm2_uccsd
from .helper import get_frag_energy, get_frag_energy_u
from .frank_sgscf_uhf import make_uhf_obj
import functools, numpy, sys
from .helper import *

def run_solver(h1, dm0, dname, nao, nocc, nfsites,
               efac, TA, hf_veff, h1_e,
               solver='MP2',eri_file='eri_file.h5',eri_files=None,
               hci_cutoff=0.001, ci_coeff_cutoff = None, select_cutoff=None,
               ompnum=4, writeh1=False,
               eeval=True, return_rdm_ao=True, use_cumulant=True, relax_density=False, frag_energy=False):
    eri = get_eri(dname, nao, eri_file=eri_file,eri_files=eri_files)    
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
        if frag_energy:
            # I am NOT returning any RDM's here, just the energies! 
            # We could return both, but I haven't tested it
            e_f = get_frag_energy(mf_.mo_coeff, nocc, nfsites, efac, TA, h1_e, hf_veff, rdm1_tmp, rdm2s, dname, eri_file, eri_files)
            return e_f
#            return (mf_.mo_coeff, rdm1, rdm2s, e_f)

    if return_rdm_ao:
        return(mf_.mo_coeff, rdm1, rdm2s, rdm1_tmp)
    
    return (mf_.mo_coeff, rdm1, rdm2s)

def run_solver_u(fobj_a, fobj_b, nocc, solver, enuc, hf_veff,
                 frag_energy=True, relax_density=False, frozen=False,
                 eri_file='eri_file.h5', use_cumulant=True, ereturn=True):

    fobj_a.scf(unrestricted=True, spin_ind=0)
    fobj_b.scf(unrestricted=True, spin_ind=1)

    full_uhf, eris = make_uhf_obj(fobj_a, fobj_b, frozen=frozen)

    rdm_return = False
    if relax_density:
        rdm_return = True

    if solver=='UCCSD':
        if rdm_return:
               ucc, rdm1_tmp, rdm2s = solve_uccsd(full_uhf, eris, relax=relax_density,
                                                    rdm_return=True, rdm2_return=True,
                                                    frozen=frozen)
        else:
               ucc = solve_uccsd(full_uhf, eris, relax = relax_density, rdm_return=False,
                                                    frozen=frozen)
               rdm1_tmp = make_rdm1_uccsd(ucc, relax=relax_density)

    else:
        print('Solver not implemented',flush=True)
        print('exiting',flush=True) 
        sys.exit()        

    fobj_a.__rdm1 = rdm1_tmp[0].copy()
    fobj_b._rdm1 = functools.reduce(numpy.dot,
                                  (fobj_a._mf.mo_coeff,
                                   rdm1_tmp[0],
                                   fobj_a._mf.mo_coeff.T))*0.5

    fobj_b.__rdm1 = rdm1_tmp[1].copy()
    fobj_b._rdm1 = functools.reduce(numpy.dot,
                                  (fobj_b._mf.mo_coeff,
                                   rdm1_tmp[1],
                                   fobj_b._mf.mo_coeff.T))*0.5
        
    if ereturn:
        if solver =='UCCSD' and not rdm_return:
            with_dm1 = True
            if use_cumulant: with_dm1=False
            rdm2s = make_rdm2_uccsd(ucc, with_dm1=with_dm1)
        else:
            print('rdm return not implemented',flush=True)
            print('exiting',flush=True)
            sys.exit()
        fobj_a.__rdm2 = rdm2s[0].copy() 
        fobj_b.__rdm2 = rdm2s[1].copy()

        if frag_energy:
            if frozen:
                h1_ab = [full_uhf.h1[0]+full_uhf.full_gcore[0]+full_uhf.core_veffs[0],
                        full_uhf.h1[1]+full_uhf.full_gcore[1]+full_uhf.core_veffs[1]]
            else:
                h1_ab = [fobj_a.h1, fobj_b.h1]
            e_f = get_frag_energy_u((fobj_a._mo_coeffs,fobj_b._mo_coeffs),
                                        (fobj_a.nsocc,fobj_b.nsocc),
                                        (fobj_a.nfsites, fobj_b.nfsites),
                                        (fobj_a.efac, fobj_b.efac),
                                        (fobj_a.TA, fobj_b.TA),
                                        h1_ab,
                                        hf_veff,
                                        rdm1_tmp,
                                        rdm2s,
                                        fobj_a.dname,
                                        eri_file=fobj_a.eri_file,
                                        gcores=full_uhf.full_gcore,
                                        frozen=frozen
                                    )
            return e_f

        else:
            print("Non-fragment-wise energy not implemented", flush=True)
            print("exiting", flush=True)


def be_func_parallel(pot, Fobjs, Nocc, solver, enuc, hf_veff=None,
                     nproc=1, ompnum=4,
                     only_chem=False,relax_density=False,use_cumulant=True,
                     eeval=False, ereturn=False, frag_energy=False, ek = 0., kp=1.,
                     hci_cutoff=0.001, ci_coeff_cutoff = None, select_cutoff=None,
                     return_vec=False, ecore=0., ebe_hf=0., be_iter=None, writeh1=False):

    from multiprocessing import Pool
    import os

    nfrag = len(Fobjs)
    if writeh1 and solver=='SCI':
        for nf in range(nfrag):
            dname = Fobjs[nf].dname
            os.system('mkdir '+dname)
    os.system('export OMP_NUM_THREADS='+str(ompnum))
    nprocs = int(nproc/ompnum)

    if not pot is None:    
        for fobj in Fobjs:
            fobj.update_heff(pot, only_chem=only_chem)
        
    pool_ = Pool(nprocs)
    results = []    
    rdms = []
    
    for nf in range(nfrag):
        print("nf", nf)
        
        h1 = Fobjs[nf].fock + Fobjs[nf].heff

        dm0 = Fobjs[nf].dm0.copy()
        dname = Fobjs[nf].dname
        nao = Fobjs[nf].nao
        nocc = Fobjs[nf].nsocc
        nfsites = Fobjs[nf].nfsites
        efac = Fobjs[nf].efac
        TA = Fobjs[nf].TA
        h1_e = Fobjs[nf].h1

        
        result = pool_.apply_async(run_solver, [h1, dm0, dname, nao, nocc, nfsites,
                                                efac, TA, hf_veff, h1_e,
                                                solver,Fobjs[nf].eri_file,Fobjs[nf].eri_files,
                                                hci_cutoff, ci_coeff_cutoff,select_cutoff,
                                                ompnum, writeh1, True, True, use_cumulant, relax_density, frag_energy])
        results.append(result)

    [rdms.append(result.get()) for result in results]
    pool_.close()
    if frag_energy:
        #rdms are the energies: trying _not_ to compile all of the rdms, we only need energy
        e_1 = 0.
        e_2 = 0.
        e_c = 0.
        for i in range(len(rdms)):
            e_1 += rdms[i][0] 
            e_2 += rdms[i][1]
            e_c += rdms[i][2]
        return (e_1+e_2+e_c, (e_1, e_2, e_c))

    Etot = 0.
    for idx, fobj in enumerate(Fobjs):
        fobj.mo_coeffs = rdms[idx][0]
        fobj._rdm1 = rdms[idx][1]
        fobj.__rdm2 = rdms[idx][2]
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

def be_func_parallel_u(pot, Fobjs, Nocc, solver, enuc, hf_veff=None,
                     nproc=1, ompnum=4, relax_density=False, use_cumulant=True,
                     eeval=False, ereturn=False, frag_energy=False,
                     ecore=0., ebe_hf=0., frozen=False):

    from multiprocessing import Pool
    import os

    os.system('export OMP_NUM_THREADS='+str(ompnum))
    nprocs = int(nproc/ompnum)

    pool_ = Pool(nprocs)
    results = []    
    energy_list = []

    for (fobj_a, fobj_b) in Fobjs:

        result = pool_.apply_async(run_solver_u, [fobj_a, fobj_b,
                                                Nocc,
                                                solver,
                                                enuc,
                                                hf_veff,
                                                frag_energy,
                                                relax_density,
                                                frozen,
                                                use_cumulant,
                                                True
                                                ])
        results.append(result)

    [energy_list.append(result.get()) for result in results]
    pool_.close()

    if frag_energy:
        #rdms are the energies: trying _not_ to compile all of the rdms, we only need energy
        e_1 = 0.
        e_2 = 0.
        e_c = 0.
        for i in range(len(energy_list)):
            e_1 += energy_list[i][0] 
            e_2 += energy_list[i][1]
            e_c += energy_list[i][2]
        return (e_1+e_2+e_c, (e_1, e_2, e_c))

    Etot = 0.
    for idx, fobj in enumerate(Fobjs):
        fobj.mo_coeffs = rdms[idx][0]
        fobj._rdm1 = rdms[idx][1]
        fobj.__rdm2 = rdms[idx][2]
        fobj.__rdm1 = rdms[idx][3]
        Etot += fobj.ebe

    del energy_list
    
    Ebe = Etot+enuc+ecore-ek

    if ereturn:        
        return Ebe
    else:
        print("only returns energy")
        sys.exit()

