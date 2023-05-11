from .solver import be_func
from .be_parallel import be_func_parallel
import scipy,sys,numpy,time, h5py

class StoreOPTQN:

    def __init__(self, max_iter=200):

        self.f0 = None
        self.pot = None
        self.J0 = None
        self.fk = [[] for i in range(max_iter)]
        self.xk = [[] for i in range(max_iter)]



class BEOPT:

    def __init__(self, pot, Fobjs, Nocc, enuc, kp,solver='MP2',ek=0., ecore=0.,
                 nproc=1,ompnum=4,
                 only_chem=False,hci_pt=False,
                 max_space=500, conv_tol = 1.e-6,relax_density = False,
                 hci_cutoff=0.001, ci_coeff_cutoff = None, select_cutoff=None,
                 self_match=False, ebe_hf =0.):
        
        self.ebe_hf=ebe_hf
        self.pot = pot
        self.Fobjs = Fobjs
        self.Nocc = Nocc
        self.enuc = enuc
        self.kp = kp
        self.solver=solver
        self.ek = ek
        self.ecore = ecore
        self.iter = 0
        self.err = 0.0
        self.Ebe = 0.0
        self.self_match= self_match
        self.max_space=max_space
        self.nproc = nproc
        self.ompnum = ompnum
        self.only_chem=only_chem
        self.conv_tol = conv_tol
        self.relax_density = relax_density
        # HCI parameters
        self.hci_cutoff = hci_cutoff
        self.ci_coeff_cutoff = ci_coeff_cutoff
        self.select_cutoff = select_cutoff
        self.hci_pt = hci_pt

    def callback(self, xk):
        
        print('-- In iter ',self.iter, flush=True)

        
        err_ = be_func(xk, self.Fobjs, self.Nocc, self.solver, self.enuc,
                       eeval=True, kp=self.kp, ek = self.ek, ecore= self.ecore)
                       
        print(flush=True)
        self.iter += 1
        if abs(err_) < 1.e-5:
            print(flush=True)
            print('CONVERGED',flush=True)
            print('-----------------------------------------------------',
                  flush=True)
            print(flush=True)
            sys.exit()

    def objfunc_(self, xk, writeh1=False):


        if self.nproc == 1:
            err_, errvec_,ebe_ = be_func(xk, self.Fobjs, self.Nocc, self.solver, self.enuc,
                                         eeval=True, kp=self.kp, ek = self.ek, return_vec=True,
                                         only_chem=self.only_chem,
                                         hci_cutoff=self.hci_cutoff,
                                         nproc=self.ompnum, relax_density=self.relax_density,
                                         ci_coeff_cutoff = self.ci_coeff_cutoff,
                                         select_cutoff = self.select_cutoff, hci_pt=self.hci_pt,
                                         ecore=self.ecore, ebe_hf=self.ebe_hf, be_iter=self.iter, writeh1=writeh1)
        else:
            err_, errvec_,ebe_ = be_func_parallel(xk, self.Fobjs, self.Nocc, self.solver, self.enuc,
                                                  eeval=True, kp=self.kp, ek = self.ek, return_vec=True,
                                                  nproc=self.nproc, ompnum=self.ompnum,
                                                  only_chem=self.only_chem,
                                                  hci_cutoff=self.hci_cutoff,relax_density=self.relax_density,
                                                  ci_coeff_cutoff = self.ci_coeff_cutoff,
                                                  select_cutoff = self.select_cutoff,
                                                  ecore=self.ecore, ebe_hf=self.ebe_hf, be_iter=self.iter,
                                                  writeh1=writeh1)
            
        self.err = err_
        self.Ebe = ebe_
        return errvec_
    
    def objfunc_err(self, xk):
        err_, errvec_,ebe_ = be_func(xk, self.Fobjs, self.Nocc, self.solver, self.enuc,
                                eeval=True, kp=self.kp, ek = self.ek, return_vec=True,
                                     ecore=self.ecore, ebe_hf=self.ebe_hf)
        self.err = err_
        self.Ebe = ebe_
        return err_

    def optimize(self, method, J0 = None, restore_debug=False, save_debug=False,
                 save_fname=None, restore_ = None, file_opt=None):
        from .optqn import FrankQN
        print(flush=True)
        print('-----------------------------------------------------',
                  flush=True)
        print('             Starting BE optimization ', flush=True)
        print('             Solver : ',self.solver,flush=True)
        print('-----------------------------------------------------',
                  flush=True)
        print(flush=True)
        if method == 'bfgs':
            self.callback(self.pot)
            
            res = scipy.optimize.minimize(self.objfunc_err, self.pot,
                                          method='BFGS',callback=self.callback,
                                          options={'maxiter':200})
        elif method=='QN':

            if save_debug:
                file_opt = h5py.File(save_fname, 'w')
            if restore_debug:
                restore_ = h5py.File(save_fname, 'r')
                
            print('-- In iter ',self.iter, flush=True)
            if not restore_debug:
                
                f0 = self.objfunc_(self.pot, writeh1=True)
                           

            if save_debug:
                
                file_opt.create_dataset('f0', data=f0)
                file_opt.create_dataset('pot', data=self.pot)
                file_opt.create_dataset('J0', data=J0)
                file_opt.close()

            
            #print('BE energy per unit cell        : {:>12.8f} Ha'.format(self.Ebe), flush=True)
            #print('BE Ecorr  per unit cell        : {:>12.8f} Ha'.format(self.Ebe-self.ebe_hf), flush=True)
            print('Error in density matching      :   {:>2.4e}'.format(self.err), flush=True)
            print(flush=True)

            if not restore_debug:
                optQN = FrankQN(self.objfunc_, numpy.array(self.pot),
                                f0, J0,
                                max_space=self.max_space)
            else:
                pot__ = numpy.array(restore_.get('pot'))
                f0__ =  numpy.array(restore_.get('f0'))
                J0__ =  numpy.array(restore_.get('J0'))
                restore_.close()
                optQN = FrankQN(self.objfunc_, numpy.array(pot__), f0__, J0__)
                
            for iter_ in range(self.max_space):
                
            

                optQN.next_step(save_debug=save_debug,restore_debug=restore_debug,
                                save_fname=save_fname)
                self.iter += 1
                print('-- In iter ',self.iter, flush=True)                
                #print('BE energy per unit cell        : {:>12.8f} Ha'.format(self.Ebe), flush=True)
                #print('BE Ecorr  per unit cell        : {:>12.8f} Ha'.format(self.Ebe-self.ebe_hf), flush=True)
                print('Error in density matching      :   {:>2.4e}'.format(self.err), flush=True)
                print(flush=True)
                if self.err < self.conv_tol:
                    print(flush=True)
                    print('CONVERGED',flush=True)
                    print('-----------------------------------------------------',
                          flush=True)
                    print(flush=True)
                    break
            
            
            

def optimize(self, solver='MP2',method='bfgs',restore_debug=False, save_debug=False,
             only_chem=False, conv_tol = 1.e-7,relax_density=False, use_cumulant=True,
             save_fname='save_optqn_h5file.h5', J0=None, nproc=1, ompnum=4, max_iter=500):
    
    if not only_chem:
        pot = self.pot
    else:
        pot = [0.]
    
    be_ = BEOPT(pot, self.Fobjs, self.Nocc, self.enuc, self.kpts,
                nproc=nproc, ompnum=ompnum,
                max_space=max_iter,conv_tol = conv_tol,
                only_chem=only_chem,
                hci_cutoff=self.hci_cutoff,
                ci_coeff_cutoff = self.ci_coeff_cutoff,relax_density=relax_density,
                select_cutoff = self.select_cutoff,hci_pt=self.hci_pt,
                solver=solver, ek=self.ek, ecore=self.E_core, self_match=self.self_match, ebe_hf=self.ebe_hf)

    if method=='bfgs':
        be_.optimize(method)
    elif method=='QN':
        if not restore_debug:

            if only_chem:
                J0 = [[0.]]
                J0 = self.get_be_error_jacobian(jac_solver='HF')
                J0 = [[J0[-1,-1]]]
                
            else:
                J0 = self.get_be_error_jacobian(jac_solver='HF')
                        
        
        be_.optimize(method, J0=J0, save_debug=save_debug,restore_debug=restore_debug,
                                save_fname=save_fname)

        
