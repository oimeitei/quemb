from pyscf import gto,scf,mp, cc, fci
from pbe.pbe import pbe
from pbe.fragment import fragpart
from pbe.helper import *
from pbe import sgeom, printatom
import sys, h5py, os, pbe_var

pbe_var.PRINT_LEVEL=5

be_type = 'be2' #sys.argv[1]


# PYSCF for integrals and HF solution
mol = gto.M(atom='''
C    -5.58580    0.52340	-0.01779
C    -4.35104    1.41997	-0.01075
H    -6.50160    1.15113	-0.03297
H    -5.60602	 -0.11314    0.89244
H    -5.58558	 -0.12710	-0.91832
C    -3.06331    0.58854    0.01019
H    -4.38813    2.07429    0.88752
H    -4.36773    2.06037	-0.91960
H    -3.03435	 -0.06600	-0.88898
H    -3.05478	 -0.05210    0.91973
C    -1.82582    1.49328    0.01721
C    -0.54129    0.66985    0.03803
H    -1.84662    2.14804    0.91569
H    -1.82622    2.13418	-0.89144
H    -0.50431    0.03493    0.94886
H    0.33667    1.34965    0.04273
H    -0.48386    0.02105	-0.86190
''',basis='sto-3g', charge=0)

mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.kernel()

#mc = cc.CCSD(mf)
#mc.verbose=4
#mc.kernel()
#rdm1 = mc.make_rdm1()
#print(sum(numpy.diag(rdm1)))
#hcore = mf.get_hcore()
#hcore = mf.mo_coeff.T @ hcore @ mf.mo_coeff
#e1 = numpy.einsum('ij,ij',hcore, rdm1)
##e1 = numpy.trace( hcore @ rdm1)
#print(e1) 
#
#
#sys.exit()


# defines fragments
# for H chain, change frag_type="chain"
# be_type=be2/be3/be4
fobj = fragpart(1, be_type=be_type, frag_type='autogen', mol=mol,
                molecule=True,
                frozen_core=False)

# BE object
# integral transformation, schmidt decompistion, localizatyion, most initialization
mybe = pbe(mf, fobj, super_cell=True) #,lo_method='boys')

# solver = CCSD/MP2/FCI/SCI/VMC
# method = GD -> Gradient descent
# jac_solver = HF/MP2
# old E expression use_cumulant=False 
# 
# relax_density=True

mybe.optimize(conv_tol=1e-7,solver='CCSD',method='QN', nproc=1, ompnum=1)
rdm1, rdm2 = mybe.get_rdm(return_ao=False)
print(rdm1.shape)
Nelec = sum(numpy.diag(rdm1))
print(Nelec)
sys.exit()
