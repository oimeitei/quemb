from pyscf import gto,scf,mp, cc
from pbe.pbe import pbe
from pbe.fragment import fragpart
from pbe.helper import *
from pbe import sgeom, printatom
import sys, h5py


mol = gto.M(atom='''
H 0. 0. 0.
H 0. 0. 1.
H 0. 0. 2. 
H 0. 0. 3.
H 0. 0. 4. 
H 0. 0. 5.
H 0. 0. 6.
H 0. 0. 7.
''',basis='sto-3g', charge=0)


mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.kernel()
Natom = 8 # only for Hchain
fobj = fragpart(Natom, frag_type='hchain_simple',mol=mol,
                be_type='be2', 
                frozen_core=False)

mybe = pbe(mf, fobj, super_cell=True)
mybe.optimize(solver='FCI',method='QN', nproc=1)

# Get 1-RDM in full basis (AO basis)
rdm1 = mybe.rdm1_fullbasis()

# test 1e- energy
h1 = mybe.hcore
e1 = numpy.einsum('ij,ij', h1,rdm1)
print('1-electron Energy: {:>12.7f} H'.format(e1))
