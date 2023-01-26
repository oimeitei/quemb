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
rdm1, rdm2 = mybe.get_rdm() # AO basis
# rdm1, rdm2 = mybe.get_rdm(return_ao=False) # MO basis
