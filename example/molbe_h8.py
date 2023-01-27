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
''',basis='6-31g', charge=0)


mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.kernel()
Natom = 8 # only for Hchain
fobj = fragpart(Natom, frag_type='chain',mol=mol,
                be_type='be2', 
                frozen_core=False)

mybe = pbe(mf, fobj, super_cell=True)
mybe.optimize(solver='FCI',method='QN', nproc=1)
rdm1, rdm2 = mybe.get_rdm(return_ao=False) # AO basis
# rdm1, rdm2 = mybe.get_rdm(return_ao=False) # MO basis

# Test energies
h1_mo = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
E1 = numpy.einsum('ij,ij', h1_mo, rdm1)
print(' E(h1)   : {:>12.8f} Ha'.format( E1))
eri_mo = ao2mo.kernel(mol, mf.mo_coeff)
eri_mo = ao2mo.restore(1,eri_mo, mf.mo_coeff.shape[1])
E2 = numpy.einsum('pqrs,pqrs', eri_mo, rdm2, optimize=True)
print(' E(V)    : {:>12.8f} Ha'.format(0.5*E2))
print(' Total E : {:>12.8f} Ha'.format(E1+0.5*E2+mf.energy_nuc()))
