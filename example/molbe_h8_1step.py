from pyscf import gto,scf, fci
from pbe.pbe import pbe
from pbe.fragment import fragpart
import pbe_var

pbe_var.SCRATCH = '/scratch' # eri files will be written here, alternately set SCRACT = '/scratch' in pbe_var.py

# PySCF HF
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

# BE1
fobj = fragpart(be_type='be1', mol=mol)
mybe = pbe(mf, fobj) 
mybe.oneshot(solver='FCI', nproc=1, ompnum=1)

