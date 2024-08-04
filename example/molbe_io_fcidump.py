# Illustrates how fcidump file containing fragment hamiltonian
# can be generated using be2fcidump

from pyscf import gto, scf
from molbe import BE
from molbe import fragpart
from molbe.misc import *
import pbe_var
pbe_var.PRINT_LEVEL=3

# Read in molecular integrals expressed in libint basis ordering
# numpy.loadtxt takes care of the input under the hood
mol, mf = libint2pyscf("h6.xyz", "hcore.dat", "sto-3g", hcore_skiprows=1)
mf.kernel()

# Construct fragments for BE
fobj = fragpart(be_type='be2', mol=mol)
hchain_be = pbe(mf, fobj)

# Write out fcidump file for each fragment
be2fcidump(hchain_be, "hchain", "fragment_mo")
