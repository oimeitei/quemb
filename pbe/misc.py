import numpy, os
from pyscf import gto, scf

def libint2pyscf(xyzfile, hcore, basis, hcore_skiprows=1):
    """Build a pyscf Mole and RHF object using the given xyz file and core Hamiltonian (in libint standard format)

    c.f.
    In libint standard format, the basis sets appear in the order
        atom#   n   l   m
        0       1   0   0   1s
                2   0   0   2s
                2   1   -1  2py
                2   1   0   2pz
                2   1   1   2px
                ...
        ...
    In pyscf, the basis sets appear in the order
        atom #  n   l   m
        0       1   0   0   1s
                2   0   0   2s
                2   1   1   2px
                2   1   -1  2py
                2   1   0   2pz
                ...
        ...
    For higher angular momentum, both use [-l, -l+1, ..., l-1, l] ordering.


    Parameters
    ----------
    xyzfile : string
        Path to the xyz file
    hcore : string
        Path to the core Hamiltonian
    basis : string
        Name of the basis set
    hcore_skiprows : int, optional
        # of first rows to skip from the core Hamiltonian file, by default 1

    Returns
    -------
    (pyscf.gto.Mole, pyscf.scf.RHF)
    """
    # Check input validity
    assert os.path.exists(xyzfile), "Input xyz file does not exist"
    assert os.path.exists(hcore), "Input core Hamiltonian file does not exist"
    
    mol = gto.M(atom=xyzfile, basis=basis)
    hcore_libint = numpy.loadtxt(hcore, skiprows=hcore_skiprows)

    libint2pyscf = []
    for labelidx, label in enumerate(mol.ao_labels()):
        # pyscf: px py pz // 1 -1 0
        # libint: py pz px // -1 0 1
        if 'p' not in label.split()[2]:
            libint2pyscf.append(labelidx)
        else:
            if 'x' in label.split()[2]: libint2pyscf.append(labelidx+2)
            elif 'y' in label.split()[2]: libint2pyscf.append(labelidx-1)
            elif 'z' in label.split()[2]: libint2pyscf.append(labelidx-1)

    hcore_pyscf = hcore_libint[numpy.ix_(libint2pyscf, libint2pyscf)]

    mol.incore_anyway = True
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: hcore_pyscf

    return mol, mf
