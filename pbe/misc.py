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

def be2fcidump(be_obj, fcidump_prefix, basis):
    """Construct FCIDUMP file for each fragment in a given BE object
    * Assumes molecular, restricted BE calculation

    Parameters
    ----------
    be_obj : pbe.pbe.pbe
        BE object
    fcidump_prefix : string
        Prefix for path & filename to the output fcidump files
        Each file is named [fcidump_prefix]_f0, ...
    basis : string
        'embedding' to get the integrals in the embedding basis
        'fragment_mo' to get the integrals in the fragment MO basis
    """
    from pyscf import ao2mo
    from pyscf.tools import fcidump
    import h5py

    for fidx, frag in enumerate(be_obj.Fobjs):
        # Read in eri
        read = h5py.File(frag.eri_file, 'r')
        eri = read[frag.dname][()] # 2e in embedding basis
        read.close()
        eri = ao2mo.restore(1, eri, frag.nao)
        if basis == 'embedding':
            h1e = frag.fock
            h2e = eri
        elif basis == 'fragment_mo':
            frag.scf() # make sure that we have mo coefficients
            h1e = numpy.einsum("ij,ia,jb->ab", frag.fock, frag.mo_coeffs, frag.mo_coeffs)
            h2e = numpy.einsum("ijkl,ia,jb,kc,ld->abcd", eri, frag.mo_coeffs, frag.mo_coeffs, frag.mo_coeffs, frag.mo_coeffs)
        else: raise Exception("Basis should be either embedding or fragment_mo")

        fcidump.from_integrals(fcidump_prefix + 'f' + str(fidx), h1e, h2e, frag.TA.shape[1], frag.nsocc, ms=0)

