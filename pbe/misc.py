import numpy, os
from pyscf import gto, scf
import time

def libint2pyscf(
    xyzfile, hcore, basis, hcore_skiprows=1, use_df=False, unrestricted=False, spin=0, charge=0
):
    """Build a pyscf Mole and RHF/UHF object using the given xyz file
       and core Hamiltonian (in libint standard format)

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
    use_df : boolean, optional
        If true, use density-fitting to evaluate the two-electron integrals
    unrestricted : boolean, optional
        If true, use UHF bath
    spin : int, optional
        2S, Difference between the number of alpha and beta electrons
    charge : int, optional
        Total charge of the system

    Returns
    -------
    (pyscf.gto.Mole, pyscf.scf.RHF or pyscf.scf.UHF)
    """
    # Check input validity
    assert os.path.exists(xyzfile), "Input xyz file does not exist"
    assert os.path.exists(hcore), "Input core Hamiltonian file does not exist"

    mol = gto.M(atom=xyzfile, basis=basis, spin=spin, charge=charge)
    hcore_libint = numpy.loadtxt(hcore, skiprows=hcore_skiprows)

    libint2pyscf = []
    for labelidx, label in enumerate(mol.ao_labels()):
        # pyscf: px py pz // 1 -1 0
        # libint: py pz px // -1 0 1
        if "p" not in label.split()[2]:
            libint2pyscf.append(labelidx)
        else:
            if "x" in label.split()[2]:
                libint2pyscf.append(labelidx + 2)
            elif "y" in label.split()[2]:
                libint2pyscf.append(labelidx - 1)
            elif "z" in label.split()[2]:
                libint2pyscf.append(labelidx - 1)

    hcore_pyscf = hcore_libint[numpy.ix_(libint2pyscf, libint2pyscf)]

    mol.incore_anyway = True
    if use_df:
        mf = scf.UHF(mol).density_fit() if unrestricted else scf.RHF(mol).density_fit()
        from pyscf import df

        mydf = df.DF(mol).build()
        mf.with_df = mydf
    else: mf = scf.UHF(mol) if unrestricted else scf.RHF(mol)
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
        read = h5py.File(frag.eri_file, "r")
        eri = read[frag.dname][()]  # 2e in embedding basis
        read.close()
        eri = ao2mo.restore(1, eri, frag.nao)
        if basis == "embedding":
            h1e = frag.fock
            h2e = eri
        elif basis == "fragment_mo":
            frag.scf()  # make sure that we have mo coefficients
            h1e = numpy.einsum(
                "ij,ia,jb->ab", frag.fock, frag.mo_coeffs, frag.mo_coeffs, optimize=True
            )
            h2e = numpy.einsum(
                "ijkl,ia,jb,kc,ld->abcd",
                eri,
                frag.mo_coeffs,
                frag.mo_coeffs,
                frag.mo_coeffs,
                frag.mo_coeffs,
                optimize=True,
            )
        else:
            raise Exception("Basis should be either embedding or fragment_mo")

        fcidump.from_integrals(
            fcidump_prefix + "f" + str(fidx), h1e, h2e, frag.TA.shape[1], frag.nsocc, ms=0
        )


def ube2fcidump(be_obj, fcidump_prefix, basis):
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

    for fidx, frag in enumerate(be_obj.Fobjs_a):
        # Read in eri
        read = h5py.File(frag.eri_file, "r")
        eri = read[frag.dname][()]  # 2e in embedding basis
        read.close()
        eri = ao2mo.restore(1, eri, frag.nao)
        if basis == "embedding":
            h1e = frag.fock
            h2e = eri
        elif basis == "fragment_mo":
            frag.scf()  # make sure that we have mo coefficients
            h1e = numpy.einsum(
                "ij,ia,jb->ab", frag.fock, frag.mo_coeffs, frag.mo_coeffs, optimize=True
            )
            h2e = numpy.einsum(
                "ijkl,ia,jb,kc,ld->abcd",
                eri,
                frag.mo_coeffs,
                frag.mo_coeffs,
                frag.mo_coeffs,
                frag.mo_coeffs,
                optimize=True,
            )
        else:
            raise Exception("Basis should be either embedding or fragment_mo")

        fcidump.from_integrals(
            fcidump_prefix + "f" + str(fidx) + "a", h1e, h2e, frag.TA.shape[1], frag.nsocc, ms=0
        )

    for fidx, frag in enumerate(be_obj.Fobjs_b):
        # Read in eri
        read = h5py.File(frag.eri_file, "r")
        eri = read[frag.dname][()]  # 2e in embedding basis
        read.close()
        eri = ao2mo.restore(1, eri, frag.nao)
        if basis == "embedding":
            h1e = frag.fock
            h2e = eri
        elif basis == "fragment_mo":
            frag.scf()  # make sure that we have mo coefficients
            h1e = numpy.einsum(
                "ij,ia,jb->ab", frag.fock, frag.mo_coeffs, frag.mo_coeffs, optimize=True
            )
            h2e = numpy.einsum(
                "ijkl,ia,jb,kc,ld->abcd",
                eri,
                frag.mo_coeffs,
                frag.mo_coeffs,
                frag.mo_coeffs,
                frag.mo_coeffs,
                optimize=True,
            )
        else:
            raise Exception("Basis should be either embedding or fragment_mo")

        fcidump.from_integrals(
            fcidump_prefix + "f" + str(fidx) + "b", h1e, h2e, frag.TA.shape[1], frag.nsocc, ms=0
        )


def be2puffin(
    xyzfile,
    hcore,
    basis,
    jk=None,
    use_df=False,
    charge=0,
    nproc=1,
    ompnum=1,
    be_type='be1',
    frozen_core=True,
    df_aux_basis=None,
    localization_method='lowdin',
    localization_basis=None
):
    """Front-facing API bridge tailored for SCINE Puffin
    Returns the CCSD oneshot energies

    Parameters
    ----------
    xyzfile : string
        Path to the xyz file
    hcore : numpy.array
        Two-dimensional array of the core Hamiltonian
    basis : string
        Name of the basis set
    jk : numpy.array
        Coulomb and Exchange matrices (pyscf will calculate this if not given)
    use_df : boolean, optional
        If true, use density-fitting to evaluate the two-electron integrals
    charge : int, optional
        Total charge of the system
    nproc : int, optional
    ompnum: int, optional
        Set number of processors and ompnum for the jobs
    frozen_core: bool, optional
        Whether frozen core approximation is used or not, by default True
    """
    from .fragment import fragpart
    from .pbe import pbe

    # Check input validity
    assert os.path.exists(xyzfile), "Input xyz file does not exist"

    mol = gto.M(atom=xyzfile, basis=basis, charge=charge)
    """
    libint2pyscf = []
    for labelidx, label in enumerate(mol.ao_labels()):
        # pyscf: px py pz // 1 -1 0
        # libint: py pz px // -1 0 1
        if "p" not in label.split()[2]:
            libint2pyscf.append(labelidx)
        else:
            if "x" in label.split()[2]:
                libint2pyscf.append(labelidx + 2)
            elif "y" in label.split()[2]:
                libint2pyscf.append(labelidx - 1)
            elif "z" in label.split()[2]:
                libint2pyscf.append(labelidx - 1)

    hcore_pyscf = hcore[numpy.ix_(libint2pyscf, libint2pyscf)]
    if not jk is None:
        jk_pyscf = (
            jk[0][numpy.ix_(libint2pyscf, libint2pyscf, libint2pyscf, libint2pyscf)],
            jk[1][numpy.ix_(libint2pyscf, libint2pyscf, libint2pyscf, libint2pyscf)],
        )

    mol.incore_anyway = True
    
    if use_df and jk is None:
        from pyscf import df
<<<<<<< HEAD
        mf = scf.RHF(mol).density_fit(auxbasis=df_aux_basis)

    else: mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: hcore_pyscf
    if not jk is None: mf.get_jk = lambda *args: jk_pyscf
    """
    mol.incore_anyway = True
    time_pre_mf = time.time()
    mf = scf.RHF(mol)
    mf.kernel()
    #print("Using auxillary basis in density fitting: ", mf.with_df.auxmol.basis)
    #print("DF auxillary nao_nr", mf.with_df.auxmol.nao_nr())
=======
        print("df_aux_basis", df_aux_basis)
        #mf = scf.RHF(mol).density_fit()
        #mydf = df.DF(mol).build()
        mf = scf.RHF(mol).density_fit(auxbasis=df_aux_basis)
#        from pyscf import df
        #mydf = df.DF(mol).build()
        #mf.with_df = mydf
        
        #mf.with_df.auxbasis = df_aux_basis
        #print(mf.with_df.auxmol.basis)
        #print(mf.with_df.auxmol.nao_nr())
    else: mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: hcore_pyscf
    if not jk is None: mf.get_jk = lambda *args: jk_pyscf
    time_pre_mf = time.time()
    mf.kernel()
    print("Using auxillary basis in density fitting: ", mf.with_df.auxmol.basis, flush=True)
    print("DF auxillary nao_nr", mf.with_df.auxmol.nao_nr(), flush=True)
    time_post_mf = time.time()
    print("Time for mf kernel to run: ", time_post_mf - time_pre_mf, flush=True)
    fobj = fragpart(
        mol.natm, be_type=be_type, frag_type="autogen", mol=mol, molecule=True, 
        frozen_core=frozen_core, valence_basis=localization_basis
    )
    time_post_fragpart = time.time()
    print("Time for fragmentation to run: ", time_post_fragpart - time_post_mf, flush=True)
    mybe = pbe(mf, fobj, lo_method=localization_method)
    mybe.oneshot(solver="CCSD", nproc=nproc, ompnum=ompnum, calc_frag_energy=True)
    return mybe.ebe_tot
