from pyscf import gto,scf, fci
from pbe.pbe import pbe
from pbe.fragment import fragpart

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

# PySCF FCI
mc = fci.FCI(mf)
fci_ecorr = mc.kernel()[0] - mf.e_tot
print(f'*** FCI Correlation Energy: {fci_ecorr:>14.8f} Ha', flush=True)

# BE1
fobj = fragpart(be_type='be1', mol=mol)
mybe = pbe(mf, fobj) 
mybe.optimize(solver='FCI', nproc=1, ompnum=1, only_chem=True) 

be_ecorr = mybe.ebe_tot - mybe.ebe_hf
err_ = (fci_ecorr - be_ecorr)*100./fci_ecorr
print(f'*** BE1 Correlation Energy Error (%) : {err_:>8.4f} %')

# BE2
fobj = fragpart(be_type='be2', mol=mol)
mybe = pbe(mf, fobj) 
mybe.optimize(solver='FCI', nproc=1, ompnum=1, only_chem=True) 

be_ecorr = mybe.ebe_tot - mybe.ebe_hf
err_ = (fci_ecorr - be_ecorr)*100./fci_ecorr
print(f'*** BE2 Correlation Energy Error (%) : {err_:>8.4f} %')

#BE3
fobj = fragpart(be_type='be3', mol=mol)
mybe = pbe(mf, fobj) 
mybe.optimize(solver='FCI', nproc=1, ompnum=1, only_chem=True) 

be_ecorr = mybe.ebe_tot - mybe.ebe_hf
err_ = (fci_ecorr - be_ecorr)*100./fci_ecorr
print(f'*** BE3 Correlation Energy Error (%) : {err_:>8.4f} %')

