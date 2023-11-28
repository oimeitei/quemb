from pyscf import gto,scf,mp, cc, fci, lo
from pbe.pbe import pbe
from pbe.fragment import fragpart
from pbe.helper import *
from pbe import sgeom, printatom
import sys, h5py, os
from pyscf.lo import iao

be_type = 'be3' #sys.argv[1]


# PYSCF for integrals and HF solution
mol = gto.M(atom='''
C          2.3960832236        0.5633010778        0.5062390236
 C          2.3269091982        0.1460579295       -0.9440715406
 O          3.1929248317        0.3064173955       -1.7643243453
 O          1.1692487722       -0.5278210907       -1.3431820857
 C          0.1154360908       -0.6695049445       -0.4594230529
 C         -0.0342215211       -1.9016686927        0.1972679251
 C         -1.1424277610       -2.1362488556        1.0242961547
 C         -2.1217613910       -1.1398925239        1.1821460421
 C         -1.9903067931        0.0806775983        0.5054595564
 C         -0.8697854672        0.3335892609       -0.3139328128
 C         -0.7233066252        1.6520395638       -1.0009270752
 O          0.3318952234        2.1720637367       -1.3170474220
 O         -1.9265690783        2.2404820789       -1.2177425257
 H          3.3920584639        0.9881603924        0.6876546981
 H          2.2249688337       -0.2958876065        1.1736039271
 H          1.6272085042        1.3212290967        0.7136029829
 H          0.7280607166       -2.6698761740        0.0339246821
 H         -1.2439950544       -3.0983029882        1.5369372811
 H         -2.9920862798       -1.3181601317        1.8211691787
 H         -2.7534930385        0.8565055717        0.6066989055
 H         -1.7068408487        3.0970393057       -1.6282494973
''',basis='6-31g', charge=0)

mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.kernel()

#mc = cc.CCSD(mf)
#mc.verbose=4
#mc.kernel()

fobj = fragpart(1, be_type=be_type, frag_type='autogen', mol=mol,
                molecule=True, valence_only =True,valence_basis='sto-3g',
                frozen_core=False)  

mybe = pbe(mf, fobj, super_cell=True, lo_method='iao')
mybe.optimize(solver='CCSD',method='QN', nproc=1, ompnum=1, relax_density=True, conv_tol=1.e-4)
rdm1, rdm2, rdm1_lo, rdm2_lo = mybe.rdm1_fullbasis(return_ao=False, return_lo=True)
sys.exit()
# Get local orbitals
C_lo = mybe.Ciao_pao.copy()

# Get RDMs
# The active space (IAO) RDMs are rdm1_lo, rdm2_lo
rdm1, rdm2, rdm1_lo, rdm2_lo = mybe.rdm1_fullbasis(return_ao=False, return_lo=True)

nlo = rdm1_lo.shape[1]
h1_lo = C_lo[:,:nlo].T @ mf.get_hcore() @ C_lo[:,:nlo]
E1 = numpy.trace(h1_lo @ rdm1_lo)

print(E1)
