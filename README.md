# QuEmb

QuEmb is a robust framework for performing Bootstrap Embedded (BE) quantum chemistry, geared towards
efficiently handling electron correlation in molecules, surfaces, and solids. The molecular and periodic BE implementations in this repository are written in Python to allow for easy integration with external codes. QuEmb relies extensively on the [PySCF](https://github.com/pyscf/pyscf) library for standard quantum chemistry functions and utilizes Python's
multiprocessing module to enable parallel computations in high-performance computing environments.

QuEmb includes two libraries: `molbe` and `kbe`.
The `molbe` library implements BE for molecules and supramolecular complexes,
while the `kbe` library is designed to handle periodic systems such as extended surfaces and solids.


## Features

- **Fragment-based quantum embedding:** Automated and versatile fragmentation scheme with overlapping regions, which improves accuracy and allows for a truly general input system.
- **Periodic Bootstrap Embedding:** Extends molecular BE to treat periodic systems (1D & 2D systems)
using reciprocal space sums. 
- **High accuracy and efficiency:** Capable of recovering ~99.9% of electron correlation energy.
- **Parallel computing:** We employ Python's multiprocessing module to perform parallel computations across multiple processors.

## Installation

### Prerequisites

- Python 3.6 or higher
- PySCF library
- Numpy
- Scipy
- libDMET<sup>##</sup> (required for periodic BE)
- [Wannier90](https://github.com/wannier-developers/wannier90)<sup>&&</sup> (to use Wannier functions)
- [dmrgscf]( https://github.com/pyscf/dmrgscf)<sup>&&</sup> (for dmrg functionality with block2)

<sub><sup>##</sup>The modified version of [libDMET](https://github.com/gkclab/libdmet_preview) available
at [here](https://github.com/oimeitei/libdmet_preview) is
recommended to run periodic BE using QuEmb.  
<sup>&&</sup>Wannier90 code is interfaced via [libDMET](https://github.com/gkclab/libdmet_preview) in QuEmb</sub>

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/oimeitei/quemb.git
   cd quemb
   
2. Install QuEmb using one of the following approaches:
    ```bash
    pip install .  
    ```
    or simply add `path/to/quemd` to `PYTHONPATH`
    ```bash
    export PYTHONPATH=/path/to/quemb:$PYTHONPATH
    ```
    
    For conda (or virtual environment) installations, after creating your environment, specify the path to mol-be source as a path file, as in:
    ```bash
    echo path/to/quemb > $(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")/quemb.pth
    ```


## Basic Usage

```bash
# Molecular
from molbe import fragpart
from molbe import BE

# Periodic
#from kbe import fragpart
#from kbe import BE

# Perform pyscf HF/KHF calculations
# get mol: pyscf.gto.M or pyscf.pbc.gto.Cell
# get mf: pyscf.scf.RHF or pyscf.pbc.KRHF

# Define fragments
myFrag = fragpart(be_type='be2', mol=mol)

# Initialize BE
mybe = BE(mf, myFrag)

# Perform density matching in BE
mybe.optimize(solver='CCSD')
```
See documentation and `quemb/example` for more details.

## Documentation

Comprehensive documentation for QuEmb is available at `quemb/docs`. The documentation provides detailed infomation on installation, usage, API reference, and examples. To build the documentation locally, simply navigate to `docs` and build using `make html` or `make latexpdf`.

Alternatively, you can view the latest documentation online [here](https://quemb.readthedocs.io/).

## References

# For this library (QuEmb):
- [INSERT SOFTWARE PAPER HERE]

# For general BE theory:
- Welborn, M., Tsuchimochi, T. & Van Voorhis, T. Bootstrap embedding: An internally consistent fragment-based method. The Journal of Chemical Physics 145, 074102 (2016). ()
- Ye, H.-Z., Ricke, N. D., Tran, H. K. & Van Voorhis, T. Bootstrap Embedding for Molecules. J. Chem. Theory Comput. 15, 4497–4506 (2019).
- Ye, H.-Z., Tran, H. K. & Van Voorhis, T. Bootstrap Embedding For Large Molecular Systems. J. Chem. Theory Comput. 16, 5035–5046 (2020).
- Ye, H.-Z. & Van Voorhis, T. Atom-Based Bootstrap Embedding For Molecules. J. Phys. Chem. Lett. 10, 6368–6374 (2019).

# For periodic BE theory and implementation: 
- OR Meitei, T Van Voorhis, Periodic bootstrap embedding, [JCTC 19 3123 2023](https://doi.org/10.1021/acs.jctc.3c00069)  
- OR Meitei, T Van Voorhis, Electron correlation in 2D periodic systems, [arXiv:2308.06185](https://arxiv.org/abs/2308.06185)  

# For BE-DMRG theory and implementation:
- [INSERT BE-DMRG PAPER HERE]
- 
# For Unrestricted BE theory and implementation:
- Tran, H. K., Ye, H.-Z. & Van Voorhis, T. Bootstrap embedding with an unrestricted mean-field bath. The Journal of Chemical Physics 153, 214101 (2020).
- [INSERT LEAH PAPER HERE]

# For QuEmb dependencies:
- *pyscf* (https://pubs.aip.org/aip/jcp/article/153/2/024109/1061482/Recent-developments-in-the-PySCF-program-package)
- *block2* (https://pubs.aip.org/aip/jcp/article-abstract/159/23/234801/2930207/Block2-A-comprehensive-open-source-framework-to?redirectedFrom=fulltext)
- *libcint* (https://onlinelibrary.wiley.com/doi/abs/10.1002/jcc.23981)

