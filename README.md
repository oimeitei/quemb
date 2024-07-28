QuEmb
=====
QuEmb is a python library to perform bootstrap embedding calculations on large molecular and periodic systems.

**Reference**
OR Meitei, T Van Voorhis, Periodic bootstrap embedding, [JCTC 19 3123 2023](https://doi.org/10.1021/acs.jctc.3c00069)  
OR Meitei, T Van Voorhis, Electron correlation in 2D periodic systems, [arXiv:2308.06185](https://arxiv.org/abs/2308.06185)  
HZ Ye, HK Tran, T Van Voorhis, Bootstrap embedding for large molecular systems, [JCTC 16 5035 2020](https://doi.org/10.1021/acs.jctc.0c00438)

## Installation
Add path/to/pbe to PYTHONPATH 

### Conda
For conda (or virtual environment) installations, after creating your environment, specify the path to mol-be source as a path file, as in:
```bash
echo path/to/pbe > $(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")/pbe.pth
```

**Contact** Oinam Meitei oimeitei@mit.edu
