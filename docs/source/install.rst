Installation
************

Prerequisites
-------------

 * Python 3.6 or higher
 * PySCF library
 * Numpy
 * Scipy
 * libDMET :sup:`##` (required for periodic BE)
 * `Wannier90 <https://github.com/wannier-developers/wannier90>`_ :sup:`&&` (to use Wannier functions)

| :sup:`##` The modified version of `libDMET <https://github.com/gkclab/libdmet_preview>`_ available at `here <https://github.com/oimeitei/libdmet_preview>`_ is recommended to run periodic BE using QuEmb.
| :sup:`&&` Wannier90 code is interfaced via `libDMET <https://github.com/gkclab/libdmet_preview>`_ in QuEmb


Obtain the source code
----------------------
Clone the Github repository::

  git clone https://github.com/oimeitei/quemb.git

pip install
-----------

::
   
  pip install .

Add to ``PYTHONPATH``
---------------------
Simply add ``path/to/quemb`` to ``PYTHONPATH``
::
   
   export PYTHONPATH=/path/to/quemb:$PYTHONPATH

Conda or virtual environment
----------------------------
For conda (or virtual environment) installations, after creating your environment, specify the path to mol-be source as a path file, as in::
  
  echo path/to/quemb > $(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")/quemb.pth
