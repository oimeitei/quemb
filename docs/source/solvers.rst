Solver Routines
***************


Orbital Localization
====================

Molecular orbital localization
------------------------------

.. autofunction:: molbe.lo.localize

Crystalline orbital localization
--------------------------------

.. autofunction:: kbe.lo.localize

Density Matching Error
======================

.. autofunction:: molbe.solver.solve_error

Interface to Quantum Chemistry Methods
======================================

.. autofunction:: molbe.solver.solve_mp2

.. autofunction:: molbe.solver.solve_ccsd

.. autofunction:: molbe.helper.get_scfObj

Schmidt Decomposition
=====================

Molecular Schmidt decomposition
-------------------------------

.. autofunction:: molbe.solver.schmidt_decomposition

Periodic Schmidt decomposition
------------------------------

.. autofunction:: kbe.solver.schmidt_decomp_svd

Handling Hamiltonian
====================

.. autofunction:: molbe.helper.get_eri

.. autofunction:: molbe.helper.get_core


Build molecular HF potential
----------------------------

.. autofunction:: molbe.helper.get_veff

Build perioidic HF potential
----------------------------

.. autofunction:: kbe.helper.get_veff


Handling Energies
=================

.. autofunction:: molbe.helper.get_frag_energy

.. autofunction:: molbe.rdm.compute_energy_full

Handling Densities
==================

.. autofunction:: molbe.rdm.rdm1_fullbasis
