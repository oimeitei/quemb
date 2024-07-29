# NOTICE: The following code is entirely written by Hong-Zhou Ye.
#         The code has been slightly modified.
#

import numpy,functools,sys, time,os

def make_rdm1_ccsd_t1(t1):
    nocc, nvir = t1.shape
    nmo = nocc + nvir
    dm = numpy.zeros((nmo,nmo), dtype=t1.dtype)
    dm[:nocc,nocc:] = t1
    dm[nocc:,:nocc] = t1.T
    dm[numpy.diag_indices(nocc)] += 2.

    return dm

def make_rdm2_urlx(t1, t2, with_dm1=True):
    nocc, nvir = t1.shape
    nmo = nocc + nvir

    goovv = (numpy.einsum("ia,jb->ijab", t1, t1) + t2) * 0.5
    dovov = goovv.transpose(0,2,1,3) * 2 - goovv.transpose(1,2,0,3)

    dm2 = numpy.zeros([nmo,nmo,nmo,nmo], dtype=t1.dtype)

    dovov = numpy.asarray(dovov)
    dm2[:nocc,nocc:,:nocc,nocc:] = dovov
    dm2[:nocc,nocc:,:nocc,nocc:]+= dovov.transpose(2,3,0,1)
    dm2[nocc:,:nocc,nocc:,:nocc] = dm2[:nocc,nocc:,:nocc,nocc:].transpose(1,0,3,2).conj()
    dovov = None

    if with_dm1:
        dm1 = make_rdm1_ccsd_t1(t1)
        dm1[numpy.diag_indices(nocc)] -= 2
        
        for i in range(nocc):
            dm2[i,i,:,:] += dm1 * 2
            dm2[:,:,i,i] += dm1 * 2
            dm2[:,i,i,:] -= dm1
            dm2[i,:,:,i] -= dm1.T
        
        for i in range(nocc):
            for j in range(nocc):
                dm2[i,i,j,j] += 4
                dm2[i,j,j,i] -= 2

    return dm2  
