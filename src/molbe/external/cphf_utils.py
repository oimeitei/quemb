# Author(s): Hong-Zhou Ye
# NOTICE: The following code is entirely written by Hong-Zhou Ye.
#         The code has been slightly modified.
#
import numpy as np
from pyscf import ao2mo


def get_cphf_A(C, moe, eri, no):
    nao = C.shape[0]
    nv = nao - no
    Co = C[:, :no]
    Cv = C[:, no:]
    nov = no * nv
    Vovov = ao2mo.incore.general(eri, (Co, Cv, Co, Cv), compact=False).reshape(
        no, nv, no, nv
    )
    Voovv = ao2mo.incore.general(eri, (Co, Co, Cv, Cv), compact=False).reshape(
        no, no, nv, nv
    )

    # 4*Viajb - Vibja - Vjiab
    A = (
        4.0 * Vovov - Vovov.transpose(0, 3, 2, 1) - Voovv.transpose(0, 2, 1, 3)
    ).reshape(nov, nov)
    denom = (moe[:no].reshape(-1, 1) - moe[no:]).ravel()
    A -= np.diag(denom)

    return A


def get_cphf_rhs(C, no, v):
    nao = C.shape[0]
    nv = nao - no
    Co = C[:, :no]
    Cv = C[:, no:]

    return (Co.T @ v @ Cv).ravel()


def cphf_kernel(C, moe, eri, no, v):
    nao = C.shape[0]
    nv = nao - no

    # build RHS vector, B0
    B0 = get_cphf_rhs(C, no, v)

    # build A matrix
    A = get_cphf_A(C, moe, eri, no)

    # solve for u
    u = np.linalg.solve(A, B0)

    return u


def cphf_kernel_batch(C, moe, eri, no, vs):
    nao = C.shape[0]
    nv = nao - no
    nov = no * nv
    npot = len(vs)

    # build RHS vectors
    B0s = np.zeros([nov, npot])
    for i, v in enumerate(vs):
        B0s[:, i] = get_cphf_rhs(C, no, v)

    # build A matrix
    A = get_cphf_A(C, moe, eri, no)

    # solve for u
    us = np.linalg.solve(A, B0s).T

    return us


def get_rhf_dP_from_u(C, no, u):
    n = C.shape[0]
    nv = n - no
    dP = -C[:, :no] @ u.reshape(no, nv) @ C[:, no:].T
    dP += dP.T

    return dP