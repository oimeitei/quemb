# Author(s): Hong-Zhou Ye
# NOTICE: The following code is entirely written by Hong-Zhou Ye.
#         The code has been slightly modified.
#

import numpy as np
import scipy.linalg as slg
from pyscf import ao2mo, scf

""" RMP2 implementation
"""


def get_Diajb_r(moe, no):
    n = moe.size
    nv = n - no
    eo = moe[:no]
    ev = moe[no:]
    Dia = (eo.reshape(-1, 1) - ev).ravel()
    Diajb = (Dia.reshape(-1, 1) + Dia).reshape(no, nv, no, nv)

    return Diajb


def get_dF_r(no, V, C, Q, u):
    n = C.shape[0]
    nv = n - no
    Co = C[:, :no]
    Cv = C[:, no:]
    uov = u.reshape(no, nv)
    dP = -Co @ uov @ Cv.T
    dP += dP.T
    vj, vk = scf.hf.dot_eri_dm(V, dP * 2.0, hermi=1)
    dF = Q + vj - 0.5 * vk

    return dF


def get_dmoe_F_r(C, dF):
    de = np.einsum("pi,qi,pq->i", C, C, dF)

    return de


def get_full_u_F_r(no, C, moe, dF, u):
    n = C.shape[0]
    nv = n - no
    Co = C[:, :no]
    Cv = C[:, no:]
    eo = moe[:no]
    ev = moe[no:]
    uov = u.reshape(no, nv)
    Dij = -eo.reshape(-1, 1) + eo
    np.fill_diagonal(Dij, 1)
    dUoo = (Co.T @ dF @ Co) / Dij
    np.fill_diagonal(dUoo, 0.0)
    Dab = -ev.reshape(-1, 1) + ev
    np.fill_diagonal(Dab, 1)
    dUvv = (Cv.T @ dF @ Cv) / Dab
    np.fill_diagonal(dUvv, 0.0)

    return np.block([[dUoo, uov], [-uov.T, dUvv]])


def get_dVovov_r(no, V, C, u):
    n = C.shape[0]
    nv = n - no
    nov = no * nv
    Co = C[:, :no]
    Cv = C[:, no:]
    if u.size == no * nv:
        uov = u.reshape(no, nv)
        dCv = Co @ uov
        dCo = -Cv @ uov.T
    else:
        dC = C @ u
        dCo = dC[:, :no]
        dCv = dC[:, no:]
    Vovov_ = ao2mo.incore.general(V, (Co, Cv, Co, dCv)).reshape(nov, nov)
    Vovo_v = ao2mo.incore.general(V, (Co, Cv, dCo, Cv)).reshape(nov, nov)
    dVovov = (Vovov_ + Vovov_.T + Vovo_v + Vovo_v.T).reshape(no, nv, no, nv)

    return dVovov


def get_Pmp2_r(t2l, t2r):
    assert t2l.ndim == t2r.ndim == 4
    no = t2l.shape[0]
    Poo = -np.einsum(
        "iajb,majb->im", t2l, 2.0 * t2r - t2r.transpose(0, 3, 2, 1), optimize=True
    )
    Pvv = np.einsum(
        "iajb,icjb->ac", t2l, 2.0 * t2r - t2r.transpose(0, 3, 2, 1), optimize=True
    )

    return slg.block_diag(Poo, Pvv)


def get_dPmp2_batch_r(C, moe, V, no, Qs, aorep=True):
    """Derivative of oo and vv block of the MP2 density"""
    n = C.shape[0]
    nv = n - no
    Co = C[:, :no]
    Cv = C[:, no:]
    Vovov = ao2mo.incore.general(V, (Co, Cv, Co, Cv)).reshape(no, nv, no, nv)
    Diajb = get_Diajb_r(moe, no)
    t2 = Vovov / Diajb

    from molbe.external.cphf_utils import cphf_kernel_batch as cphf_kernel

    us = cphf_kernel(C, moe, V, no, Qs)
    nQ = len(Qs)
    dPs = [None] * nQ
    Phf = np.diag([1 if i < no else 0 for i in range(n)])
    iQ = 0
    for u, Q in zip(us, Qs):
        dF = get_dF_r(no, V, C, Q, u)
        dmoe = get_dmoe_F_r(C, dF)
        dDiajb = get_Diajb_r(dmoe, no)
        U = get_full_u_F_r(no, C, moe, dF, u)
        dVovov = get_dVovov_r(no, V, C, U)
        dt2 = (dVovov - t2 * dDiajb) / Diajb

        P = get_Pmp2_r(t2, t2)
        P += Phf
        dP = U @ P - P @ U

        dP2 = get_Pmp2_r(dt2, t2)
        dP2 += dP2.T

        dPs[iQ] = (dP + dP2) * 2.0

        iQ += 1

    if aorep:
        for iQ in range(nQ):
            dPs[iQ] = C @ dPs[iQ] @ C.T

    return dPs

