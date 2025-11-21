# Utility package to scale target values
import torch
import numpy as np

def boostinv(qx, pboost):
    """
    Invert a Lorentz boost of a four-vector qx by the boost defined by four-vector pboost.
    This function interprets both input four-vectors qx and pboost as 4-element sequences
    in the ordering (px, py, pz, E). It internally rearranges them to the
    (E, px, py, pz) ordering for computation, computes the inverse boost defined
    by pboost, and returns the boosted q four-vector in the original input ordering
    (px, py, pz, E).
    Parameters
    ----------
    qx : array-like, shape (4,)
        The four-vector to be transformed, given as (px, py, pz, E).
    pboost : array-like, shape (4,)
        The four-vector that defines the boost (usually the momentum/energy of
        the boost frame), given as (px, py, pz, E).
    Returns
    -------
    qprime : ndarray-like, shape (4,)
        The transformed four-vector in the same ordering as the inputs:
        (px', py', pz', E'). The energy component is stored at index 3.
    """
    # basic validation and output container
    if not (hasattr(qx, "__len__") and hasattr(pboost, "__len__")) or len(qx) != 4 or len(pboost) != 4:
        raise ValueError("qx and pboost must be length-4 sequences (px,py,pz,E)")

    qprime = torch.zeros_like(qx)

    rmboost = torch.sqrt(max([pboost[3]**2 - (pboost[0:3]**2).sum(), 0.0]))

    aux  = (qx[3]*pboost[3] - qx[0:3]@pboost[0:3]) / rmboost
    aaux = (aux + qx[3]) / (pboost[3] + rmboost)

    qprime[3] = aux
    qprime[0:3] = qx[0:3] - aaux * pboost[0:3]
    return qprime

def boost_into_four_lepton_cm_frame(features):
    momenta = features.reshape(4, -1)  # Assuming features contain x leptons with 4 momentum components each.
                                             # p1 = momenta[0], p2 = momenta[1], p3 = momenta[2], p4 = momenta[3].
    p_tot   = momenta.sum(dim=0)

    momenta_prime = torch.zeros_like(momenta)
    for i in range(4):
        momenta_prime[i] = boostinv(momenta[i], p_tot)

    # p_tot_prime = momenta_prime.sum(dim=0)
    return momenta_prime.reshape(-1)

def scale_target(x):
    # Utility function to scale target values
    return x * 1000  # Scale target by 1000

def find_scale_var_ratios(labels):
    bll = labels[:7]
    buu = labels[:14]
    ratios_uncorrelated = bll[:,None] / buu[None,:]
    labels_prime = torch.zeros_like(labels)
    labels_prime[0] = labels[0]/labels[7]
    labels_prime[1] = ratios_uncorrelated.min()
    labels_prime[2] = ratios_uncorrelated.max()
    for i in range(3,14):
        labels_prime[i] = 0.0
    return labels_prime

