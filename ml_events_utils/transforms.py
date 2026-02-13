# Utility package to scale target values
import torch
import numpy as np
from .analysis import costhetastar, get_pt, get_rapidity, get_phi


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
    if not (hasattr(qx, "shape") and hasattr(pboost, "shape")) or qx.shape[-1] != 4 or pboost.shape[-1] != 4:
        raise ValueError("qx and pboost must have last dimension of size 4 (px,py,pz,E)")

    qprime = torch.zeros_like(qx)

    rmboost = torch.sqrt(torch.clamp(pboost[...,3]**2 - (pboost[...,0:3]**2).sum(dim=-1), min=0.0))

    aux  = (qx[...,3]*pboost[...,3] - (qx[...,0:3] * pboost[...,0:3]).sum(dim=-1)) / rmboost
    aaux = (aux + qx[...,3]) / (pboost[...,3] + rmboost)

    qprime[...,3] = aux
    qprime[...,0:3] = qx[...,0:3] - aaux.unsqueeze(-1) * pboost[...,0:3]
    return qprime

def boost_into_four_lepton_cm_frame(features):
    """
    Boost the four leptons into their combined center-of-mass frame.
    Parameters:
    features : torch.Tensor, shape (..., 16)
        Input features containing four leptons' four-momenta in the order:
        (px1, py1, pz1, E1, px2, py2, pz2, E2, px3, py3, pz3, E3, px4, py4, pz4, E4).
    Returns:
    torch.Tensor, shape (..., 16)
        The boosted four leptons' four-momenta in the same order as the input.
    """
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
    bll = labels[...,  : 7]
    buu = labels[..., 7:14]
    ratios_uncorrelated = bll.unsqueeze(-1) / buu.unsqueeze(0)

    new_shape    = labels.shape[:-1] + (3,)
    labels_prime = labels.new_zeros(*new_shape)

    labels_prime[..., 0] = labels[..., 0] / torch.clamp(labels[..., 7], min=1e-8)
    labels_prime[..., 1] = ratios_uncorrelated.min()
    labels_prime[..., 2] = ratios_uncorrelated.max()

    return labels_prime

# Define logarithmic target transform to reduce outlier impact
def log_target_transform(target):
    """Apply log transformation to target values to reduce outlier impact"""
    # Convert to torch tensor if it's not already
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target, dtype=torch.float32)

    # Add small epsilon to handle zero values and ensure positive input to log
    epsilon = 1e-10
    # Use log1p for better numerical stability: log(1 + x)
    return torch.log1p(torch.clamp(target, min=epsilon))

def boost_into_Zjet_cm_frame(features):
    """
    Boost the Z boson and jet into their combined center-of-mass frame, i.e. where the Z boson and the jet are back to back.
    Parameters:
    features : torch.Tensor, shape (..., 12)
        Input features containing Z boson and jet four-momenta in the order:
        (px_Z, py_Z, pz_Z, E_Z, px_jet, py_jet, pz_jet, E_jet, ...).
    Returns:
    torch.Tensor, shape (..., 12)
        The boosted Z boson and jet four-momenta in the same order as the input.
    """
    momenta = features.reshape(3, -1)  # Assuming features contain lepton momenta (2) and jet with 4 momentum components each.
                                       # p_l1 = momenta[0], p_l2 = momenta[1], p_jet = momenta[2].
    p_tot   = momenta.sum(dim=0)

    momenta_prime = torch.zeros_like(momenta)
    for i in range(3):
        momenta_prime[i] = boostinv(momenta[i], p_tot)

    # p_tot_prime = momenta_prime.sum(dim=0)
    return momenta_prime.reshape(-1)


def januar2026_input_choice(features):
    """
    Transform four leptons' four-momenta according to correspond to the following input choice:
    .. math::
        p_{T, Z_{1}}, \ y_{Z_{1}}, \ p_{T, Z_{2}}, \ y_{Z_{2}}, \ \cos(\theta^{*}_{e^{+}}), \ \cos(\theta^{*}_{\mu^{+}})
    where :math:`Z_{1}` and :math:`Z_{2}` are the two Z bosons formed by the four leptons :math:`l_{1} + l_{2}`
    and :math:`l_{3} + l_{4}` respectively.

    :param features: Input features containing four leptons' four-momenta in the order:
                    (px1, py1, pz1, E1, px2, py2, pz2, E2, px3, py3, pz3, E3, px4, py4, pz4, E4)
    :type features: torch.Tensor
    :param shape: (..., 16)

    :returns: The boosted four leptons' four-momenta in the same order as the input
    :rtype: torch.Tensor
    :return shape: (..., 16)
    """
    momenta = features.reshape(4, -1)  # Assuming features contain x leptons with 4 momentum components each.
                                       # p1 = momenta[0], p2 = momenta[1], p3 = momenta[2], p4 = momenta[3].

    V1 = momenta[...,0,:] + momenta[...,1,:]
    V2 = momenta[...,2,:] + momenta[...,3,:]

    pT_V1 = get_pt(V1)
    y_V1  = get_rapidity(V1)
    pT_V2 = get_pt(V2)
    y_V2  = get_rapidity(V2)
    cthep, _, cthmup, _ = torch.stack(costhetastar(momenta), dim=-1)

    return torch.stack([pT_V1, y_V1, pT_V2, y_V2, cthep, cthmup], dim=-1)
