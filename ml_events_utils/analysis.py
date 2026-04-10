import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

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
    qx : array-like, shape (,4)
        The four-vector to be transformed, given as (px, py, pz, E).
    pboost : array-like, shape (,4)
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


def costhetastar(momenta):
    """
    Take the momenta of 4 leptons/particles, coming from 2 bosons.
    Boost them into the diboson CMS -> bv"i".
    Boost lepton momenta into CMS of the respective boson -> bbv"i".

    Parameters
    ----------
    momenta : torch.Tensor, shape (N, 4, 4)
        4 four-vectors of the particles as (px, py, pz, E)

    Returns
    -------
    ct1, ct2, ct3, ct4 : torch.Tensor
        Cosine of theta* angles for each particle
    """
    v1, v2, v3, v4 = momenta[...,0,:], momenta[...,1,:], momenta[...,2,:], momenta[...,3,:]
    v12 = v1 + v2    # Momentum of 1st Vector boson
    v34 = v3 + v4    # Momentum of 2nd Vector boson
    vv  = v12 + v34  # Diboson momentum

    # Boost into diboson cms
    bv12 = boostinv(v12, vv)
    bv34 = boostinv(v34, vv)
    bv1  = boostinv(v1, vv)
    bv2  = boostinv(v2, vv)
    bv3  = boostinv(v3, vv)
    bv4  = boostinv(v4, vv)

    # test = boostinv(vv, vv)

    # Boost into the restframe of the respective boson
    bbv1 = boostinv(bv1, bv12)
    bbv2 = boostinv(bv2, bv12)
    bbv3 = boostinv(bv3, bv34)
    bbv4 = boostinv(bv4, bv34)

    # Calculate cosine of angles (dot product of normalized 3-vectors)
    ct1 = (bbv1[...,0:3] * bv12[...,0:3]).sum(dim=-1) / (torch.norm(bbv1[...,0:3], dim=-1) * torch.norm(bv12[...,0:3], dim=-1))
    ct2 = (bbv2[...,0:3] * bv12[...,0:3]).sum(dim=-1) / (torch.norm(bbv2[...,0:3], dim=-1) * torch.norm(bv12[...,0:3], dim=-1))
    ct3 = (bbv3[...,0:3] * bv34[...,0:3]).sum(dim=-1) / (torch.norm(bbv3[...,0:3], dim=-1) * torch.norm(bv34[...,0:3], dim=-1))
    ct4 = (bbv4[...,0:3] * bv34[...,0:3]).sum(dim=-1) / (torch.norm(bbv4[...,0:3], dim=-1) * torch.norm(bv34[...,0:3], dim=-1))

    return ct1, ct2, ct3, ct4


def get_pt(p4):
    """
    Calculate the transverse momentum (pT) of a particle given its 4-momentum.

    Args:
        p4 (torch.Tensor): Tensor of shape (..., 4) representing the 4-momentum (px, py, pz, E)
    Returns:
        torch.Tensor: Tensor of shape (...) representing the transverse momentum
    """
    px = p4[..., 0]
    py = p4[..., 1]
    pt = torch.sqrt(px**2 + py**2)
    return pt


def get_phi(p4):
    """
    Calculate the azimuthal angle (phi) of a particle given its 4-momentum.

    Args:
        p4 (torch.Tensor): Tensor of shape (..., 4) representing the 4-momentum (px, py, pz, E)
    Returns:
        torch.Tensor: Tensor of shape (...) representing the azimuthal angle in radians
    """
    px = p4[..., 0]
    py = p4[..., 1]
    phi = torch.atan2(py, px)
    return phi


def get_rapidity(p4):
    """
    Calculate the rapidity of a particle given its 4-momentum.

    Args:
        p4 (torch.Tensor): Tensor of shape (..., 4) representing the 4-momentum (px, py, pz, E)
    Returns:
        torch.Tensor: Tensor of shape (...) representing the rapidity
    """
    pz = p4[..., 2]
    E  = p4[..., 3]

    tiny = 1e-10
    # Handle edge cases similar to the Fortran code
    # Handle edge cases where E ≈ ±pz
    rapidity = torch.where(
        torch.abs(E - pz) < tiny,
        torch.sign(pz) * 1e8,
        torch.where(
            torch.abs(E + pz) < 1e-5 * torch.abs(E - pz),  # mauro ? (e~-pz=> large negative rap ~log0)
            torch.sign(pz) * 1e8,
            0.5 * torch.log((E + pz) / (E - pz))  # Ordinary definition
        )
    )

    return rapidity


def getdphi(p1, p2):
    """
    Calculate the delta phi between two particles given their 4-momenta.

    Args:
        p1 (torch.Tensor): Tensor of shape (..., 4) representing the 4-momentum of the first particle
        p2 (torch.Tensor): Tensor of shape (..., 4) representing the 4-momentum of the second particle
    Returns:
        torch.Tensor: Tensor of shape (...) representing the delta phi in radians, in the range [0, pi]
    """
    phi1 = get_phi(p1)
    phi2 = get_phi(p2)

    getdphi = torch.abs(phi1 - phi2)
    getdphi = torch.min(getdphi, 2*torch.pi - getdphi)

    return getdphi * 180. / torch.pi  # Convert to degrees if desired, or return in radians by removing this factor


# Z jet:
def cosmujet(momenta):
    """
    Take the momenta of 2 leptons coming from 1 boson and 1 jet.
    Boost them into the Z+jet CMS -> bv"i".

    Boost lepton momenta into CMS of the respective boson -> bbv"i".

    Parameters
    ----------
    momenta : torch.Tensor, shape (N, 3, 4)
        3 four-vectors of the particles as (px, py, pz, E)

    Returns
    -------
    ct1, ct2 : torch.Tensor
        Cosine of theta* angles for lepton with respect to the jet.
    """
    l1, l2, jet = momenta[...,0,:], momenta[...,1,:], momenta[...,2,:]
    # v12 = l1 + l2    # Momentum of 1st Vector boson
    # vj  = v12 + jet  # Boson + jet momentum

    # Calculate cosine of angles (dot product of normalized 3-vectors)
    ct1 = (l1[...,0:3] * jet[...,0:3]).sum(dim=-1) / (torch.norm(l1[...,0:3], dim=-1) * torch.norm(jet[...,0:3], dim=-1))
    ct2 = (l2[...,0:3] * jet[...,0:3]).sum(dim=-1) / (torch.norm(l2[...,0:3], dim=-1) * torch.norm(jet[...,0:3], dim=-1))

    return ct1, ct2