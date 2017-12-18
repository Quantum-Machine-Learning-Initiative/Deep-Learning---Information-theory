# -*- coding: utf-8 -*-
# Author: Ville Bergholm 2011-2014
r"""
Harmonic oscillators (:mod:`qit.ho`)
====================================

.. currentmodule:: qit.ho

This module simulates harmonic oscillators by truncating the state
space dimension to a finite value. Higher truncation limits give more accurate results.
All the functions in this module operate in the truncated number basis
:math:`\{|0\rangle, |1\rangle, ..., |n-1\rangle\}`
of the harmonic oscillator, where n is the truncation dimension.

The corresponding truncated annihilation operator can be obtained with :func:`qit.utils.boson_ladder`.


Contents
--------

.. autosummary::

   coherent_state
   position_state
   momentum_state
   position
   momentum
   displace
   squeeze
   husimi
   wigner
"""

from __future__ import division, absolute_import, print_function, unicode_literals

from numpy import array, mat, empty, arange, diag, sqrt, ones, prod, sqrt, pi, isscalar, linspace, newaxis
from scipy.misc import factorial
from scipy.linalg import expm, norm

from .base import tol
from .state import state
from .utils import boson_ladder, comm

__all__ = ['coherent_state', 'displace', 'squeeze', 'position', 'momentum',
           'position_state', 'momentum_state', 'husimi', 'wigner']


# default truncation limit for number states
default_n = 30

def coherent_state(alpha, n=default_n):
    r"""Coherent states of a harmonic oscillator.

    Returns the n-dimensional approximation to the
    coherent state :math:`|\alpha\rangle`,

    .. math::

       |\alpha\rangle := D(\alpha) |0\rangle
       = e^{-\frac{|\alpha|^2}{2}} \sum_{k=0}^\infty \frac{\alpha^k}{\sqrt{k!}} |k\rangle,

    in the number basis. :math:`a|\alpha\rangle = \alpha |\alpha\rangle`.
    """
    # Ville Bergholm 2010

    k = arange(n)
    ket = (alpha ** k) / sqrt(factorial(k))
    return state(ket, n).normalize()
    #s = state(0, n).u_propagate(expm(alpha * mat(boson_ladder(n)).H))
    #s = state(0, n).u_propagate(displace(alpha, n))
    #s *= exp(-abs(alpha) ** 2 / 2) # normalization


def displace(alpha, n=default_n):
    r"""Bosonic displacement operator.

    Returns the n-dimensional approximation for the bosonic
    displacement operator

    .. math::

       D(\alpha) := \exp\left(\alpha a^\dagger - \alpha^* a\right)
       = \exp\left( i \sqrt{2} \left(Q \mathrm{Im}(\alpha) -P \mathrm{Re}(\alpha)\right)\right)

    in the number basis. This yields

    .. math::

       D(\alpha) Q D^\dagger(\alpha) &= Q -\sqrt{2} \textrm{Re}(\alpha) \mathbb{I},\\
       D(\alpha) P D^\dagger(\alpha) &= P -\sqrt{2} \textrm{Im}(\alpha) \mathbb{I},

    and thus the displacement operator displaces the state of a harmonic oscillator in phase space.
    """
    # Ville Bergholm 2010

    if not isscalar(alpha):
        raise TypeError('alpha must be a scalar.')

    a = mat(boson_ladder(n))
    return array(expm(alpha * a.H -alpha.conjugate() * a))


def squeeze(z, n=default_n):
    r"""Bosonic squeezing operator.

    Returns the n-dimensional approximation for the bosonic
    squeezing operator

    .. math::

       S(z) := \exp\left(\frac{1}{2} (z^* a^2 - z a^{\dagger 2})\right)
       = \exp\left(\frac{i}{2} \left((QP+PQ)\mathrm{Re}(z) +(P^2-Q^2)\mathrm{Im}(z)\right)\right)

    in the number basis.
    """
    # Ville Bergholm 2010
    if not isscalar(z):
        raise TypeError('z must be a scalar.')

    a = mat(boson_ladder(n))
    return array(expm(0.5 * (z.conjugate() * (a ** 2) - z * (a.H ** 2))))


def position(n=default_n):
    r"""Position operator.

    Returns the n-dimensional approximation of the
    dimensionless position operator Q in the number basis.

    .. math::

       Q &= \sqrt{\frac{m \omega}{\hbar}}   q =    (a+a^\dagger) / \sqrt{2},\\
       P &= \sqrt{\frac{1}{m \hbar \omega}} p = -i (a-a^\dagger) / \sqrt{2}.

    (Equivalently, :math:`a = (Q + iP) / \sqrt{2}`).
    These operators fulfill :math:`[q, p] = i \hbar, \quad  [Q, P] = i`.
    The Hamiltonian of the harmonic oscillator is

    .. math::

       H = \frac{p^2}{2m} +\frac{1}{2} m \omega^2 q^2
         = \frac{1}{2} \hbar \omega (P^2 +Q^2)
         = \hbar \omega (a^\dagger a +\frac{1}{2}).
    """
    # Ville Bergholm 2010

    a = mat(boson_ladder(n))
    return array(a + a.H) / sqrt(2)


def momentum(n=default_n):
    """Momentum operator.

    Returns the n-dimensional approximation of the
    dimensionless momentum operator P in the number basis.

    See :func:`position`.
    """
    # Ville Bergholm 2010

    a = mat(boson_ladder(n))
    return -1j*array(a - a.H) / sqrt(2)


def position_state(q, n=default_n):
    r"""Position eigenstates of a harmonic oscillator.

    Returns the n-dimensional approximation of the eigenstate :math:`|q\rangle`
    of the dimensionless position operator Q in the number basis.

    See :func:`position`, :func:`momentum`.

    Difference equation:

    .. math::

       r_1 &= \sqrt{2} q r_0,\\
       \sqrt{k+1} r_{k+1} &= \sqrt{2} q r_k -\sqrt{k} r_{k-1}, \qquad \text{when} \quad k >= 1.
    """
    # Ville Bergholm 2010

    ket = empty(n, dtype=complex)
    temp = sqrt(2) * q
    ket[0] = 1  # arbitrary nonzero initial value r_0
    ket[1] = temp * ket[0]
    for k in range(2, n):
        ket[k] = temp/sqrt(k) * ket[k - 1] -sqrt((k-1) / k) * ket[k - 2]
    ket /= norm(ket)  # normalize
    return state(ket, n)


def momentum_state(p, n=default_n):
    r"""Momentum eigenstates of a harmonic oscillator.

    Returns the n-dimensional approximation of the eigenstate :math:`|p\rangle`
    of the dimensionless momentum operator P in the number basis.

    See :func:`position`, :func:`momentum`.

    Difference equation:

    .. math::

       r_1 &= i \sqrt{2} p r_0,\\
       \sqrt{k+1} r_{k+1} &= i \sqrt{2} p r_k +\sqrt{k} r_{k-1}, \qquad \text{when} \quad k >= 1.
    """
    # Ville Bergholm 2010

    ket = empty(n, dtype=complex)
    temp = 1j * sqrt(2) * p
    ket[0] = 1  # arbitrary nonzero initial value r_0
    ket[1] = temp * ket[0]
    for k in range(2, n):
        ket[k] = temp/sqrt(k) * ket[k - 1] +sqrt((k-1) / k) * ket[k - 2]
    ket /= norm(ket)  # normalize
    return state(ket, n)


def husimi(s, alpha=None, z=0, res=(40, 40), lim=(-2, 2, -2, 2)):
    r"""Husimi probability distribution.
    ::

      H       = husimi(s, alpha[, z=])
      H, a, b = husimi(s, res=xxx, lim=yyy[, z=])

    Returns the Husimi probability distribution
    :math:`H(\mathrm{Im} \alpha, \mathrm{Re} \alpha)` corresponding to the harmonic
    oscillator state s given in the number basis:

    .. math::

       H(s, \alpha, z) = \frac{1}{\pi} \langle\alpha, z| \rho_s |\alpha, z\rangle

    z is the optional squeezing parameter for the reference state:
    :math:`|\alpha, z\rangle := D(\alpha) S(z) |0\rangle`.
    The integral of H is normalized to unity.
    """
    # Ville Bergholm 2010

    if alpha == None:
        # return a 2D grid of W values
        a = linspace(lim[0], lim[1], res[0])
        b = linspace(lim[2], lim[3], res[1])
        #a, b = ogrid[lim[0]:lim[1]:1j*res[0], lim[2]:lim[3]:1j*res[1]]
        alpha = a + 1j*b[:, newaxis]
        return_ab = True
    else:
        return_ab = False

    # reference state
    n = prod(s.dims())
    ref = state(0, n).u_propagate(squeeze(z, n))
    ref /= sqrt(pi) # normalization included for convenience

    H = empty(alpha.shape)
    for k, c in enumerate(alpha.flat):
        temp = ref.u_propagate(displace(c, n))
        H.flat[k] = s.fidelity(temp) ** 2

    if return_ab:
        H = (H, a, b)
    return H


def wigner(s, alpha=None, res=(20, 20), lim=(-2, 2, -2, 2)):
    r"""Wigner quasi-probability distribution.
    ::

      W       = wigner(s, alpha)
      W, a, b = wigner(s, res=xxx, lim=yyy)

    Returns the Wigner quasi-probability distribution
    :math:`W(\mathrm{Im} \alpha, \mathrm{Re} \alpha)` corresponding to the harmonic
    oscillator state s given in the number basis.

    For a normalized state, the integral of W is normalized to unity.

    NOTE: The truncation of the number state space to a finite dimension
    results in spurious circular ripples in the Wigner function outside
    a given radius. To increase the accuracy, increase the state space dimension.
    """
    # Ville Bergholm 2010

    if alpha == None:
        # return a grid of W values for a grid of alphas
        a = linspace(lim[0], lim[1], res[0])
        b = linspace(lim[2], lim[3], res[1])
        #a, b = ogrid[lim[0]:lim[1]:1j*res[0], lim[2]:lim[3]:1j*res[1]]
        alpha = a + 1j*b[:, newaxis]
        return_ab = True
    else:
        return_ab = False

    # parity operator (diagonal)
    n = prod(s.dims())
    P = ones(n)
    P[1:n:2] = -1
    P *= 2 / pi  # include Wigner normalization here for convenience

    W = empty(alpha.shape)
    for k, c in enumerate(alpha.flat):
        temp = s.u_propagate(displace(-c, n))
        W.flat[k] = sum(P * temp.prob().real) # == ev(temp, P).real

    if return_ab:
        W = (W, a, b)
    return W
