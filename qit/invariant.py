# -*- coding: utf-8 -*-
"""
Local invariants (:mod:`qit.invariant`)
=======================================

This module contains tools for computing and plotting the values of
various local gate and state invariants.

.. currentmodule:: qit.invariant

Contents
--------

.. autosummary::

   LU
   canonical
   makhlin
   max_concurrence
   plot_makhlin_2q
   plot_weyl_2q   
"""
# Ville Bergholm 2011

from __future__ import division, absolute_import, print_function, unicode_literals

import numpy as np
from numpy import (array, asarray, arange, empty, zeros, ones, sqrt, sin, cos, dot, sort, trace, kron,
    pi, r_, c_, linspace, meshgrid, roll, concatenate, angle)
from numpy.linalg import det, eigvals
from scipy.linalg import norm
import matplotlib.pyplot as plt

from .base import sy, Q_Bell
from .lmap import lmap, tensor


# TODO these function names are terrible
__all__ = ['canonical', 'makhlin', 'max_concurrence', 'plot_weyl_2q', 'plot_makhlin_2q', 'LU']


def LU(rho, k, perms):
    r"""Local unitary polynomial invariants of quantum states.

    Computes the permutation invariant :math:`I_{k; \pi_1, \pi_2, \ldots, \pi_n}` for the state :math:`\rho`.
    perms is a tuple containing n k-permutation tuples.

    Example: :math:`I_{3; (123),(12)}(\rho)` = LU_inv(rho, 3, [(1, 2, 0), (1, 0, 2)])

    This function can be very inefficient for some invariants, since
    it does no partial traces etc. which might simplify the calculation.

    Uses the algorithm in :cite:`BBL2012`.
    """
    # Ville Bergholm 2011-2012

    def tensor_pow(rho, n):
        """Returns $\rho^{\otimes n}$."""
        rho.to_op(inplace=True)
        ret = lmap(rho)
        for _ in range(1, n):
            ret = tensor(ret, rho)
        return ret


    n = len(perms)
    if n != rho.subsystems():
        raise ValueError('Need one permutation per subsystem.')

    # convert () to identity permutation
    id_perm = tuple(range(k))
    perms = list(perms)
    for j, p in enumerate(perms):
        if len(p) == 0:
            perms[j] = id_perm

    # splice k sequential copies of the entire system into k copies of each subsystem
    s = arange(n * k).reshape((k, n)).flatten()

    # permute the k copies of each subsystem
    temp = kron(k * arange(n), ones(k, int))
    p = asarray(perms).flatten() + temp

    # Permutations: a*b = a(b), x = y * z^{-1}  <=>  x * z = x(z) = y.
    s_inv = empty(s.shape, int)
    s_inv[s] = arange(n*k)
    total = s_inv[p[s]] # total = s^{-1} * p * s

    # TODO this could be done much more efficiently
    return tensor_pow(rho, k).reorder((total, None)).trace()



def canonical(U):
    """Canonical local invariants of a two-qubit gate.

    Returns a vector of three real canonical local invariants for the
    U(4) matrix U, normalized to the range [0,1].
    Uses the algorithm in :cite:`Childs`.
    """
    # Ville Bergholm 2004-2010

    sigma = kron(sy, sy)
    U_flip = dot(dot(sigma, U.transpose()), sigma)  # spin flipped U
    temp = dot(U, U_flip) / sqrt(complex(det(U)))

    Lambda = eigvals(temp) #[exp(i*2*phi_1), etc]
    # logarithm to the branch (-1/2, 3/2]
    Lambda = angle(Lambda) / pi # divide pi away
    for k in range(len(Lambda)):
        if Lambda[k] <= -0.5:
            Lambda[k] += 2
    S = Lambda / 2
    S = sort(S)[::-1]  # descending order

    n = int(round(sum(S)))  # sum(S) must be an integer
    # take away extra translations-by-pi
    S -= r_[ones(n), zeros(4-n)]
    # put the elements in the correct order
    S = roll(S, -n)

    M = [[1, 1, 0], [1, 0, 1], [0, 1, 1]] # scaled by factor 2
    c = dot(M, S[:3])
    # now 0.5 >= c[0] >= c[1] >= |c[2]|
    # and into the Berkeley chamber using a translation and two Weyl reflections
    if c[2] < 0:
        c[0] = 1 - c[0]
        c[2] = -c[2]
    return c


def makhlin(U):
    """Makhlin local invariants of a two-qubit gate.

    Returns a vector of the three real Makhlin invariants (see :cite:`Makhlin`) corresponding
    to the U(4) gate U.

    Alternatively, given a vector of canonical invariants normalized to [0, 1],
    returns the corresponding Makhlin invariants (see :cite:`Zhang`).
    """
    # Ville Bergholm 2004-2010

    if U.shape[-1] == 3:
        c = U
        # array consisting of vectors of canonical invariants
        c *= pi
        g = empty(c.shape)
        
        g[..., 0] = (cos(c[..., 0]) * cos(c[..., 1]) * cos(c[..., 2])) ** 2 -(sin(c[..., 0]) * sin(c[..., 1]) * sin(c[..., 2])) ** 2
        g[..., 1] = 0.25 * sin(2 * c[..., 0]) * sin(2 * c[..., 1]) * sin(2 * c[..., 2])
        g[..., 2] = 4 * g[..., 0] - cos(2 * c[..., 0]) * cos(2 * c[..., 1]) * cos(2*c[..., 2])
    else:
        # U(4) gate matrix    
        V = dot(Q_Bell.conj().transpose(), dot(U, Q_Bell))
        M = dot(V.transpose(), V)

        t1 = trace(M) ** 2
        t2 = t1 / (16 * det(U))
        g = array([t2.real, t2.imag, ((t1 - trace(dot(M, M))) / (4 * det(U))).real])
    return g


def max_concurrence(U):
    """Maximum concurrence generated by a two-qubit gate.

    Returns the maximum concurrence generated by the two-qubit
    gate U (see :cite:`Kraus`), starting from a tensor state.

    Alternatively, U may be given in terms of a vector of three
    canonical local invariants.
    """
    # Ville Bergholm 2006-2010

    if U.shape[-1] == 4:
        # gate into corresponding invariants
        c = canonical(U)
    else:
        c = U
    temp = roll(c, 1, axis = -1)
    return np.max(abs(sin(pi * concatenate((c - temp, c + temp), axis = -1))), axis = -1)


def plot_makhlin_2q(sdiv=31, tdiv=31):
    """Plots the set of two-qubit gates in the space of Makhlin invariants.

    Plots the set of two-qubit gates in the space of Makhlin
    invariants (see :func:`makhlin`), returns the Axes3D object.

    The input parameters are the s and t divisions of the mesh.
    """
    # Ville Bergholm 2006-2011

    import matplotlib.cm as cm
    import matplotlib.colors as colors

    s = linspace(0, pi,   sdiv)
    t = linspace(0, pi/2, tdiv)

    # more efficient than meshgrid
    #g1 = kron(cos(s).^2, cos(t).^4) - kron(sin(s).^2, sin(t).^4)
    #g2 = 0.25*kron(sin(2*s), sin(2*t).^2)
    #g3 = 4*g1 - kron(cos(2*s), cos(2*t).^2)
    #S = kron(s, ones(size(t)))
    #T = kron(ones(size(s)), t)

    # canonical coordinate plane (s, t, t) gives the entire surface of the set of gate equivalence classes
    S, T = meshgrid(s, t)
    c = c_[S.ravel(), T.ravel(), T.ravel()]
    G = makhlin(c).reshape(sdiv, tdiv, 3)
    C = max_concurrence(c).reshape(sdiv, tdiv)

    fig = plt.gcf()
    ax = fig.add_subplot(111, projection='3d')

    # mesh, waterfall?
    polyc = ax.plot_surface(G[:, :, 0], G[:, :, 1], G[:, :, 2], rstride = 1, cstride = 1,
        cmap = cm.jet, norm = colors.Normalize(vmin=0, vmax=1, clip=True), alpha = 0.6)
    polyc.set_array(C.ravel() ** 2)  # FIXME colors
    ax.axis('equal')
    #ax.axis([-1, 1, -0.5, 0.5, -3, 3])
    #ax.shading('interp')

    ax.set_xlabel('$g_1$')
    ax.set_ylabel('$g_2$')
    ax.set_zlabel('$g_3$')
    plt.title('Makhlin stingray')

    # labels
    ax.text(1.05, 0, 2.7, 'I')
    ax.text(-1.05, 0, -2.7, 'SWAP')
    ax.text(-0.1, 0, 1.2, 'CNOT')
    ax.text(0.1, 0, -1.2, 'DCNOT')
    ax.text(0.1, 0.26, 0, 'SWAP$^{1/2}$')
    ax.text(0, -0.26, 0, 'SWAP$^{-1/2}$')

    fig.colorbar(polyc, ax = ax)
    plt.show()
    return ax


def plot_weyl_2q(ax=None):
    """Plots the two-qubit Weyl chamber.

    Plots the Weyl chamber for the local invariants
    of 2q gates. See :cite:`Zhang`.

    Returns the Axes3D object.
    """
    # Ville Bergholm 2005-2012

    if ax == None:
        ax = plt.subplot(111, projection='3d')
    ax.hold(True)
    ax.plot_surface(array([[0, 0.5, 1], [0, 0.5, 1]]), array([[0, 0, 0], [0, 0.5, 0]]), array([[0, 0, 0], [0, 0.5, 0]]), alpha = 0.2)
    ax.plot_surface(array([[0, 0.5], [0, 0.5]]), array([[0, 0.5], [0, 0.5]]), array([[0, 0], [0, 0.5]]), alpha = 0.2)
    ax.plot_surface(array([[0.5, 1], [0.5, 1]]), array([[0.5, 0], [0.5, 0]]), array([[0, 0], [0.5, 0]]), alpha = 0.2)
    #axis([0 1 0 0.5 0 0.5])
    ax.axis('equal')
    ax.set_xlabel('$c_1/\\pi$')
    ax.set_ylabel('$c_2/\\pi$')
    ax.set_zlabel('$c_3/\\pi$')
    plt.title('Two-qubit Weyl chamber')

    ax.text(-0.05, -0.05, 0, 'I')
    ax.text(1.05, -0.05, 0, 'I')
    ax.text(0.45, 0.55, 0.55, 'SWAP')
    ax.text(0.45, -0.05, 0, 'CNOT')
    ax.text(0.45, 0.55, -0.05, 'DCNOT')
    ax.text(0.20, 0.25, 0, 'SWAP$^{1/2}$')
    ax.text(0.75, 0.25, 0, 'SWAP$^{-1/2}$')
    return ax
