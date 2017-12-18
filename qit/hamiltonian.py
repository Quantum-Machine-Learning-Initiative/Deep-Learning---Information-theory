# -*- coding: utf-8 -*-
"""
Model Hamiltonians (:mod:`qit.hamiltonian`)
===========================================

This module has methods that generate several common types of model Hamiltonians used in quantum mechanics.


.. currentmodule:: qit.hamiltonian

Contents
--------

.. autosummary::

   heisenberg
   jaynes_cummings
   hubbard
   bose_hubbard
   holstein
"""
# Ville Bergholm 2014

from __future__ import division, absolute_import, print_function, unicode_literals

import numpy as np
from numpy import asarray, conj, transpose, dot

from .base import sx, sz
from .utils import angular_momentum, op_list, boson_ladder, fermion_ladder


__all__ = [
    'heisenberg',
    'jaynes_cummings',
    'hubbard',
    'bose_hubbard',
    'holstein']


def _cdot(v, A):
    """Real dot product of a vector and a tuple of operators."""
    res = 0j
    for vv, AA in zip(v, A):
        res += vv * AA
    return res


def heisenberg(dim, C=None, J=(0, 0, 2), B=(0, 0, 1)):
    r"""Heisenberg spin network model.

    Returns the Hamiltonian H for the Heisenberg model, describing a network
    of n interacting spins in an external magnetic field.

    dim is an n-tuple of the dimensions of the spins, i.e. dim == (2, 2, 2)
    would be a system of three spin-1/2's.

    C is the :math:`n \times n` connection matrix of the spin network, where C[i,j]
    is the coupling strength between spins i and j. Only the upper triangle is used.

    J defines the form of the spin-spin interaction. It is either a 3-tuple or a
    function J(i, j) returning a 3-tuple for site-dependent interactions.
    Element k of the tuple is the coefficient of the Hamiltonian term :math:`S_k^{(i)} S_k^{(j)}`,
    where :math:`S_k^{(i)}` is the k-component of the angular momentum of spin i.

    B defines the effective magnetic field the spins locally couple to. It's either
    a 3-tuple (homogeneous field) or a function B(a) that returns a 3-tuple for
    site-dependent field.

    .. math::

      H = \sum_{\langle i,j \rangle} \sum_{k = x,y,z} J(i,j)[k] S_k^{(i)} S_k^{(j)}  +\sum_i \vec{B}(i) \cdot \vec{S}^{(i)})

    Examples::

      C = np.eye(n, n, 1)  linear n-spin chain
      J = (2, 2, 2)        isotropic Heisenberg coupling
      J = (2, 2, 0)        XX+YY coupling
      J = (0, 0, 2)        Ising ZZ coupling
      B = (0, 0, 1)        homogeneous Z-aligned field
    """
    # Ville Bergholm 2009-2014

    n = len(dim) # number of spins in the network

    if C == None:
        # linear chain
        C = np.eye(n, n, 1)

    # make J and B into functions
    if isinstance(J, tuple):
        if len(J) != 3:
            raise ValueError('J must be either a 3-tuple or a function.')
        J = asarray(J)
        Jf = lambda i, j: C[i, j] * J
    else:
        Jf = J

    if isinstance(B, tuple):
        if len(B) != 3:
            raise ValueError('B must be either a 3-tuple or a function.')
        Bf = lambda i: B
    else:
        Bf = B

    # local magnetic field terms
    temp = []
    for i in range(n):
        A = angular_momentum(dim[i])  # spin ops
        temp.append([(_cdot(Bf(i), A), i)])
    H = op_list(temp, dim)

    # spin-spin couplings: loop over nonzero entries of C
    # only use the upper triangle
    C = np.triu(C)
    for i, j in transpose(C.nonzero()):
        # spin ops for sites i and j
        Si = angular_momentum(dim[i])
        Sj = angular_momentum(dim[j])
        temp = []
        # coupling between sites a and b
        c = Jf(i, j)
        for k in range(3):
            temp.append([(c[k] * Si[k], i), (Sj[k], j)])
        H += op_list(temp, dim)

    return H, dim



def jaynes_cummings(om_a, om_c, omega, m=10):
    r"""Jaynes-Cummings model, a two-level atom in a single-mode cavity.

    Returns the Hamiltonian H and the dimension vector dim for an
    implementation of the Jaynes-Cummings model, describing a two-level atom coupled
    to a harmonic oscillator (e.g. a single EM field mode in an optical cavity).

    .. math::

      H/\hbar = \frac{\omega_a}{2} \sigma_z +\omega_c a^\dagger a +\frac{\Omega}{2} \sigma_x (a+a^\dagger)

    The dimension of the Hilbert space of the bosonic cavity mode (infinite in principle) is truncated to m.
    """
    # Ville Bergholm 2014

    dim = (2, m)
    a = boson_ladder(m) 
    ax = a.conj().transpose()

    temp = [[(om_a/2 * sz, 0)],
            [(om_c * dot(ax, a), 1)],
            [(omega * sx, 0), (a+ax, 1)]]
    H = op_list(temp, dim)

    return H, dim



def hubbard(C, U=1, mu=0):
    r"""Hubbard model, fermions on a lattice.

    Returns the Hamiltonian H and the dimension vector dim for an
    implementation of the Hubbard model.

    The model consists of spin-1/2 fermions confined in a graph defined by the
    symmetric connection matrix C (only upper triangle is used).
    The fermions interact with other fermions at the same site with interaction strength U,
    as well as with an external chemical potential mu.
    The Hamiltonian has been normalized by the fermion hopping constant t.

    .. math::

      H = -\sum_{\langle i,j \rangle, \sigma} c^\dagger_{i,\sigma} c_{j,\sigma}
        +\frac{U}{t} \sum_i n_{i,up} n_{i,down} -\frac{\mu}{t} \sum_i (n_{i,up}+n_{i,down})
    """
    # Ville Bergholm 2010-2014

    n = len(C)
    dim = 2 * np.ones(2*n)  # n sites, two fermionic modes per site

    # fermion annihilation ops f[site, spin]
    f = fermion_ladder(2 * n).reshape((n, 2))
    # NOTE all the f ops have the full Hilbert space dimension

    H = 0j

    for k in range(n):
        # number operators for this site
        n1 = dot(f[k, 0].conj().transpose(), f[k, 0])
        n2 = dot(f[k, 1].conj().transpose(), f[k, 1])
        # on-site interaction
        H += U * dot(n1, n2)
        # chemical potential
        H += -mu * (n1 + n2)

    # fermions hopping: loop over nonzero entries of C
    # only use the upper triangle
    C = np.triu(C)
    for i, j in transpose(C.nonzero()):
        for s in range(2):
            H -= dot(f[i, s].conj().transpose(), f[j, s]) +dot(f[j, s].conj().transpose(), f[i, s])

    return H, dim



def bose_hubbard(C, U=1, mu=0, m=10):
    r"""Bose-Hubbard model, bosons on a lattice.

    Returns the Hamiltonian H and the dimension vector dim for an
    implementation of the Bose-Hubbard model.

    The model consists of spinless bosons confined in a graph defined by the
    symmetric connection matrix C (only upper triangle is used).
    The bosons interact with other bosons at the same site with interaction strength U,
    as well as with an external chemical potential mu.
    The Hamiltonian has been normalized by the boson hopping constant t.

    .. math::

      H = -\sum_{\langle i,j \rangle} b^\dagger_i b_{j} +\frac{U}{2t} \sum_i n_i (n_i-1) -\frac{\mu}{t} \sum_i n_i

    The dimensions of the boson Hilbert spaces (infinite in principle) are truncated to m.
    """
    # Ville Bergholm 2010-2014

    n = len(C)
    dim = m * np.ones(n)

    b = boson_ladder(m)  # boson annihilation op
    b_dagger = b.conj().transpose()  # boson creation op
    nb = dot(b_dagger, b)  # boson number op

    I = np.eye(m)
    A = U/2 * dot(nb, (nb-I)) # on-site interaction
    B = -mu * nb # chemical potential

    temp = []
    for k in range(n):
        temp.append([(A+B, k)])
    H = op_list(temp, dim)

    temp = []
    # bosons hopping: loop over nonzero entries of C
    # only use the upper triangle
    C = np.triu(C)
    for i, j in transpose(C.nonzero()):
        temp.extend([[(b_dagger, i), (b, j)], [(b, i), (b_dagger, j)]])

    H -= op_list(temp, dim)
    return H, dim



def holstein(C, omega=1, g=1, m=10):
    r"""Holstein model, electrons on a lattice coupled to phonons.

    Returns the Hamiltonian H and the dimension vector dim for an implementation of the Holstein model.

    The model consists of spinless electrons confined in a graph defined by the
    symmetric connection matrix C (only upper triangle is used),
    coupled to phonon modes represented by a harmonic oscillator at each site.
    The dimensions of phonon Hilbert spaces (infinite in principle) are truncated to m.

    The order of the subsystems is [e1, ..., en, p1, ..., pn].
    The Hamiltonian has been normalized by the electron hopping constant t.

    .. math::

      H = -\sum_{\langle i,j \rangle} c_i^\dagger c_j  +\frac{\omega}{t} \sum_i b_i^\dagger b_i
        -\frac{g \omega}{t} \sum_i (b_i + b_i^\dagger) c_i^\dagger c_i
    """
    # Ville Bergholm 2010-2014

    n = len(C)
    # Hilbert space: electrons first, then phonons
    dim = np.r_[2**n, m * np.ones(n)]  # Jordan-Wigner clumps all fermion dims together

    c = fermion_ladder(n)  # electron annihilation ops
    b = boson_ladder(m)    # phonon annihilation
    b_dagger = b.conj().transpose()  # phonon creation
    q = b + b_dagger       # phonon position
    nb = dot(b_dagger, b)  # phonon number operator

    temp = []
    for k in range(n):
        # phonon harmonic oscillators
        temp.append([(omega * nb, 1+k)])
        # electron-phonon interaction
        temp.append([(-g * omega * dot(c[k].conj().transpose(), c[k]), 0), (q, 1+k)])
    H = op_list(temp, dim)

    # fermions hopping: loop over nonzero entries of C
    # only use the upper triangle
    C = np.triu(C)
    T = 0j
    for i, j in transpose(C.nonzero()):
        T += dot(c[i].conj().transpose(), c[j]) +dot(c[j].conj().transpose(), c[i])
    H += op_list([[(-T, 0)]], dim)

    # actual dimensions
    #dim = [2*ones(1, n), m*ones(1, n)]
    return H, dim
