# -*- coding: utf-8 -*-
"""
Born-Markov noise (:mod:`qit.markov`)
=====================================


This module simulates the effects of a heat bath coupled to a quantum
system, using the Born-Markov approximation.

The treatment in this module mostly follows Ref. :cite:`BP`.



Contents
--------

.. currentmodule:: qit.markov

.. autosummary::

   ops
   lindblad_ops
   superop   
   bath


:class:`bath` methods
---------------------

.. currentmodule:: qit.markov.bath

.. autosummary::

   build_LUT
   S_func
   set_cutoff
   corr
   fit
"""
# Ville Bergholm 2011-2014

from __future__ import division, absolute_import, print_function, unicode_literals

from numpy import (array, sqrt, exp, sin, cos, arctan2, tanh, dot, argsort, pi,
    r_, linspace, logspace, searchsorted, inf, newaxis, unravel_index)
from scipy.linalg import norm
from scipy.integrate import quad
import scipy.constants as const

from .base import sx, sz, tol
from .utils import lmul, rmul, lrmul, rand_hermitian, superop_lindblad, spectral_decomposition


__all__ = ['bath', 'ops', 'lindblad_ops', 'superop']


class bath(object):
    r"""Markovian heat bath.

    Currently only one type of bath is supported, a bosonic
    canonical ensemble at absolute temperature T, with a
    single-term coupling to the system.
    The bath spectral density is Ohmic with a cutoff:

    .. math::

       J(\omega) = \hbar^2 \omega \mathrm{cut}(\omega) \Theta(\omega)

    Two types of cutoffs are supported: exponential and sharp.

    .. math::

       \mathrm{cut}_{\text{exp}}(\omega)   &= \exp(-\omega / \omega_c),\\
       \mathrm{cut}_{\text{sharp}}(\omega) &= \Theta(\omega_c - \omega)

    The effects of the bath on the system are contained in the complex spectral correlation tensor

    .. math::

       \Gamma(\omega) = \frac{1}{2} \gamma(\omega) +i S(\omega)

    where :math:`\gamma` and :math:`S` are real.
    Computing values of this tensor is the main purpose of this class.

    .. math::

       \gamma(\omega) &= \frac{2 \pi}{\hbar^2} (J(\omega) -J(-\omega))(1 + n(\omega)),\\
       S(\omega) &= \frac{1}{\hbar^2} \int_0^\infty \mathrm{d}\nu J(\nu) \frac{\omega \coth(\beta \hbar \nu/2) +\nu}{\omega^2 -\nu^2}.


    where :math:`n(\omega) := 1/(e^{\beta \hbar \omega} - 1)` is the Planc function and :math:`\beta = 1/(k_B T)`. 
    Since :math:`\Gamma` is pretty expensive to compute, we store the computed results into a lookup table which is used to interpolate nearby values.


    Public data:

    ===========  ===========
    Data member  Description
    ===========  ===========
    type         Bath type. Currently only 'ohmic' is supported.
    omega0       Energy scale (in Hz). :math:`\hbar \omega_0` is the unit of energy for all Hamiltonians related to this bath.
    T            Absolute temperature of the bath (in K).
    scale        Dimensionless temperature scaling parameter :math:`\hbar \omega_0 / (k_B T)`.
    cut_type     Spectral density cutoff type (string).
    cut_limit    Spectral density cutoff limit :math:`\omega_c / \omega_0`.
    ===========  ===========    


    Private data (set automatically):

    ===========  ===========
    Data member  Description
    ===========  ===========
    cut_func     Spectral density cutoff function.
    j            Spectral density profile. :math:`J(\omega_0 x)/\omega_0 = \hbar^2 j(x) \mathrm{cut\_func}(x) \Theta(x)`.
    g_func       Spectral correlation tensor, real part. :math:`\gamma(\omega_0 x) / \omega_0 = \mathrm{g\_func}(x) \mathrm{cut\_func}(x)`. For the imaginary part, see :func:`S_func`.
    g0           :math:`\lim_{\omega \to 0} \gamma(\omega)`.
    s0           :math:`\lim_{\omega \to 0} S(\omega)`.
    dH           Lookup table.
    gs_table     Lookup table. :math:`(\gamma / S)(\omega_0 \text{dH[k]}) / \omega_0` = gs_table[k, (0/1)].
    ===========  ===========
    """
    # Ville Bergholm 2009-2011

    def __init__(self, type, omega0, T):
        """constructor
    
        Sets up a descriptor for a heat bath coupled to a quantum system.
        """
        # basic bath parameters
        self.type   = type
        self.omega0 = omega0
        self.T      = T
        # shorthand
        self.scale = const.hbar * omega0 / (const.k * T)

        if type == 'ohmic':
            # Ohmic bosonic bath, canonical ensemble, single-term coupling
            self.g_func = lambda x: 2 * pi * x * (1 + 1 / (exp(self.scale * x) - 1))
            self.g0 = 2 * pi / self.scale  # limit of g at x == 0
            self.j = lambda x: x
        else:
            raise ValueError('Unknown bath type.')

        # defaults, can be changed later
        self.set_cutoff('sharp', 20)


    def __repr__(self):
        """String representation."""
        return """Markovian heat bath.  Spectral density: {sd}, T = {temp:g}, omega0 = {omega:g}""".format(sd = self.type, temp = self.T, omega = self.omega0)


    def build_LUT(self):
        """Build a lookup table for the S integral. Unused.
        """
        raise RuntimeError('unused')
        # TODO justify limits for S lookup
        if limit > 10:
            temp = logspace(log10(10.2), log10(limit), 50)
            temp = r_[linspace(0.1, 10, 100), temp] # sampling is denser near zero, where S changes more rapidly
        else:
            temp = linspace(0.1, limit, 10)
        self.dH = r_[-temp[::-1], 0, temp]

        self.s_table = []
        for k in range(len(self.dH)):
            self.s_table[k] = self.S_func(self.dH[k])

        # limits at inifinity
        self.dH      = r_[-inf, self.dH, inf]
        self.s_table = r_[0, self.s_table, 0]

        plot(self.dH, self.s_table, 'k-x')


    def S_func(self, x):
        r"""Spectral correlation tensor, imaginary part.

        .. math::

           \mathrm{S\_func}(x) = S(x \omega_0) / \omega_0
           = \frac{1}{\hbar^2 \omega_0} P\int_0^\infty \mathrm{d}\nu J(\omega_0 \nu) \frac{x \coth(\nu \mathrm{scale}/2) +\nu}{x^2 -\nu^2}.
        """
        ep = 1e-5 # epsilon for Cauchy principal value
        if abs(x) <= 1e-8:
            return self.s0
        else:
            # Cauchy principal value, integrand has simple poles at \nu = \pm x.
            f = lambda nu: self.j(nu) * self.cut_func(nu) * (x / tanh(self.scale * nu / 2) + nu) / (x**2 -nu**2)
            a, abserr = quad(f, ep, abs(x) - ep) 
            b, abserr = quad(f, abs(x) + ep, inf) # 100 * self.cut_limit)
            return a + b


    def set_cutoff(self, type, lim):
        """Set the spectral density cutoff."""
        self.cut_type = type
        self.cut_limit = lim  # == omega_c/omega0

        # update cutoff function
        if self.cut_type == 'sharp':
            self.cut_func = lambda x: abs(x) <= self.cut_limit  # Heaviside theta cutoff
        elif self.cut_type == 'exp':
            self.cut_func = lambda x: exp(-abs(x) / self.cut_limit)  # exponential cutoff
        else:
            raise ValueError('Unknown cutoff type "{0}"'.format(self.cut_type))

        if self.type == 'ohmic':
            self.s0 = -self.cut_limit  # limit of S at dH == 0
  
        # clear lookup tables, since changing the cutoff requires recalc of S
        # start with a single point precomputed
        self.dH = array([-inf, 0, inf])
        self.gs_table = array([[0, 0], [self.g0, self.s0], [0, 0]])



    def corr(self, x):
        r"""Bath spectral correlation tensor.
        ::

          g, s = corr(x)

        Returns the bath spectral correlation tensor :math:`\Gamma` evaluated at :math:`\omega_0 x`:

        .. math::

           \Gamma(\omega_0 x) / \omega_0 = \frac{1}{2} g +i s
        """
        # Ville Bergholm 2009-2011

        tol = 1e-8
        max_w = 0.1 # maximum interpolation distance, TODO justify

        # assume parameters are set and lookup table computed
        #s = interp1(self.dH, self.s_table, x, 'linear', 0)

        # TODO dH and gs_table into a single dictionary?
        # binary search for the interval [dH_a, dH_b) in which x falls
        b = searchsorted(self.dH, x, side = 'right')
        a = b - 1
        ee = self.dH[[a, b]]
        tt = self.gs_table[[a, b], :]
        # now x is in [ee[0], ee[1])

        gap = ee[1] - ee[0]
        d1 = abs(x - ee[0])
        d2 = abs(x - ee[1])

        def interpolate(ee, tt, x):
            "Quick interpolation."
            # interp1 does way too many checks
            return tt[0] + ((x - ee[0]) / (ee[1] - ee[0])) * (tt[1] - tt[0])

        # x close enough to either endpoint?
        if d1 <= tol:
            return self.gs_table[a, :]
        elif d2 <= tol:
            return self.gs_table[b, :]
        elif gap <= max_w + tol:  # short enough gap to interpolate?
            return interpolate(ee, tt, x)
        else: # compute a new point p, then interpolate
            if gap <= 2 * max_w:
                p = ee[0] + gap / 2 # gap midpoint
                if x < p:
                    idx = 1 # which ee p will replace
                else:
                    idx = 0
            elif d1 <= max_w: # x within interpolation distance from one of the gap endpoints?
                p = ee[0] + max_w
                idx = 1
            elif d2 <= max_w:
                p = ee[1] - max_w
                idx = 0
            else: # x not near anything, don't interpolate
                p = x
                idx = 0

            # compute new g, s values at p and insert them into the table
            s = self.S_func(p)
            if abs(p) <= tol:
                g = self.g0 # limit at p == 0
            else:
                g = self.g_func(p) * self.cut_func(p)
            temp = array([[g, s]])

            self.dH = r_[self.dH[:b], p, self.dH[b:]]
            self.gs_table = r_[self.gs_table[:b], temp, self.gs_table[b:]]

            # now interpolate the required value
            ee[idx] = p
            tt[idx, :] = temp
            return interpolate(ee, tt, x)



    def fit(self, delta, T1, T2):
        r"""Qubit-bath coupling that reproduces given decoherence times.
        ::

          H, D = fit(delta, T1, T2)

        Returns the qubit Hamiltonian H and the qubit-bath coupling operator D
        that reproduce the decoherence times T1 and T2 (in units of :math:`1/\omega_0`)
        for a single-qubit system coupled to the bath.
        delta is the energy splitting for the qubit (in units of :math:`\hbar \omega_0`).

        The bath object is not modified.
        """
        # Ville Bergholm 2009-2010

        if self.type == 'ohmic':
            # Fitting an ohmic bath to a given set of decoherence times

            iTd = 1 / T2 -0.5 / T1 # inverse pure dephasing time
            if iTd < 0:
                raise ValueError('Unphysical decoherence times!')
    
            # match bath couplings to T1, T2
            temp = self.scale * delta / 2
            alpha = arctan2(1, sqrt(T1 * iTd / tanh(temp) * temp * self.cut_func(delta)))
            # dimensionless system-bath coupling factor squared
            N = iTd * self.scale / (4 * pi * cos(alpha)**2)

            # qubit Hamiltonian
            H = -delta/2 * sz

            # noise coupling
            D = sqrt(N) * (cos(alpha) * sz + sin(alpha) * sx)

            # decoherence times in scaled time units
            #T1 = 1/(N * sin(alpha)**2 * 2*pi * delta * coth(temp) * self.cut_func(delta))
            #T_dephase = self.scale/(N *4*pi*cos(alpha)**2)
            #T2 = 1/(0.5/T1 +1/T_dephase)
        else:
            raise NotImplementedError('Unknown bath type.')
        return H, D



def ops(H, D):
    r"""Jump operators for a Born-Markov master equation.
    ::

      dH, A = ops(H, D)

    Builds the jump operators for a Hamiltonian operator H and
    a (Hermitian) interaction operator D.

    Returns (dH, A), where dH is a list of the sorted unique nonnegative differences between
    eigenvalues of H, and A is a sequence of the corresponding jump operators:
    :math:`A_k(dH_i) = A[k][i]`.

    Since :math:`A_k(-dH) = A_k^\dagger(dH)`, only the nonnegative dH:s and corresponding A:s are returned.
    """
    # Ville Bergholm 2009-2011

    E, P = spectral_decomposition(H)
    m = len(E) # unique eigenvalues
    # energy difference matrix is antisymmetric, so we really only need the lower triangle
    deltaE = E[:, newaxis] - E  # deltaE[i,j] == E[i] - E[j]

    # mergesort is a stable sorting algorithm
    ind = argsort(deltaE, axis = None, kind = 'mergesort')
    # index of first lower triangle element
    s = m * (m - 1) / 2
    #assert(ind[s], 0)
    ind = ind[s:] # lower triangle indices only
    deltaE = deltaE.flat[ind] # lower triangle flattened

    if not isinstance(D, (list, tuple)):
        D = [D] # D needs to be a sequence, even if it has just one element
    n_D = len(D) # number of bath coupling ops

    # combine degenerate deltaE, build jump ops
    A = []
    # first dH == 0
    dH = [deltaE[0]]
    r, c = unravel_index(ind[0], (m, m))
    for d in D:
        A.append( [ dot(dot(P[c], d), P[r]) ] )

    for k in range(1, len(deltaE)):
        r, c = unravel_index(ind[k], (m, m))
        if abs(deltaE[k] - deltaE[k-1]) > tol:
            # new omega value, new jump op
            dH.append(deltaE[k])
            for op in range(n_D):
                A[op].append( dot(dot(P[c], D[op]), P[r]) )
        else:
            # extend current op
            for op in range(n_D):
                A[op][-1] += dot(dot(P[c], D[op]), P[r])

    return dH, A



def _check_baths(B):
    """Internal helper."""
    if not isinstance(B, (list, tuple)):
        B = [B] # needs to be a list, even if it has just one element

    # make sure the baths have the same omega0!
    temp = B[0].omega0
    for k in B:
        if k.omega0 != temp:
            raise ValueError('All the baths must have the same energy scale omega0!')
    return B



def lindblad_ops(H, D, B):
    r"""Lindblad operators for a Born-Markov master equation.
    ::

       L, H_LS = lindblad_ops(H, D, B)

    Builds the Lindblad operators corresponding to a
    base Hamiltonian H and a (Hermitian) interaction operator D
    coupling the system to bath B.

    Returns :math:`L = \{A_i / \omega_0 \}_i` and :math:`H_{\text{LS}} / (\hbar \omega_0)`,
    where :math:`A_i` are the Lindblad operators and :math:`H_{\text{LS}}` is the Lamb shift.

    B can also be a list of baths, in which case D has to be
    a list of the corresponding interaction operators.
    """
    # Ville Bergholm 2009-2011

    B = _check_baths(B)

    # jump ops
    dH, X = ops(H, D)
    H_LS = 0
    L = []
    for n, b in enumerate(B):
        A = X[n] # jump ops for bath/interaction op n

        # dH == 0 terms
        g, s = b.corr(0)
        L.append(sqrt(g) * A[0])
        H_LS += s * dot(A[0].conj().transpose(), A[0])  # Lamb shift

        for k in range(1, len(dH)):
            # first the positive energy shift
            g, s = b.corr(dH[k])
            L.append(sqrt(g) * A[k])
            H_LS += s * dot(A[k].conj().transpose(), A[k])

            # now the corresponding negative energy shift
            g, s = b.corr(-dH[k])
            L.append(sqrt(g) * A[k].conj().transpose())   # note the difference here, A(-omega) = A'(omega)
            H_LS += s * dot(A[k], A[k].conj().transpose()) # here too

    return L, H_LS
    # TODO ops for different baths can be combined into a single basis,
    # N^2-1 ops max in total



def superop(H, D, B):
    r"""Liouvillian superoperator for a Born-Markov master equation.

    Builds the Liouvillian superoperator L corresponding to a
    base Hamiltonian H and a (Hermitian) interaction operator D
    coupling the system to bath B.

    Returns :math:`L/\omega_0`, which includes the system Hamiltonian, the Lamb shift,
    and the Lindblad dissipator.

    B can also be a list of baths, in which case D has to be
    a list of the corresponding interaction operators.
    """
    # Ville Bergholm 2009-2011

    B = _check_baths(B)

    # jump ops
    dH, X = ops(H, D)
    iH_LS = 1j * H  # i * (system Hamiltonian + Lamb-Stark shift)
    acomm = 0
    diss = 0
    for n, b in enumerate(B):
        A = X[n] # jump ops for bath/interaction op n

        # we build the Liouvillian in a funny order to be a bit more efficient
        # dH == 0 terms
        [g, s] = b.corr(0)
        temp = dot(A[0].conj().transpose(), A[0])

        iH_LS += (1j * s) * temp  # Lamb shift
        acomm += (-0.5 * g) * temp # anticommutator
        diss  += lrmul(g * A[0], A[0].conj().transpose()) # dissipator (part)

        for k in range(1, len(dH)):
            # first the positive energy shift
            g, s = b.corr(dH[k])
            temp = dot(A[k].conj().transpose(), A[k])
            iH_LS += (1j * s) * temp
            acomm += (-0.5 * g) * temp
            diss  += lrmul(g * A[k], A[k].conj().transpose())

            # now the corresponding negative energy shift
            g, s = b.corr(-dH[k])
            temp = dot(A[k], A[k].conj().transpose()) # note the difference here, A(-omega) = A'(omega)
            iH_LS += (1j * s) * temp
            acomm += (-0.5 * g) * temp
            diss  += lrmul(g * A[k].conj().transpose(), A[k]) # here too

    return lmul(acomm -iH_LS) +rmul(acomm +iH_LS) +diss
