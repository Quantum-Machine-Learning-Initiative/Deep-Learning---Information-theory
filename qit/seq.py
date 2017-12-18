# -*- coding: utf-8 -*-
r"""
Control sequences (:mod:`qit.seq`)
==================================


Piecewise constant control sequences for quantum systems.
Each control sequence is a dictionary with the following keys:

=======  ===============================================================================================
A        Drift generator (typically :math:`i/\hbar` times a Hamiltonian and a time unit of your choice).
B        List of control generators. c := len(B).
tau      Vector of durations of the time slices. m := len(tau).
control  Array, shape (m, c). control[i,j] is the value of control field j during time slice i.
=======  ===============================================================================================

The total generator for the time slice j is thus given by

.. math::

   G_j = A +\sum_k \text{control}_{jk} B_k,

and the corresponding propagator is

.. math::

   P_j = \exp(-\tau_j G_j).


NOTE the unusual sign convention.


.. currentmodule:: qit.seq

Contents
--------

.. autosummary::

   nmr
   corpse
   bb1
   scrofulous
   cpmg
   seq2prop
   propagate
"""
# Ville Bergholm 2011-2014

from __future__ import division, absolute_import, print_function, unicode_literals

from numpy import array, sin, cos, arcsin, arccos, pi, asarray, eye, zeros, r_, c_, dot, nonzero, ceil, linspace
from scipy.linalg import expm
from scipy.optimize import brentq

from .base import sx, sy, tol


__all__ = ['nmr', 'bb1', 'corpse', 'cpmg', 'scrofulous', 'seq2prop', 'propagate']


def nmr(a):
    r"""Convert NMR-style rotations into a one-qubit control sequence.

    Returns a one-qubit control sequence corresponding to the array a:

    .. math::

       a = [[\theta_1, \phi_1], [\theta_2, \phi_2], ...]

    Each :math:`\theta, \phi` pair corresponds to a NMR rotation
    of the form :math:`\theta_\phi`,
    or a rotation of the angle :math:`\theta`
    about the unit vector :math:`[\cos(\phi), \sin(\phi), 0]`.

    .. math::

       R_{\vec{a}}(\theta) = \exp(-i \vec{a} \cdot \vec{\sigma} \theta/2) = \exp(-i H t) \quad \Leftarrow \quad
       H = \vec{a} \cdot \vec{\sigma}/2, \quad t = \theta.
    """
    # Ville Bergholm 2006-2009

    a = asarray(a, dtype=float)
    theta = a[:, 0]
    phi   = a[:, 1]

    # find theta angles that are negative, convert them to corresponding positive rotations
    rows = nonzero(theta < 0)[0]
    theta[rows] = -theta[rows]
    phi[rows] = phi[rows] + pi

    # construct the sequence TODO make it a class?
    # NOTE the strange sign convention in A and B
    s = {'A': zeros((2, 2)),
         'B': [0.5j * sx, 0.5j * sy],
         'tau': theta,
         'control': c_[cos(phi), sin(phi)]
         }
    return s 


def bb1(theta, phi=0, location=0.5):
    r"""Sequence for correcting pulse length errors.

    Returns the Broadband number 1 control sequence for correcting
    proportional errors in pulse length (or amplitude) :cite:`Wimperis`.

    The target rotation is :math:`\theta_\phi` in the NMR notation.
    """
    # Ville Bergholm 2009-2012

    ph1 = arccos(-theta / (4*pi))
    W1  = [[pi, ph1], [2*pi, 3*ph1], [pi, ph1]]
    return nmr([[location * theta, phi]] + W1 + [[(1-location) * theta, phi]])


def corpse(theta, phi=0):
    r"""Sequence for correcting off-resonance errors.

    Returns the CORPSE control sequence for correcting off-resonance
    errors, i.e. ones arising from a constant but unknown
    :math:`\sigma_z` bias in the Hamiltonian :cite:`Cummins`

    The target rotation is :math:`\theta_\phi` in the NMR notation.

    CORPSE: Compensation for Off-Resonance with a Pulse SEquence
    """
    # Ville Bergholm 2009

    n = [1, 1, 0] # CORPSE
    #n = [0, 1, 0] # short CORPSE

    temp = arcsin(sin(theta / 2) / 2)

    th1 = 2*pi*n[0] +theta/2 -temp
    th2 = 2*pi*n[1] -2*temp
    th3 = 2*pi*n[2] +theta/2 -temp
    return nmr([[th1, phi], [th2, phi+pi], [th3, phi]])


def cpmg(t, n):
    r"""Carr-Purcell-Meiboom-Gill sequence.

    Returns the Carr-Purcell-Meiboom-Gill sequence of n repeats with waiting time t.
    The purpose of the CPMG sequence is to facilitate a T_2 measurement
    under a nonuniform z drift, it is not meant to be a full memory protocol.
    The target operation for this sequence is identity.
    """
    # Ville Bergholm 2007-2012

    s = nmr([[pi/2, pi/2]]) # initial y rotation

    # step: wait, pi x rotation, wait
    step_tau  = array([t, pi, t])
    step_ctrl = array([[0, 0], [1, 0], [0, 0]])
    for k in range(n):
        s['tau'] = r_[s['tau'], step_tau]
        s['control'] = r_[s['control'], step_ctrl]
    return s


def scrofulous(theta, phi=0):
    r"""Sequence for correcting pulse length errors.

    Returns the SCROFULOUS control sequence for correcting errors
    in pulse duration (or amplitude) :cite:`Cummins`.

    The target rotation is :math:`\theta_\phi` in the NMR notation.

    SCROFULOUS: Short Composite ROtation For Undoing Length Over- and UnderShoot
    """
    # Ville Bergholm 2006-2014

    th1 = brentq(lambda t: (sin(t)/t -(2 / pi) * cos(theta / 2)), 0.1, 4.6)
    ph1 = arccos(-pi * cos(th1) / (2 * th1 * sin(theta / 2)))
    ph2 = ph1 - arccos(-pi / (2 * th1))

    u1 = [[th1, ph1 +phi]]
    u2 = [[pi,  ph2 +phi]]
    return nmr(u1 + u2 + u1)


def seq2prop(s):
    r"""Propagator corresponding to a control sequence.

    Returns the propagator matrix corresponding to the
    action of the control sequence s.

    Governing equation: :math:`\dot(X)(t) = -(A +\sum_k u_k(t) B_k) X(t) = -G(t) X(t)`.
    """
    # Ville Bergholm 2009-2012

    A = s['A']
    B = s['B']

    n = len(s['tau'])
    P = eye(A.shape[0])
    for j in range(n):
        G = A
        for k, b in enumerate(B):
            G = G + s['control'][j, k] * b

        temp = expm(-s['tau'][j] * G)  # NOTE the sign convention here
        P = dot(temp, P)

    return P


def propagate(s, seq, out_func=lambda x: x, base_dt=0.1):
    """Propagate a state in time using a control sequence.
    
    If no output function is given, we use an identity map.
    """
    # Ville Bergholm 2009-2014

    A = seq['A']
    B = seq['B']

    n = len(seq['tau'])
    t = [0]  # initial time
    out = [out_func(s)]  # initial state

    # loop over the sequence
    for j in range(n):
        G = A
        for k, b in enumerate(B):
            G = G + seq['control'][j, k] * b

        T = seq['tau'][j]  # pulse duration
        n_steps = max(int(ceil(T / base_dt)), 1)
        dt = T / n_steps

        P = expm(-G * dt)  # NOTE TODO the sign convention here
        for k in range(n_steps):
            s = s.u_propagate(P)
            out.append(out_func(s))

        temp = t[-1]
        t.extend(list(linspace(temp+dt, temp+T, n_steps)))
    return out, t
