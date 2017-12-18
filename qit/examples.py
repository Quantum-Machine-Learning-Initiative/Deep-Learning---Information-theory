# -*- coding: utf-8 -*-
"""
Demos and examples (:mod:`qit.examples`)
========================================

.. currentmodule:: qit.examples

This module contains various examples which demonstrate the features
of QIT. The :func:`tour` function is a "guided tour" to the toolkit
which runs all the demos in succession.


Demos
-----

.. autosummary::

   adiabatic_qc_3sat
   bb84
   bernstein_vazirani
   grover_search
   markov_decoherence
   nmr_sequences
   phase_estimation_precision
   qft_circuit
   quantum_channels
   quantum_walk
   qubit_and_resonator
   shor_factorization
   superdense_coding
   teleportation
   tour


General-purpose quantum algorithms
----------------------------------

.. autosummary::

   adiabatic_qc
   phase_estimation
   find_order
"""
# Ville Bergholm 2011-2014

from __future__ import division, absolute_import, print_function, unicode_literals

from math import asin
from operator import mod
from copy import deepcopy

import numpy as np
from numpy import (asarray, array, diag, kron, prod, floor, ceil, sqrt, log2, exp, angle, arange, linspace,
    logical_not, sin, cos, arctan2, empty, zeros, ones, eye, sort, nonzero, pi, trace, dot, meshgrid, r_)
from numpy.linalg import eig, norm
import numpy.random as npr
from scipy.misc import factorial
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D   # FIXME makes buggy 3d plotting work for some reason


from .base import sx, sy, sz, p0, p1, tol
from .lmap import lmap, tensor, numstr_to_array
from .utils import gcd, lcm, qubits, R_y, rand_U, rand_SU, mkron, boson_ladder
from .state import state, fidelity
from . import gate, ho, plot


__all__ = [
 'adiabatic_qc_3sat',
 'adiabatic_qc',
 'bb84',
 'bernstein_vazirani',
 'grover_search',
 'markov_decoherence',
 'nmr_sequences',
 'phase_estimation',
 'phase_estimation_precision',
 'qft_circuit',
 'quantum_channels',
 'quantum_walk',
 'qubit_and_resonator',
 'shor_factorization',
 'find_order',
 'superdense_coding',
 'teleportation',
 'tour']


def adiabatic_qc_3sat(n=6, n_clauses=None, clauses=None, problem='3sat'):
    """Adiabatic quantum computing demo.

    This example solves random 3-SAT problems by simulating the
    adiabatic quantum algorithm of :cite:`Farhi`.

    Note that this is incredibly inefficient because we first essentially
    solve the NP-complete problem classically using an exhaustive
    search when computing the problem Hamiltonian H1, and then
    simulate an adiabatic quantum computer solving the same problem
    using the quantum algorithm.
    """
    # Ville Bergholm 2009-2011

    print('\n\n=== Solving 3-SAT using adiabatic qc ===\n')

    if n < 3:
        n = 3

    if clauses == None:
        if n_clauses == None:
            if problem == '3sat':
                n_clauses = 5*n
            else:  # exact cover
                n_clauses = n//2

        # generate clauses
        clauses = zeros((n_clauses, 3), int)
        for k in range(n_clauses):
            bits = list(range(n))
            for j in range(3):
                clauses[k, j] = bits.pop(npr.randint(len(bits))) + 1 # zero can't be negated, add one
        clauses = sort(clauses, 1)

        if problem == '3sat':
            for k in range(n_clauses):
                for j in range(3):
                    clauses[k, j] *= (-1) ** (npr.rand() < 0.5) # negate if bit should be inverted
    else:
        n_clauses = clauses.shape[0]

    # cache some stuff (all the matrices in this example are diagonal, so)
    zb  = array([0, 1], int) # 0.5*(I - sz)
    z_op = []
    for k in range(n):
        z_op.append(mkron(ones(2 ** k, int), zb, ones(2 ** (n-k-1), int)))

    print('{0} bits, {1} clauses'.format(n, n_clauses))
    # Encode the problem into the final Hamiltonian.
    H1 = 0
    if problem == '3sat':
        b = [0, 0, 0]
        for k in range(n_clauses):
            # h_c = (b1(*) v b2(*) v b3(*))
            for j in range(3):
                temp = clauses[k, j]
                b[j] = z_op[abs(temp) - 1] # into zero-based index
                if temp < 0:
                    b[j] = logical_not(b[j])
            H1 += logical_not(b[0] + b[1] + b[2]) # the not makes this a proper OR
        print('Find a bit string such that all the following "(b_i or b_j or b_k)" clauses are satisfied.')
        print('Minus sign means the bit is negated.')
    else:
        # exact cover
        for k in range(n_clauses):
            # h_c = (b1 ^ b2* ^ b3*) v (b1* ^ b2 ^ b3*) v (b1* ^ b2* ^ b3)
            b1 = z_op[clauses[k, 0] - 1]
            b2 = z_op[clauses[k, 1] - 1]
            b3 = z_op[clauses[k, 2] - 1]
            s1 = b1 * logical_not(b2) * logical_not(b3)
            s2 = b2 * logical_not(b3) * logical_not(b1)
            s3 = b3 * logical_not(b1) * logical_not(b2)
            H1 += logical_not(s1 + s2 + s3 -s1*s2 -s2*s3 -s3*s1 +s1*s2*s3)
        print('Find a bit string such that all the following "exactly one of bits {a, b, c} is 1" clauses are satisfied:')
    print(clauses)

    # build initial Hamiltonian
    xb = 0.5 * (eye(2) -sx)
    H0 = 0
    for k in range(n):
        H0 += mkron(eye(2 ** k), xb, eye(2 ** (n-k-1)))

    # initial state (ground state of H0)
    s0 = state(ones(2 ** n) / sqrt(2 ** n), qubits(n)) # n qubits, uniform superposition

    # adiabatic simulation
    adiabatic_qc(H0, H1, s0)
    return H1, clauses




def adiabatic_qc(H0, H1, s0, tmax=50):
    """Adiabatic quantum computing.

    This is a helper function for simulating the adiabatic quantum
    algorithm of :cite:`Farhi` and plotting the results.
    """
    # Ville Bergholm 2009-2011

    H1_full = diag(H1) # into a full matrix

    # adiabatic simulation
    steps = tmax*10
    t = linspace(0, tmax, steps)

    # linear path
    H_func = lambda t: (1-t/tmax)*H0 +(t/tmax)*H1_full
    res = s0.propagate(H_func, t)

    # plots
    # final state probabilities
    plt.figure()
    plot.adiabatic_evolution(t, res, H_func)

    plt.figure()
    res[-1].plot()
    plt.title('Final state')

    print('Final Hamiltonian (diagonal):')
    print(H1)

    print('Measured result:')
    dummy, dummy, res = res[-1].measure(do = 'C')
    print(res)
    if H1[nonzero(res.data)[0]] == 0:
        print('Which is a valid solution!')
    else:
        print('Which is not a solution!')
        if np.min(H1) > 0:
            print("(In this problem instance there aren't any.)")



def bb84(n=50):
    """Bennett-Brassard 1984 quantum key distribution protocol demo.

    Simulate the protocol with n qubits transferred.

    """
    # Ville Bergholm 2009-2012
    from .base import H

    print('\n\n=== BB84 protocol ===\n')
    print('Using {0} transmitted qubits, intercept-resend attack.\n'.format(n))

    # Alice generates two random bit vectors
    sent    = npr.rand(n) > 0.5
    basis_A = npr.rand(n) > 0.5

    # Bob generates one random bit vector
    basis_B = npr.rand(n) > 0.5
    received = zeros(n, bool)

    # Eve hasn't yet decided her basis
    basis_E = zeros(n, bool)
    eavesdrop = zeros(n, bool)

    print("""Alice transmits a sequence of qubits to Bob using a quantum channel.
For every qubit, she randomly chooses a basis (computational or diagonal)
and randomly prepares the qubit in either the |0> or the |1> state in that basis.""")
    print("\nbasis_A = {0}\nbits_A  = {1}".format(asarray(basis_A, dtype=int),
                                                  asarray(sent, dtype=int)))

    temp = state('0')
    for k in range(n):
        # Alice has a source of qubits in the zero state.
        q = temp

        # Should Alice flip the qubit?
        if sent[k]:  q = q.u_propagate(sx)

        # Should Alice apply a Hadamard?
        if basis_A[k]:  q = q.u_propagate(H)

        # Alice now sends the qubit to Bob...
        # ===============================================
        # ...but Eve intercepts it, and conducts an intercept/resend attack

        # Eve might have different strategies here... TODO
        # Eve's strategy (simplest choice, random)
        basis_E[k] = npr.rand() > 0.5
  
        # Eve's choice of basis: Hadamard?
        if basis_E[k]:  q = q.u_propagate(H)

        # Eve measures in the basis she has chosen
        _, res, q = q.measure(do = 'C')
        eavesdrop[k] = res

        # Eve tries to reverse the changes she made...
        if basis_E[k]:  q = q.u_propagate(H)

        # ...and sends the result to Bob.
        # ===============================================

        # Bob's choice of basis
        if basis_B[k]:  q = q.u_propagate(H)

        # Bob measures in the basis he has chosen, and discards the qubit.
        _, res = q.measure()
        received[k] = res

    #sum(xor(sent, eavesdrop))/n
    #sum(xor(sent, received))/n

    print("""\nHowever, there's an eavesdropper, Eve, on the line. She intercepts the qubits,
randomly measures them in either basis (thus destroying the originals!), and then sends
a new batch of qubits corresponding to her measurements and basis choices to Bob.
Since Eve on the average can choose the right basis only 50% of the time,
about 1/4 of her bits differ from Alice's.""")
    print("\nbasis_E = {0}\nbits_E  = {1}".format(asarray(basis_E, dtype=int),
                                                  asarray(eavesdrop, dtype=int)))

    print("\nWhen Bob receives the qubits, he randomly measures them in either basis.")
    print("\nbasis_B = {0}\nbits_B  = {1}".format(asarray(basis_B, dtype=int),
                                                  asarray(received, dtype=int)))

    print("""\nNow Bob announces on a public classical channel that he has received all the qubits.
Alice and Bob then reveal the bases they used. Whenever the bases happen to match,
(about 50% of the time on the average), they both add their corresponding bit to
their personal key. The two keys should be identical unless there's been an eavesdropper.""")

    match = np.logical_not(np.logical_xor(basis_A, basis_B))
    key_A = sent[match]
    key_B = received[match]
    m = len(key_A)
    print('\nmatch = {0}\nkey_A = {1}\nkey_B = {2}'.format(asarray(match, dtype=int),
                                                           asarray(key_A, dtype=int),
                                                           asarray(key_B, dtype=int)))
    print('\nMismatch frequency between Alice and Bob: {0}'.format(np.sum(np.logical_xor(key_A, key_B)) / m))

    print("""\nAlice and Bob then sacrifice k bits of their shared key to compare them.
If a nonmatching bit is found, the reason is either an eavesdropper or a noisy channel.
Since the probability for each eavesdropped bit to be wrong is 1/4, they will detect
Eve's presence with the probability 1-(3/4)^k.""")



def bernstein_vazirani(n=6, linear=True):
    r"""Bernstein-Vazirani algorithm demo.

    Simulates the Bernstein-Vazirani algorithm :cite:`BV`, which, given a black box oracle
    implementing a linear Boolean function :math:`f_a(x) := a \cdot x`, returns the bit
    vector a (and thus identifies the function) with just a single oracle call.
    If the oracle function is not linear, the algorithm will fail.
    """
    # Ville Bergholm 2011-2012

    print('\n\n=== Bernstein-Vazirani algorithm ===\n')

    print('Using {0} qubits.'.format(n))
    dim = qubits(n)
    H = gate.walsh(n) # n-qubit Walsh-Hadamard gate
    N = 2 ** n

    def oracle(f):
        """Returns a unitary oracle for the Boolean function f(x), given as a truth table."""
        return (-1) ** f.reshape((-1, 1))

    def linear_func(a):
        r"""Builds the linear Boolean function f(x) = a \cdot x as a truth table."""
        dim = qubits(len(a))
        N = prod(dim)
        U = empty(N, dtype = int)
        for k in range(N):
            x = np.unravel_index(k, dim)
            U[k] = mod(dot(a, x), 2)
        return U

    # black box oracle encoding the Boolean function f (given as the diagonal)
    if linear:
        # linear f
        a = asarray(npr.rand(n) > 0.5, dtype = int)
        f = linear_func(a)
        print('\nUsing the linear function f_a(x) := dot(a, x), defined by the binary vector a = {0}.'.format(a))
    else:
        # general f
        f = asarray(npr.rand(N) > 0.5, dtype = int)
        # special case: not(linear)
        #a = asarray(npr.rand(n) > 0.5, dtype = int)
        #f = 1-linear_func(a)
        print('\nNonlinear function f:\n{0}.'.format(f))
    U_oracle = oracle(f)

    # start with all-zero state
    s = state(0, dim)
    # initial superposition
    s = s.u_propagate(H)
    # oracle phase flip
    s.data = U_oracle * s.data
    # final Hadamards
    s = s.u_propagate(H)

    plt.figure()
    s.plot()
    title = 'Bernstein-Vazirani algorithm, '
    if linear:
        title += 'linear oracle'
    else:
        title += 'nonlinear oracle (fails)'
    plt.title(title)

    p, res = s.measure()
    # measured binary vector
    b = asarray(np.unravel_index(res, dim))
    print('\nMeasured binary vector b = {0}.'.format(b))
    if not(linear):
        g = linear_func(b)
        print('\nCorresponding linear function g_b:\n{0}.'.format(g))
        print('\nNormalized Hamming distance: |f - g_b| = {0}.'.format(sum(abs(f-g)) / N))

    return p



def grover_search(n=8):
    """Grover search algorithm demo.

    Simulate the Grover search algorithm :cite:`Grover` formulated using amplitude amplification
    in a system of n qubits.
    """
    # Ville Bergholm 2009-2010

    print('\n\n=== Grover search algorithm ===\n')

    A = gate.walsh(n) # Walsh-Hadamard gate for generating uniform superpositions
    N = 2 ** n # number of states

    sol = npr.randint(N)
    reps = int(pi/(4*asin(sqrt(1/N))))

    print('Using {0} qubits.'.format(n))
    print('Probability maximized by {0} iterations.'.format(reps))
    print('Correct solution: {0}'.format(sol))

    # black box oracle capable of recognizing the correct answer (given as the diagonal)
    # TODO an oracle that actually checks the solutions by computing (using ancillas?)
    U_oracle = ones((N, 1))
    U_oracle[sol, 0] = -1

    U_zeroflip = ones((N, 1))
    U_zeroflip[0, 0] = -1

    s = state(0, qubits(n))

    # initial superposition
    s = s.u_propagate(A)

    # Grover iteration
    for _ in range(reps):
        # oracle phase flip
        s.data = -U_oracle * s.data

        # inversion about the mean
        s = s.u_propagate(A.ctranspose())
        s.data = U_zeroflip * s.data
        s = s.u_propagate(A)

    p, res = s.measure()
    print('\nMeasured {0}.'.format(res))
    return p



def markov_decoherence(T1, T2, B=None):
    """Markovian decoherence demo.

    Given decoherence times T1 and T2, creates a markovian bath B
    and a coupling operator D which reproduce them on a single-qubit system.

    """
    # Ville Bergholm 2009-2011

    from . import markov
    print('\n\n=== Markovian decoherence in a qubit ===\n')

    omega0 = 2*pi* 1e9 # Hz
    T = 1 # K
    delta = 3 + 3*npr.rand() # qubit energy splitting (GHz)

    # setup the bath
    if B == None:
        B = markov.bath('ohmic', omega0, T) # defaults

    # find the correct qubit-bath coupling
    H, D = B.fit(delta, T1*omega0, T2*omega0)
    L = markov.superop(H, D, B)
    t = linspace(0, 10, 200)

    # T1 demo
    eq = 1 / (1 + exp(delta * B.scale)) # equilibrium rho_11
    s = state('1') # qubit in the |1> state
    out = s.propagate(L, t, lambda x, h: x.ev(p1))
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(t, out, 'r-', t, eq +(1-eq)*exp(-t/(T1*omega0)), 'b-.', [0, t[-1]], [eq, eq], 'k:', linewidth = 2)
    plt.xlabel('$t \\omega_0$')
    plt.ylabel('probability')
    plt.axis([0, t[-1], 0, 1])
    plt.title('$T_1$: relaxation')
    plt.legend(('simulated $P_1$', '$P_{1}^{eq} +(1-P_{1}^{eq}) \\exp(-t/T_1)$'))

    # T2 demo
    s = state('0')
    s = s.u_propagate(R_y(pi/2)) # rotate to (|0>+|1>)/sqrt(2)
    out = s.propagate(L, t, lambda x, h: x.u_propagate(R_y(-pi/2)).ev(p0))
    plt.subplot(2, 1, 2)
    plt.plot(t, out, 'r-', t, 0.5*(1+exp(-t/(T2*omega0))), 'b-.', linewidth = 2)
    plt.xlabel('$t \omega_0$')
    plt.ylabel('probability')
    plt.axis([0, t[-1], 0, 1])
    plt.title('$T_2$: dephasing')
    plt.legend(('simulated $P_0$', '$\\frac{1}{2} (1+\\exp(-t/T_2))$'))
    return H, D, B



def nmr_sequences(seqs=None, titles=None):
    """NMR control sequences demo.

    Compares the performance of different single-qubit NMR control
    sequences in the presence of systematic errors.
    Plots the fidelity of each control sequence as a function
    of both off-resonance error f and fractional pulse length error g.

    Reproduces fidelity plots in :cite:`Cummins`.
    """
    # Ville Bergholm 2006-2014

    from . import seq

    print('\n\n=== NMR control sequences for correcting systematic errors ===\n')

    if seqs == None:
        seqs = [seq.nmr([[pi, 0]]), seq.corpse(pi), seq.scrofulous(pi), seq.bb1(pi)]
        titles = [r'Plain $\pi$ pulse', 'CORPSE', 'SCROFULOUS', 'BB1']
    elif titles == None:
        titles = ['User-given seq'] * len(seqs)

    psi = state('0') # initial state
    f = arange(-1, 1, 0.05)
    g = arange(-1, 1, 0.08)
    nf = len(f)
    ng = len(g)

    def helper(ax, s_error, title):
        """Apply the sequence on the state psi, plot the evolution."""
        out, _ = seq.propagate(psi, s_error, out_func = state.bloch_vector)
        ax.set_title(title)
        plot.state_trajectory(out, ax = ax)


    for tt, ss in zip(titles, seqs):
        fig = plt.figure()
        gs = GridSpec(2, 2)
        U = seq.seq2prop(ss) # target propagator

        # The two systematic error types we are interested here can be
        # incorporated into the control sequence.
        #==================================================
        s_error = deepcopy(ss)
        s_error['A'] = ss['A'] + 0.1 * 0.5j * sz  # off-resonance error (constant \sigma_z drift term)
        ax = fig.add_subplot(gs[0, 0], projection = '3d')
        helper(ax, s_error, tt + ' evolution, off-resonance error')

        #==================================================
        s_error = deepcopy(ss)
        s_error['tau'] = ss['tau'] * 1.1  # proportional pulse length error
        ax = fig.add_subplot(gs[1, 0], projection = '3d')
        helper(ax, s_error, tt + ' evolution, pulse length error')

        #==================================================
        s_error = deepcopy(ss)
        fid = empty((ng, nf))

        def u_fidelity(a, b):
            """fidelity of two unitary rotations, [0,1]"""
            return 0.5 * abs(trace(dot(a.conj().transpose(), b)))

        for u in range(nf):
            s_error['A'] = ss['A'] + f[u] * 0.5j * sz  # off-resonance error
            for v in range(ng):
                s_error['tau'] = ss['tau'] * (1 + g[v])  # proportional pulse length error
                fid[v, u] = u_fidelity(U, seq.seq2prop(s_error))

        ax = fig.add_subplot(gs[:, 1])
        X, Y = meshgrid(f, g)
        ax.contour(X, Y, 1-fid)
        #ax.surf(X, Y, 1-fid)
        ax.set_xlabel('Off-resonance error')
        ax.set_ylabel('Pulse length error')
        ax.set_title(tt + ' fidelity')
        plt.show()



def phase_estimation(t, U, s, implicit=False):
    r"""Quantum phase estimation algorithm.

    Estimate an eigenvalue of the unitary :class:`lmap` U using t qubits,
    starting from the state s.

    Returns the state of the index register after the phase estimation
    circuit, but before final measurement.

    To get a result accurate to n bits with probability :math:`\ge 1 -\epsilon`,
    choose

    .. math::

       t \ge n + \left\lceil \log_2\left(2 +\frac{1}{2 \epsilon}\right) \right \rceil.

    See :cite:`Cleve`, :cite:`NC` chapter 5.2.
    """
    # Ville Bergholm 2009-2010

    T = 2 ** t
    S = U.data.shape[0]

    # index register in uniform superposition
    #reg_t = u_propagate(state(0, qubits(t)), gate.walsh(t)) # use Hadamards
    reg_t = state(ones(T) / sqrt(T), qubits(t)) # skip the Hadamards

    # state register (ignore the dimensions)
    reg_s = state(s, S)

    # full register
    reg = reg_t.tensor(reg_s)

    # controlled unitaries
    for k in range(t):
        ctrl = -ones(t)
        ctrl[k] = 1
        temp = gate.controlled(U ** (2 ** (t-1-k)), ctrl)
        reg = reg.u_propagate(temp)
    # from this point forward the state register is not used anymore

    if implicit:
        # save memory and CPU: make an implicit measurement of the state reg, discard the results
        _, _, reg = reg.measure(t, do = 'D')
        #print('Implicit measurement of state register: {0}\n', res)
    else:
        # more expensive computationally: trace over the state register
        reg = reg.ptrace((t,))

    # do an inverse quantum Fourier transform on the index reg
    QFT = gate.qft(qubits(t))
    return reg.u_propagate(QFT.ctranspose())



def phase_estimation_precision(t, U, u=None):
    """Quantum phase estimation demo.

    Estimate an eigenvalue of unitary operator U using t bits, starting from the state u.
    Plots and returns the probability distribution of the resulting t-bit approximations.
    If u is not given, use a random eigenvector of U.

    Uses :func:`phase_estimation`.
    """
    # Ville Bergholm 2009-2010

    print('\n\n=== Phase estimation ===\n\n')

    # find eigenstates of the operator
    N = U.shape[0]
    d, v = eig(U)
    if u == None:
        u = state(v[:, 0], N) # exact eigenstate

    print('Use {0} qubits to estimate the phases of the eigenvalues of a U({1}) operator.\n'.format(t, N))
    p = phase_estimation(t, lmap(U), u).prob()
    T = 2 ** t
    x = arange(T) / T
    w = 0.8 / T

    # plot probability distribution
    plt.figure()
    plt.hold(True)
    plt.bar(x, p, width = w) # TODO align = 'center' ???
    plt.xlabel(r'phase / $2\pi$')
    plt.ylabel('probability')
    plt.title('Phase estimation')
    #axis([-1/(T*2), 1-1/(T*2), 0, 1])

    # compare to correct answer
    target = angle(d) / (2*pi) + 1
    target -= floor(target)
    plt.plot(target, 0.5*max(p)*ones(len(target)), 'mo')

    plt.legend(('Target phases', 'Measurement probability distribution'))
    return p



def qft_circuit(dim=(2, 3, 3, 2)):
    r"""Quantum Fourier transform circuit demo.

    Simulate the quadratic QFT circuit construction.
    dim is the dimension vector of the subsystems.

    NOTE: If dim is not palindromic the resulting circuit also
    reverses the order of the dimensions in the SWAP cascade.

    .. math::

       U |x_1, x_2, \ldots, x_n\rangle
       = \frac{1}{\sqrt{d}} \sum_{k_i} |k_n,\ldots, k_2, k_1\rangle \exp\left(i 2 \pi \left(\sum_{r=1}^n k_r 0.x_r x_{r+1}\ldots x_n\right)\right)
       = \frac{1}{\sqrt{d}} \sum_{k_i} |k_n,\ldots, k_2, k_1\rangle \exp\left(i 2 \pi 0.x_1 x_2 \ldots x_n \left(\sum_{r=1}^n d_1 d_2 \cdots d_{r-1} k_r \right)\right).
    """
    # Ville Bergholm 2010-2011

    print('\n\n=== Quantum Fourier transform using a quadratic circuit ===\n')
    print('Subsystem dimensions: {0}'.format(dim))

    def Rgate(d):
        r"""R = \sum_{xy} exp(i*2*pi * x*y/prod(dim)) |xy><xy|"""
        temp = kron(arange(d[0]), arange(d[-1])) / prod(d)
        return gate.phase(2 * pi * temp, (d[0], d[-1]))

    n = len(dim)
    U = gate.id(dim)

    for k in range(n):
        H = gate.qft((dim[k],))
        U = gate.single(H, k, dim) * U
        for j in range(k+1, n):
            temp = Rgate(dim[k : j+1])
            U = gate.two(temp, (k, j), dim) * U

    for k in range(n // 2):
        temp = gate.swap(dim[k], dim[n-1-k])
        U = gate.two(temp, (k, n-1-k), dim) * U

    err = norm(U.data - gate.qft(dim).data)
    print('Error: {0}'.format(err))
    return U



def quantum_channels(p=0.3):
    """Visualization of simple one-qubit channels.

    Visualizes the effect of different quantum channels on a qubit using the Bloch sphere representation.
    """
    # Ville Bergholm 2009-2012

    print('\n\n=== Quantum channels ===\n')

    I = eye(2)
    E_bitflip      = [sqrt(1-p)*I, sqrt(p)*sx]
    E_phaseflip    = [sqrt(1-p)*I, sqrt(p)*sz]
    E_bitphaseflip = [sqrt(1-p)*I, sqrt(p)*sy]
    E_depolarize   = [sqrt(1-3*p/4)*I, sqrt(p)*sx/2, sqrt(p)*sy/2, sqrt(p)*sz/2]
    #t = arcsin(sqrt(gamma))
    t = pi/3
    E_amplitudedamp = [sqrt(p)*diag([1, cos(t)]), sqrt(p)*array([[0, sin(t)], [0, 0]]), sqrt(1-p)*diag([cos(t), 1]), sqrt(1-p)*array([[0, 0], [sin(t), 0]])]
    channels = [E_bitflip, E_phaseflip, E_bitphaseflip, E_depolarize, E_amplitudedamp]
    titles   = ['Bit flip', 'Phase flip', 'Bit and phase flip', 'Depolarization', 'Amplitude damping']

    X, Y, Z = plot.sphere()
    S = array([X, Y, Z])

    def present(ax, E, T):
        """Helper"""
        s = S.shape
        res = empty(s)

        for a in range(s[1]):
            for b in range(s[2]):
                temp = state.bloch_state(r_[1, S[:, a, b]])
                res[:, a, b] = temp.kraus_propagate(E).bloch_vector()[1:]  # skip the normalization

        plot.bloch_sphere(ax)
        ax.plot_surface(res[0], res[1], res[2], rstride = 1, cstride = 1, color = 'm', alpha = 0.2, linewidth = 0)
        ax.set_title(T)

    fig = plt.figure()
    plt.title('Quantum channels')
    n = len(channels)
    gs = GridSpec(2, int(ceil(n / 2)))
    for k in range(n):
        ax = fig.add_subplot(gs[k], projection = '3d')
        present(ax, channels[k], titles[k])

    plt.show()



def quantum_walk(steps=7, n=11, p=0.05, n_coin=2):
    """Quantum random walk demo.

    Simulates a 1D quantum walker controlled by a unitary quantum coin.
    On each step the coin is flipped and the walker moves either to the left
    or to the right depending on the result.

    After each step, the position of the walker is measured with probability p.
    p == 1 results in a fully classical random walk, whereas
    p == 0 corresponds to the "fully quantum" case.
    """
    # Ville Bergholm 2010-2011
    from .base import H

    print('\n\n=== Quantum walk ===\n')
    # initial state: coin shows heads, walker in center node
    coin   = state('0', n_coin)
    walker = state(n // 2, n)

    s = coin.tensor(walker).to_op(inplace = True)

    # translation operators (wrapping)
    left  = gate.mod_inc(-1, n)
    right = left.ctranspose()

    if n_coin == 2:
        # coin flip operator
        #C = R_x(pi / 2)
        C = H
        # shift operator: heads, move left; tails, move right
        S = kron(p0, left.data) +kron(p1, right.data)
    else:
        C = rand_U(n_coin)
        S = kron(diag([1, 0, 0]), left.data) +kron(diag([0, 1, 0]), right.data) +kron(diag([0, 0, 1]), eye(n))

    # propagator for a single step (flip + shift)
    U = dot(S, kron(C, eye(n)))

    # Kraus ops for position measurement
    M = []
    for k in range(n):
        temp = zeros((n, n))
        temp[k, k] = sqrt(p)
        M.append(kron(eye(n_coin), temp))

    # "no measurement"
    M.append(sqrt(1-p) * eye(n_coin * n))

    for k in range(steps):
        s = s.u_propagate(U)
        #s = s.kraus_propagate(M)
        s.data = p*diag(diag(s.data)) + (1-p)*s.data  # equivalent but faster...

    s = s.ptrace([0]) # ignore the coin
    plt.figure()
    s.plot()
    plt.title('Walker state after {0} steps'.format(steps))

    plt.figure()
    plt.bar(arange(n), s.prob(), width = 0.8)
    plt.title('Walker position probability distribution after {0} steps'.format(steps))
    return s


def qubit_and_resonator(d_r=30):
    """Qubit coupled to a microwave resonator demo.

    Simulates a qubit coupled to a microwave resonator.
    Reproduces plots from the experiment in :cite:`Hofheinz`.
    """
    # Ville Bergholm 2010-2014

    print('\n\n=== Qubit coupled to a single-mode microwave resonator ===\n')
    if d_r < 10:
        d_r = 10 # truncated resonator dim

    #omega0 = 1e9 # Energy scale/\hbar, Hz
    #T = 0.025 # K

    # omega_r = 2*pi* 6.570 # GHz, resonator angular frequency
    # qubit-resonator detuning Delta(t) = omega_q(t) -omega_r

    Omega     =  2*pi* 19e-3  # GHz, qubit-resonator coupling
    Delta_off = -2*pi* 463e-3 # GHz, detuning at off-resonance point
    Omega_q   =  2*pi* 0.5    # GHz, qubit microwave drive amplitude
    #  Omega << |Delta_off| < Omega_q << omega_r

    # decoherence times
    #T1_q = 650e-9 # s
    #T2_q = 150e-9 # s
    #T1_r = 3.5e-6 # s
    #T2_r = 2*T1_r # s

    # TODO heat bath and couplings
    #bq = markov.bath('ohmic', omega0, T)
    #[H, Dq] = fit(bq, Delta_off, T1_q, T2_q)
    #[H, Dr] = fit_ho(bq, ???, T1_r, T2_r???)
    #D = kron(eye(d_r), D) # +kron(Dr, qit.I)

    #=================================
    # Hamiltonians etc.

    # qubit raising and lowering ops
    sp = 0.5*(sx -1j*sy)
    sm = sp.conj().transpose()

    # resonator annihilation op
    a = boson_ladder(d_r)
    # resonator identity op
    I_r = eye(d_r)

    # qubit H
    Hq = kron(I_r, dot(sp, sm))
    # resonator H
    #Hr = kron(a'*a, I_q)

    # coupling H, rotating wave approximation
    Hint = Omega/2 * (kron(a, sp) +kron(a.conj().transpose(), sm))
    # microwave control H
    HOq = lambda ampl, phase: kron(I_r, ampl * 0.5 * Omega_q * (cos(phase) * sx + sin(phase) * sy))
    #Q = ho.position(d_r)
    #P = ho.momentum(d_r)
    #HOr = lambda ampl, phase: kron(ampl*Omega_r/sqrt(2)*(cos(phase)*Q +sin(phase)*P), qit.I)

    # system Hamiltonian, in rotating frame defined by H0 = omega_r * (Hq + Hr)
    # D = omega_q - omega_r is the detuning between the qubit and the resonator
    H = lambda D, ampl, phase:  D*Hq + Hint + HOq(ampl, phase)

    # readout: qubit in excited state?
    readout = lambda s, h: s.ev(kron(I_r, p1))

    s0 = state(0, (d_r, 2)) # resonator + qubit, ground state


    #=================================
    # Rabi test

    t = linspace(0, 500, 100)
    detunings = linspace(0, 2*pi* 40e-3, 100) # GHz
    out = empty((len(detunings), len(t)))

    #L = markov.superop(H(0, 1, 0), D, bq)
    L = H(0, 1, 0)  # zero detuning, sx pulse
    for k, d in enumerate(detunings):
        s = s0.propagate(L, (2/Omega_q)*pi/2) # Q (pi pulse for the qubit)
        #LL = markov.superop(H(d, 0, 0), D, bq)
        LL = H(d, 0, 0)  # detuned propagation
        out[k, :] = s.propagate(LL, t, out_func = readout)

    plt.figure()
    plot.pcolor(out, t, detunings / (2*pi*1e-3))
    #plt.colorbar(orientation = 'horizontal')
    plt.xlabel('Interaction time $\\tau$ (ns)')
    plt.ylabel('Detuning, $\\Delta/(2\\pi)$ (MHz)')
    plt.title('One photon Rabi-swap oscillations between qubit and resonator, $P_e$')

    #figure
    #f = fft(out, [], 2)
    #pcolor(abs(fftshift(f, 2)))
    #=================================
    # state preparation

    def demolish_state(targ):
        """Convert a desired (possibly truncated) resonator state ket into a program for constructing that state.
        State preparation in reverse, uses approximate idealized Hamiltonians.
        """
        # Ideal H without interaction
        A = lambda D, ampl, phase: D*Hq +HOq(ampl, phase)

        # resonator ket into a full normalized qubit+resonator state
        n = len(targ)
        targ = state(r_[targ, zeros(d_r-n)], (d_r,)).normalize().tensor(state(0, (2,)))
        t = deepcopy(targ)

        n = n-1 # highest excited level in resonator
        prog = zeros((n, 4))
        for k in reversed(range(n)):
            # |k+1,g> to |k,e> 
            dd = targ.data.reshape(d_r, 2)
            prog[k, 3] = (angle(dd[k, 1]) -angle(dd[k+1, 0]) -pi/2 +2*pi) / -Delta_off
            targ = targ.propagate(A(-Delta_off, 0, 0), prog[k, 3]) # Z

            dd = targ.data.reshape(d_r, 2)
            prog[k, 2] = (2/(sqrt(k+1) * Omega)) * arctan2(abs(dd[k+1, 0]), abs(dd[k, 1]))
            targ = targ.propagate(-Hint, prog[k, 2]) # S

            # |k,e> to |k,g>
            dd = targ.data.reshape(d_r, 2)
            phi = angle(dd[k, 1]) -angle(dd[k, 0]) +pi/2
            prog[k, 1] = phi
            prog[k, 0] = (2/Omega_q) * arctan2(abs(dd[k, 1]), abs(dd[k, 0]))
            targ = targ.propagate(A(0, -1, phi), prog[k, 0]) # Q
        return prog, t

    def prepare_state(prog):
        """Prepare a state according to the program."""
        s = s0 # start with ground state
        #s = tensor(ho.coherent_state(0.5, d_r), state(0, 2)) # coherent state
        for k in prog:
            # Q, S, Z
            s = s.propagate(H(0, 1, k[1]), k[0]) # Q
            s = s.propagate(H(0, 0, 0), k[2]) # S
            s = s.propagate(H(Delta_off, 0, 0), k[3]) # Z
        return s


    #=================================
    # readout plot (not corrected for limited visibility)

    prog, dummy = demolish_state([0, 1, 0, 1]) # |1> + |3>
    s1 = prepare_state(prog)

    prog, dummy = demolish_state([0, 1, 0, 1j]) # |1> + i|3>
    s2 = prepare_state(prog)

    t = linspace(0, 350, 200)
    out1 = s1.propagate(H(0, 0, 0), t, readout)
    out2 = s2.propagate(H(0, 0, 0), t, readout)

    plt.figure()
    plt.plot(t, out1, 'b-', t, out2, 'r-')
    plt.xlabel('Interaction time $\\tau$ (ns)')
    plt.ylabel('$P_e$')
    plt.title('Resonator readout through qubit.')
    plt.legend(('$|1\\rangle + |3\\rangle$', '$|1\\rangle + i|3\\rangle$'))


    #=================================
    # Wigner spectroscopy

    if False:
        # "voodoo cat"
        targ = zeros(d_r)
        for k in range(0, d_r, 3):
            targ[k] = (2 ** k) / sqrt(factorial(k))
    else:
        targ = [1, 0, 0, exp(1j * pi * 3/8), 0, 0, 1]

    # calculate the pulse sequence for constructing targ
    prog, t = demolish_state(targ)
    s = prepare_state(prog)

    print('Trying to prepare the state')
    print(t)
    print('Fidelity of prepared state with target state: {0:g}'.format(fidelity(s, t)))
    print('Time required for state preparation: {0:g} ns'.format(np.sum(prog[:, [0, 2, 3]])))
    print('\nComputing the Wigner function...')
    s = s.ptrace((1,))
    W, a, b = ho.wigner(s, res = (80, 80), lim = (-2.5, 2.5, -2.5, 2.5))
    plt.figure()
    plot.pcolor(W, a, b, (-1, 1))
    plt.title('Wigner function $W(\\alpha)$')
    plt.xlabel('Re($\\alpha$)')
    plt.ylabel('Im($\\alpha$)')



def shor_factorization(N=9, cheat=False):
    """Shor's factorization algorithm demo.

    Simulates Shor's factorization algorithm, tries to factorize the integer N.
    If cheat is False, simulates the full algorithm. Otherwise avoids the quantum part.

    NOTE: This is a very computationally intensive quantum algorithm
    to simulate classically, and probably will not run for any
    nontrivial value of N (unless you choose to cheat, in which case
    instead of simulating the quantum part (implemented in :func:`find_order`)
    we use a more efficient classical algorithm for the order-finding).

    See :cite:`Shor`, :cite:`NC` chapter 5.3.
    """
    # Ville Bergholm 2010-2011

    def find_order_cheat(a, N):
        """Classical order-finding algorithm."""
        for r in range(1, N+1):
            if mod_pow(a, r, N) == 1:
                return r

    def mod_pow(a, x, N):
        """Computes mod(a^x, N) using repeated squaring mod N.
        x must be a positive integer.
        """
        X = numstr_to_array(bin(x)[2:]) # exponent in big-endian binary
        res = 1
        for b in reversed(X): # little-endian
            if b:
                res = mod(res*a, N)
            a = mod(a*a, N) # square a
        return res


    print('\n\n=== Shor\'s factorization algorithm ===\n')
    if cheat:
        print('(cheating)\n')

    # number of bits needed to represent mod N arithmetic:
    m = int(ceil(log2(N)))
    print('Trying to factor N = {0} ({1} bits).'.format(N, m))

    # maximum allowed failure probability for the quantum order-finding part
    epsilon = 0.25
    # number of index qubits required for the phase estimation
    t = 2*m +1 +int(ceil(log2(2 + 1 / (2 * epsilon))))
    print('The quantum order-finding subroutine will need {0} + {1} qubits.\n'.format(t, m))

    # classical reduction of factoring to order-finding
    while True:
        a = npr.randint(2, N) # random integer, 2 <= a < N
        print('Random integer: a = {0}'.format(a))
  
        p = gcd(a, N)
        if p != 1:
            # a and N have a nontrivial common factor p.
            # This becomes extremely unlikely as N grows.
            print('Lucky guess, we found a common factor!')
            break

        print('Trying to find the period of f(x) = a^x mod N')

        if cheat:
            # classical cheating shortcut
            r = find_order_cheat(a, N)
        else:
            while True:
                print('.')
                # ==== quantum part of the algorithm ====
                [s1, r1] = find_order(a, N, epsilon)
                [s2, r2] = find_order(a, N, epsilon)
                # =============  ends here  =============

                if gcd(s1, s2) == 1:
                    # no common factors
                    r = lcm(r1, r2)
                    break

        print('\n  =>  r = {0}'.format(r))

        # if r is odd, try again
        if gcd(r, 2) == 2:
            # r is even

            x = mod_pow(a, r // 2, N)
            if mod(x, N) != N-1:
                # factor found
                p = gcd(x-1, N) # ok?
                if p == 1:
                    p = gcd(x+1, N) # no, try this
                break
            else:
                print('a^(r/2) = -1 (mod N), try again...\n')
        else:
            print('r is odd, try again...\n')

    print('\nFactor found: {0}'.format(p))
    return p


def find_order(a, N, epsilon=0.25):
    """Quantum order-finding subroutine.

    Finds the period of the function f(x) = a^x mod N.
    epsilon is the maximum allowed failure probability for the subroutine.

    Returns r, s where r/s approximates period/T.

    Uses :func:`phase_estimation`.

    See :cite:`Shor`, :cite:`NC` chapter 5.3.
    """
    # number of bits needed to represent mod N arithmetic:
    m = int(ceil(log2(N)))
    # number of index qubits required for the phase estimation
    t = 2*m +1 +int(ceil(log2(2 + 1 / (2 * epsilon))))

    T = 2 ** t # index register dimension
    M = 2 ** m # state register dimension

    # applying f(x) is equivalent to the sequence of controlled modular multiplications in phase estimation
    U = gate.mod_mul(a, M, N)

    # state register initialized in the state |1>
    st = state(1, M)

    # run the phase estimation algorithm
    reg = phase_estimation(t, U, st, implicit = True) # use implicit measurement to save memory

    # measure index register
    dummy, num = reg.measure()

    def find_denominator(x, y, max_den):
        r"""
        Finds the denominator s for r/s \approx x/y such that s < max_den
        using the convergents of the continued fraction representation

        .. math::

           \frac{x}{y} = a_0 +\frac{1}{a_1 +\frac{1}{a_2 +\ldots}}

        We use floor and mod here, which could be both efficiently implemented using
        the classical Euclidean algorithm.
        """
        d_2 = 1
        d_1 = 0
        while True:
            a = x // y # integer part == a_n
            temp = a*d_1 +d_2 # n:th convergent denumerator d_n = a_n*d_{n-1} +d_{n-2}
            if temp >= max_den:
                break

            d_2 = d_1
            d_1 = temp
            temp = mod(x, y)  # x - a*y # subtract integer part
            if temp == 0:
            #if (temp/y < 1 / (2*max_den ** 2))
                break  # continued fraction terminates here, result is exact

            # invert the remainder (swap numerator and denominator)
            x = y
            y = temp
        return d_1

    # another classical part
    s = find_denominator(num, T, T+1)
    r = (num * s) // T
    return r, s


def superdense_coding(d=2):
    """Superdense coding demo.

    Simulate Alice sending two d-its of information to Bob using 
    a shared EPR qudit pair.

    """
    # Ville Bergholm 2010-2011

    print('\n\n=== Superdense coding ===\n')

    H   = gate.qft(d)        # qft (generalized Hadamard) gate
    add = gate.mod_add(d, d) # modular adder (generalized CNOT) gate
    I   = gate.id(d)

    dim = (d, d)

    # EPR preparation circuit
    U = add * tensor(H, I)

    print('Alice and Bob start with a shared EPR pair:')
    reg = state('00', dim).u_propagate(U)
    print(reg)

    # two random d-its
    a = floor(d * npr.rand(2)).astype(int)
    print('\nAlice wishes to send two d-its of information (d = {0}) to Bob: a = {1}.'.format(d, a))

    Z = H * gate.mod_inc(a[0], d) * H.ctranspose()
    X = gate.mod_inc(-a[1], d)

    print('Alice encodes the d-its to her half of the EPR pair using local transformations,')
    reg = reg.u_propagate(tensor(Z*X, I))
    print(reg)

    print('\nand sends it to Bob. He then disentangles the pair,')
    reg = reg.u_propagate(U.ctranspose())
    print(reg)

    _, b = reg.measure()
    b = array(np.unravel_index(b, dim))
    print('\nand measures both qudits in the computational basis, obtaining the result  b = {0}.'.format(b))

    if all(a == b):
        print('The d-its were transmitted succesfully.')
    else:
        raise RuntimeError('Should not happen.')


def teleportation(d=2):
    """Quantum teleportation demo.

    Simulate the teleportation of a d-dimensional qudit from Alice to Bob.

    """
    # Ville Bergholm 2009-2011

    print('\n\n=== Quantum teleportation ===\n')

    H   = gate.qft(d)        # qft (generalized Hadamard) gate
    add = gate.mod_add(d, d) # modular adder (generalized CNOT) gate
    I   = gate.id(d)

    dim = (d, d)

    # EPR preparation circuit
    U = add * tensor(H, I)

    print('Alice and Bob start with a shared EPR pair:')
    epr = state('00', dim).u_propagate(U)
    print(epr)

    print('\nAlice wants to transmit this payload to Bob:')
    payload = state('0', d).u_propagate(rand_SU(d)).fix_phase()
    # choose a nice global phase
    print(payload)

    print('\nThe total |payload> \otimes |epr> register looks like')
    reg = payload.tensor(epr)
    print(reg)

    print('\nNow Alice entangles the payload with her half of the EPR pair,')
    reg = reg.u_propagate(tensor(U.ctranspose(), I))
    print(reg)

    _, b, reg = reg.measure((0, 1), do = 'C')
    b = array(np.unravel_index(b, dim))
    print('\nand measures her qudits, getting the result {0}.'.format(b))
    print('She then transmits the two d-its to Bob using a classical channel.')
    print('Since Alice\'s measurement has unentangled the state,')
    print('Bob can ignore her qudits. His qudit now looks like')
    reg_B = reg.ptrace((0, 1)).to_ket() # pure state
    print(reg_B)

    print('\nUsing the two classical d-its of data Alice sent him,')
    print('Bob performs a local transformation on his half of the EPR pair.')
    Z = H * gate.mod_inc(b[0], d) * H.ctranspose()
    X = gate.mod_inc(-b[1], d)
    reg_B = reg_B.u_propagate(Z*X).fix_phase()
    print(reg_B)

    ov = fidelity(payload, reg_B)
    print('\nThe overlap between the resulting state and the original payload state is |<payload|B>| = {0}'.format(ov))
    if abs(ov-1) > tol:
        raise RuntimeError('Should not happen.')
    else:
        print('The payload state was succesfully teleported from Alice to Bob.')
    return reg_B, payload



def tour():
    """Guided tour to the quantum information toolkit.

    """
    # Ville Bergholm 2009-2014

    print('This is the guided tour for the Quantum Information Toolkit.')
    print("It should be run in the interactive mode, 'ipython --pylab'.")
    print('Between examples, press any key to proceed to the next one.')

    def pause():
        plt.waitforbuttonpress()

    temp = plt.figure()
    teleportation()
    pause()
    superdense_coding()
    pause()
    plt.close(temp)

    adiabatic_qc_3sat(5, 25)
    pause()

    phase_estimation_precision(5, rand_U(4))
    plt.title('Phase estimation, eigenstate')
    phase_estimation_precision(5, rand_U(4), state(0, [4]))
    plt.title('Phase estimation, random state')
    pause()

    nmr_sequences()
    pause()

    quantum_channels()
    pause()

    bernstein_vazirani(6, linear = True)
    bernstein_vazirani(6, linear = False)
    pause()

    grover_search(6)
    pause()

    # shor_factorization(9) # TODO need to use sparse matrices to get even this far(!)
    shor_factorization(311*269, cheat = True)
    pause()

    bb84(40)
    pause()

    markov_decoherence(7e-10, 1e-9)
    pause()

    qubit_and_resonator(20)
    pause()

    qft_circuit((2, 3, 2))

    #quantum_walk()
