# -*- coding: utf-8 -*-
"""
Quantum gates (:mod:`qit.gate`)
===============================

This module implements many common types of quantum gates (and some
other useful linear maps). The returned gates are represented as sparse :class:`lmap` instances.
The data type is float unless complex entries are actually needed.

.. currentmodule:: qit.gate


Contents
--------

.. autosummary::

   dist
   id
   phase
   qft
   swap
   walsh
   mod_inc
   mod_mul
   mod_add
   controlled
   single
   two
"""

from __future__ import division, absolute_import, print_function, unicode_literals

from numpy import pi, prod, empty, zeros, trace, exp, sqrt, mod, isscalar, kron, array, ones, array_equal
import scipy.sparse as sparse

from .lmap import lmap, tensor
from .utils import qubits, op_list, assert_o, copy_memoize, gcd


__all__ = ['dist', 'id', 'phase', 'qft', 'swap', 'walsh',
           'mod_add', 'mod_inc', 'mod_mul', 
           'controlled', 'single', 'two']

# TODO reshape will cause problems for sparse matrices!
# TODO utils.op_list too!
# TODO which one is faster in element assignment -style init, dok or lil?
# TODO make input interface consistent, do we want arrays or lmaps?

def eye(D):
    """FIXME Temp. wrapper, unnecessary after when we update to scipy 0.12
    Then we can just do   from scipy.sparse import eye
    """
    return sparse.eye(D, D) 


def dist(A, B):
    r"""Distance between two unitary lmaps.

    Returns

    .. math::

       \inf_{\phi \in \mathbb{R}} \|A - e^{i \phi} B\|_F^2
       = 2 (\dim_A - |\mathrm{Tr}(A^\dagger B)|)
    """
    # Ville Bergholm 2007-2010

    if not A.is_compatible(B):
        raise ValueError('The lmaps have different dimensions.')

    temp = A.ctranspose() * B
    return 2 * (prod(temp.dim[0]) - abs(trace(temp.data)))



def id(dim):
    """Identity gate.

    Returns the identity gate I for the specified system.
    dim is a tuple of subsystem dimensions.
    """
    if isscalar(dim):
        dim = (dim,)  # scalar into a tuple
    return lmap(eye(prod(dim)), (dim, dim))


def mod_add(dim1, dim2, N=None):
    r"""Modular adder gate.

    Returns the gate U, which, operating on the computational state
    :math:`|x, y\rangle`, produces
    :math:`|x, y+x (\mod N)\rangle`.
    dim1 and dim2 are the control and target register dimension vectors.

    By default N == prod(dim2).
    If N is explicitly given we must have N <= prod(dim2), and U will act trivially on target states >= N.

    Notes:
    The modular subtractor gate can be obtained by taking the
    Hermitian conjugate of mod_add.
    mod_add(2, 2) is equal to CNOT.
    """
    # Ville Bergholm 2010

    d1 = prod(dim1)
    d2 = prod(dim2)
    if N == None:
        N = d2
    elif d2 < N:
        raise ValueError('Target register dimension must be >= N.')

    # NOTE: a real quantum computer would implement this gate using a
    # sequence of reversible arithmetic gates but since we don't have
    # one we might as well cheat
    dim = d1 * d2
    U = sparse.dok_matrix((dim, dim))
    for a in range(d1):
        for b in range(d2):
            y = d2*a + b
            if b < N:
                x = d2*a +mod(a+b, N)
            else:
                # U acts trivially for target states >= N
                x = y
            U[x, y] = 1

    dim = (dim1, dim2)
    return lmap(U.tocsr(), (dim, dim))


def mod_inc(x, dim, N=None):
    r"""Modular incrementation gate.

    U = mod_inc(x, dim)     N == prod(dim)
    U = mod_inc(x, dim, N)  gate dimension prod(dim) must be >= N

    Returns the gate U, which, operating on the computational state
    :math:`|y\rangle`, increments it by x (mod N):
    :math:`U |y\rangle = |y+x (mod N)\rangle`.

    If N is given, U will act trivially on computational states >= N.
    """
    # Ville Bergholm 2010

    if isscalar(dim):
        dim = (dim,)  # scalar into a tuple
    d = prod(dim)
    if N == None:
        N = d
    elif d < N:
        raise ValueError('Gate dimension must be >= N.')

    U = sparse.dok_matrix((d, d))
    for y in range(N):
        U[mod(x+y, N), y] = 1
    # U acts trivially for states >= N
    for y in range(N, d):
        U[y, y] = 1

    return lmap(U.tocsr(), (dim, dim))


def mod_mul(x, dim, N=None):
    r"""Modular multiplication gate.

    U = mod_mul(x, dim)     N == prod(dim)
    U = mod_mul(x, dim, N)  gate dimension prod(dim) must be >= N

    Returns the gate U, which, operating on the computational state
    :math:`|y\rangle`, multiplies it by x (mod N):
    :math:`U |y\rangle = |x*y (mod N)\rangle`.
    x and N must be coprime for the operation to be reversible.

    If N is given, U will act trivially on computational states >= N.
    """
    # Ville Bergholm 2010-2011

    if isscalar(dim):
        dim = (dim,)  # scalar into a tuple
    d = prod(dim)
    if N == None:
        N = d
    elif d < N:
        raise ValueError('Gate dimension must be >= N.')

    if gcd(x, N) != 1:
        raise ValueError('x and N must be coprime for the mul operation to be reversible.')

    # NOTE: a real quantum computer would implement this gate using a
    # sequence of reversible arithmetic gates but since we don't have
    # one we might as well cheat
    U = sparse.dok_matrix((d, d))
    for y in range(N):
        U[mod(x*y, N), y] = 1
    # U acts trivially for states >= N
    for y in range(N, d):
        U[y, y] = 1

    return lmap(U.tocsr(), (dim, dim))


def phase(theta, dim=None):
    """Diagonal phase shift gate.

    Returns the (diagonal) phase shift gate U = diag(exp(i*theta)).
    """
    # Ville Bergholm 2011

    if isscalar(dim):
        dim = (dim,)  # scalar into a tuple
    n = len(theta)
    if dim == None:
        dim = (n,)
    d = prod(dim)
    if d != n:
        raise ValueError('Dimension mismatch.')

    return lmap(sparse.diags(exp(1j * theta), 0) , (dim, dim))


@copy_memoize
def qft(dim):
    """Quantum Fourier transform gate.

    Returns the quantum Fourier transform gate for the specified system.
    dim is a vector of subsystem dimensions.
    The returned lmap is dense.
    """
    # Ville Bergholm 2004-2011

    if isscalar(dim):
        dim = (dim,)  # scalar into a tuple
    N = prod(dim)
    U = empty((N, N), complex)  # completely dense, so we don't have to initialize it with zeros
    for j in range(N):
        for k in range(N):
            U[j, k] = exp(2j * pi * j * k / N) / sqrt(N)
    return lmap(U, (dim, dim))


def swap(d1, d2):
    r"""Swap gate.

    Returns the swap gate which swaps the order of two subsystems with dimensions [d1, d2].

    .. math::

       S: A_1 \otimes A_2 \to A_2 \otimes A_1, \quad
       v_1 \otimes v_2 \mapsto v_2 \otimes v_1
    """
    # Ville Bergholm 2010

    temp = d1*d2
    U = sparse.dok_matrix((temp, temp))
    for x in range(d1):
        for y in range(d2):
            U[d1*y + x, d2*x + y] = 1
    return lmap(U.tocsr(), ((d2, d1), (d1, d2)))


def walsh(n):
    """Walsh-Hadamard gate.

    Returns the Walsh-Hadamard gate for n qubits.
    The returned lmap is dense.
    """
    # Ville Bergholm 2009-2010

    from .base import H

    U = 1
    for _ in range(n):
        U = kron(U, H)
    dim = qubits(n)
    return lmap(U, (dim, dim))


def controlled(U, ctrl=(1,), dim=None):
    r"""Controlled gate.

    Returns the (t+1)-qudit controlled-U gate, where t == length(ctrl).
    U has to be a square matrix.

    ctrl is an integer vector defining the control nodes. It has one entry k per
    control qudit, denoting the required computational basis state :math:`|k\rangle`
    for that particular qudit. Value k == -1 denotes no control.

    dim is the dimensions vector for the control qudits. If not given, all controls
    are assumed to be qubits.

    Examples::

      controlled(NOT, [1]) gives the standard CNOT gate.
      controlled(NOT, [1, 1]) gives the Toffoli gate.
    """
    # Ville Bergholm 2009-2011

    # TODO generalization, uniformly controlled gates?
    if isscalar(dim):
        dim = (dim,)  # scalar into a tuple
    t = len(ctrl)
    if dim == None:
        dim = qubits(t) # qubits by default

    if t != len(dim):
        raise ValueError('ctrl and dim vectors have unequal lengths.')

    if any(array(ctrl) >= array(dim)):
        raise ValueError('Control on non-existant state.')

    yes = 1  # just the diagonal
    for k in range(t):
        if ctrl[k] >= 0:
            temp = zeros(dim[k])
            temp[ctrl[k]] = 1  # control on k
            yes = kron(yes, temp) 
        else:
            yes = kron(yes, ones(dim[k])) # no control on this qudit

    no = 1 - yes
    T = prod(dim)
    dim = list(dim)

    if isinstance(U, lmap):
        d1 = dim + list(U.dim[0])
        d2 = dim + list(U.dim[1])
        U = U.data
    else:
        d1 = dim + [U.shape[0]]
        d2 = dim + [U.shape[1]]

    # controlled gates only make sense for square matrices U (we need an identity transformation for the 'no' cases!)
    U_dim = U.shape[0]
    out = sparse.diags(kron(no, ones(U_dim)), 0) +sparse.kron(sparse.diags(yes, 0), U)
    return lmap(out, (d1, d2))


def single(L, t, d_in):
    """Single-qudit operator.

    Returns the operator U corresponding to the local operator L applied
    to subsystem t (and identity applied to the remaining subsystems).

    d_in is the input dimension vector for U.
    """
    # James Whitfield 2010
    # Ville Bergholm 2010

    if isinstance(L, lmap):
        L = L.data  # into ndarray

    d_in = list(d_in)
    if d_in[t] != L.shape[1]:
        raise ValueError('Input dimensions do not match.')
    d_out = d_in
    d_out[t] = L.shape[0]
    return lmap(op_list([[[L, t]]], d_in), (d_out, d_in))


def two(B, t, d_in):
    """Two-qudit operator.

    Returns the operator U corresponding to the bipartite operator B applied
    to subsystems t == [t1, t2] (and identity applied to the remaining subsystems).

    d_in is the input dimension vector for U.
    """
    # James Whitfield 2010
    # Ville Bergholm 2010-2011

    if len(t) != 2:
        raise ValueError('Exactly two target subsystems required.')

    n = len(d_in)
    t = array(t)
    if any(t < 0) or any(t >= n) or t[0] == t[1]:
        raise ValueError('Bad target subsystem(s).')

    d_in = array(d_in)
    if not array_equal(d_in[t], B.dim[1]):
        raise ValueError('Input dimensions do not match.')

    # dimensions for the untouched subsystems
    a = min(t)
    b = max(t)
    before    = prod(d_in[:a])
    inbetween = prod(d_in[a+1:b])
    after     = prod(d_in[b+1:])

    # how tensor(B_{01}, I_2) should be reordered
    if t[0] < t[1]:
        p = [0, 2, 1]
    else:
        p = [1, 2, 0]
    U = tensor(B, lmap(eye(inbetween))).reorder((p, p), inplace = True)
    U = tensor(lmap(eye(before)), U, lmap(eye(after)))

    # restore dimensions
    d_out = d_in.copy()
    d_out[t] = B.dim[0]
    return lmap(U, (d_out, d_in))
