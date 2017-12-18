# -*- coding: utf-8 -*-
"""
Quantum states (:mod:`qit.state`)
=================================


In QIT, quantum states are represented by the :class:`state` class,
defined in this module.


.. currentmodule:: qit.state.state


Utilities
---------

.. autosummary::

   check
   subsystems
   dims
   clean_selection
   invert_selection
   fix_phase
   normalize
   to_ket
   to_op
   trace
   ptrace
   ptranspose
   reorder
   tensor
   plot


Physics
-------

.. autosummary::

   ev
   var
   prob
   projector
   u_propagate
   propagate
   kraus_propagate
   measure
   

Quantum information
-------------------

.. autosummary::

   fidelity
   trace_dist
   purity
   schmidt
   entropy
   concurrence
   negativity
   lognegativity
   scott
   locc_convertible


Other state representations
---------------------------

.. autosummary::

   bloch_vector
   bloch_state
"""
# Ville Bergholm 2008-2012

from __future__ import division, absolute_import, print_function, unicode_literals

import collections
import numbers
import types
from copy import deepcopy

import numpy as np
from numpy import (array, asarray, empty, diag, sort, prod, cumsum, cumprod, exp, sqrt, trace,
    dot, vdot, roll, zeros, ones, r_, kron, isscalar, nonzero, ix_, linspace, meshgrid)
from numpy.random import rand, randn
import scipy as sp
from scipy.linalg import norm, eigh, eigvalsh, sqrtm, svd, svdvals, det
from scipy.integrate import ode

from .base import sy, Q_Bell, tol
from .lmap import tensor as lmap_tensor
from .lmap import numstr_to_array, array_to_numstr, lmap
from .utils import (_warn, vec, inv_vec, qubits, expv, rand_U, rand_SU, rand_positive, mkron,
    eighsort, spectral_decomposition, majorize, tensorbasis, assert_o)


__all__ = ['equal_dims', 'index_muls', 'state', 'fidelity', 'trace_dist']



def equal_dims(s, t):
    """True if s and t have equal dimensions."""
    return s.dims() == t.dims()


def index_muls(dim):
    """Index multipliers for C-ordered data"""
    # ravel_multi_index(s, dim) == dot(index_muls(dim), s)
    if len(dim) == 0:
        return array(())
    muls = roll(cumprod(dim[::-1]), 1)[::-1]
    muls[-1] = 1  # muls == [d_{n-1}*...*d_1, d_{n-1}*...*d_2, ..., d_{n-1}, 1]
    return muls



class state(lmap):
    """Class for quantum states.

    Describes the state (pure or mixed) of a discrete, possibly composite quantum system.
    The subsystem dimensions can be obtained with :func:`dims` (big-endian ordering).

    State class instances are special cases of lmaps. They have exactly two indices.
    If self.dim[1] == (1,), it is a ket representing a pure state.
    Otherwise both indices must have equal dimensions and the object represents a state operator.

    Does not require the state to be physical (it does not have to be trace-1, Hermitian, or nonnegative).
    """
    # Ville Bergholm 2008-2011
    # by default, all state methods leave self unchanged 

    def __init__(self, s, dim=None):
        """Construct a state.

        calling syntax            result
        ==============            ======
        state('00101')            standard basis ket |00101> in a five-qubit system
        state('02', (2, 3))       standard basis ket |02> in a qubit+qutrit system
        state(k, (2, 3))          linearized standard basis ket |k> in a qubit+qutrit system, k must be an integer scalar
        state(rand(4))            ket, infer dim[0] == (4,)
        state(rand(4), (2, 2))    ket, two qubits
        state(rand(4,4))          state operator, infer dim[0] == (4,)
        state(rand(6,6), (3, 2))  state operator, qutrit+qubit
        state('GHZ', (2, 2, 2))   named states (in this case the three-qubit GHZ state)

        state(s)                  (s is a state) copy constructor
        state(s, dim)             (s is a state) copy constructor, redefine the dimensions

        The currently supported named states are
          GHZ (Greenberger-Horne-Zeilinger),
          W,
          Bell1, Bell2, Bell3, Bell4 
        """
        from six import string_types

        # we want a tuple for dim
        if isinstance(dim, collections.Iterable):
            dim = tuple(dim)
        elif isscalar(dim):
            dim = (dim,)

        if isinstance(s, lmap):
            # copy constructor
            # state vector or operator? (also works with dim == None)
            if s.is_ket():
                dim = (dim, (1,))
            else:
                if s.dim[0] != s.dim[1]:
                    raise ValueError('State operator must be square.')
                dim = (dim, dim)
            # call the lmap copy constructor
            super(state, self).__init__(s, dim)
            return

        elif isinstance(s, string_types):
            # string

            if s[0].isalpha():
                # named state
                name = s.lower()

                if dim == None:
                    dim = (2, 2, 2) # default: three-qubit state

                n = len(dim) # subsystems
                s = zeros(prod(dim)) # ket
                dmin = min(dim)

                if name in ('bell1', 'bell2', 'bell3', 'bell4'):
                    # Bell state
                    dim = (2, 2)
                    s = deepcopy(Q_Bell[:, ord(name[-1]) - ord('1')])
                elif name == 'ghz':
                    # Greenberger-Horne-Zeilinger state
                    for k in range(dmin):
                        s[np.ravel_multi_index(n*(k,), dim)] = 1
                elif name == 'w':
                    # W state
                    ind = 1
                    for k in reversed(range(n)):
                        s[ind * (dim[k] - 1)] = 1
                        ind *= dim[k]
                else:
                    raise ValueError("Unknown named state '{0}'.".format(name))

                s /= norm(s) # normalize
      
            else:
                # number string defining a standard basis ket
                if dim == None:
                    n = len(s)  # number of subsystems
                    dim = qubits(n)  # assume they're qubits

                # calculate the linear index
                s = numstr_to_array(s)
                if any(s >= dim):
                    raise ValueError('Invalid basis ket.')

                ind = np.ravel_multi_index(s, dim)
                s = zeros(prod(dim)) # ket
                s[ind] = 1

            dim = (dim, (1,))  # ket

        elif isinstance(s, (numbers.Number, np.number)):
            # FIXME when numpy number hierarchy inherits the Python one
            # integer defining a standard basis ket
            if dim == None:
                raise ValueError('Need system dimension.')

            ind = s
            temp = prod(dim)  # total number of states
            if ind >= temp:
                raise ValueError('Invalid basis ket.')

            s = zeros(prod(dim)) # ket
            s[ind] = 1
            dim = (dim, (1,))  # ket

        else:
            # valid ndarray initializer representing a state vector or a state op
            s = array(s)
            if not 1 <= s.ndim <= 2:
                raise ValueError('State must be given as a state vector or a state operator.')

            # state vector or operator?
            if s.ndim == 2 and s.shape[1] != 1:
                if s.shape[0] != s.shape[1]:
                    raise ValueError('State operator matrix must be square.')
                dim = (dim, dim)  # op
            else:
                dim = (dim, (1,))  # ket

        # now s is an ndarray
        # call the lmap constructor
        super(state, self).__init__(s, dim)


# utility methods
# TODO design issue: for valid states, lots of these funcs should return reals (marked with a commented-out .real). should we just drop the imaginary part? what about if the state is invalid, how will the user know? what about numerical errors?
# TODO same thing except with normalization, should we assume states are normalized?

    def check(self):
        """Checks the validity of the state.

        Makes sure it is normalized, and if an operator, Hermitian and semipositive.
        """
        ok = True
        if abs(self.trace() - 1) > tol:
            _warn('State not properly normalized.')
            ok = False

        if not self.is_ket():
            if norm(self.data - self.data.conj().transpose()) > tol:
                _warn('State operator not Hermitian.')
                ok = False
            if min(eigvalsh(self.data)) < -tol:
                _warn('State operator not semipositive.')
                ok = False

        if not ok:
            raise ValueError('Not a valid state.')


    def subsystems(self):
        """Number of subsystems in the state."""
        return len(self.dim[0])


    def dims(self):
        """Dimensions of the state."""
        return self.dim[0] # dims of the other index must be equal (or 1)


    def clean_selection(self, sys):
        """Make a subsystem set unique and sorted, return it as an array.
        TODO valid too?"""
        return array(list(set(range(self.subsystems())).intersection(sys)), int)


    def invert_selection(self, sys):
        """Invert and sort a subsystem set."""
        return array(list(set(range(self.subsystems())).difference(sys)), int)


    def fix_phase(self, inplace=False):
        """Apply a global phase convention to a ket state.

        Returns a copy of the state. Additionally, if the state is represented with
        a ket, the copy has a global phase such that the first nonzero element in the
        state vector is real and positive.
        """
        # Ville Bergholm 2009-2011

        s = self._inplacer(inplace)
        if s.is_ket():
            # apply the phase convention: first nonzero element in state vector is real, positive
            v = s.data
            for k in range(v.size):
                temp = v.flat[k]
                if abs(temp) > tol:
                    phase = temp / abs(temp)
                    v /= phase
                    break
        return s


    def normalize(self, inplace=False):
        """Normalize the state to unity."""
        s = self._inplacer(inplace)
        if s.is_ket():
            s.data /= norm(s.data)
        else:
            s.data /= trace(s.data)
        return s


    def purity(self):
        r"""Purity of the state.

        Returns the purity of a normalized state, :math:`p = \mathrm{Tr}(\rho^2)`.
        Equivalent to linear entropy, :math:`S_l = 1-p`.
        """
        # Ville Bergholm 2008-2011

        if self.is_ket():
            return 1
        else:
            # rho is hermitian so purity should be real
            return trace(dot(self.data, self.data)) # .real


    def to_ket(self, inplace=False):
        """Convert the state representation into a ket (if possible).

        If the state is pure returns q, a copy of the state, for which the
        internal representation (q.data) is guaranteed to be a ket vector.
        """
        # Ville Bergholm 2009-2010

        s = self._inplacer(inplace)
        if not s.is_ket():
            # state op
            if abs(s.purity() - 1) > tol:
                raise ValueError('The state is not pure, and thus cannot be represented by a ket vector.')

            d, v = eighsort(s.data)
            s.data = v[:, [0]]  # corresponds to the highest eigenvalue, i.e. 1
            s.fix_phase(inplace = True)  # clean up global phase
            s.dim = (s.dim[0], (1,))

        return s


    def to_op(self, inplace=False):
        """Convert state representation into a state operator.

        Returns q, a copy of the state for which the internal representation 
        (q.data) is guaranteed to be a state operator.
        """
        # Ville Bergholm 2009-2010
        # slight inefficiency when self.is_ket() and inplace==False: the ket data is copied for no reason
        s = self._inplacer(inplace)
        if s.is_ket():
            s.data = np.outer(s.data, s.data.conj())
            s.dim = (s.dim[0], s.dim[0])

        return s


    def trace(self):
        """Trace of the state operator.

        Returns the trace of the state operator of quantum state s.
        For a pure state this is equal to the squared norm of the state vector.
        """
        # Ville Bergholm 2008-2012

        if self.is_ket():
            # squared norm, thus always real
            return vdot(self.data, self.data).real
        else:
            return trace(self.data)  # .real


    def ptrace(self, sys, inplace=False):
        """Partial trace.

        Returns the partial trace of the state
        over the subsystems listed in the vector sys.
        """
        # Ville Bergholm 2008-2010

        s = self.to_op(inplace)
        dim = s.dims()
        n = s.subsystems()
        sys = s.clean_selection(sys)
        keep = s.invert_selection(sys)

        def tensorsum(a, b):
            """Like tensor product, but with sum."""
            #c = log(kron(exp(a), exp(b))) # a perverted way of doing it, the exp overflows...
            c = []
            for k in a:
                c.extend(k + b)
            return array(c)

        # big-endian (C) data ordering
        # we trace over the subsystems in order, starting from the first one
        # partial trace over single system j, performed for every j in sys
        d = list(dim)
        for j in sys:
            muls = index_muls(d)  # muls == [d_{n-1}*...*d_1, d_{n-1}*...*d_2, ..., d_{n-1}, 1]

            # build the index "stencil"
            inds = array([0])
            for k in range(n):
                if k != j:
                    inds = tensorsum(inds, r_[0 : muls[k] * d[k] : muls[k]])
                    # np.arange(d[k]) * muls[k]

            stride = muls[j] # stride for moving the stencil while summing
            temp = len(inds)
            res = zeros((temp, temp), complex) # result
            for k in range(d[j]):
                temp = inds + stride * k
                res += s.data[ix_(temp, temp)]

            s.data = res # replace data
            d[j] = 1  # remove traced-over dimension.

        dim = tuple(array(dim)[keep]) # remove traced-over dimensions for good
        if len(dim) == 0:
            dim = (1,) # full trace gives a scalar

        s.dim = (dim, dim)
        return s



    def ptranspose(self, sys, inplace=False):
        """Partial transpose.

        Returns the partial transpose of the state
        wrt. the subsystems listed in the vector sys.
        """
        # Ville Bergholm 2008-2011

        # TODO what about kets? can we do better?
        s = self.to_op(inplace)
        dim = s.dims()
        n = s.subsystems()
        # total dimension
        orig_d = s.data.shape
        # which systems to transpose
        sys = s.clean_selection(sys)

        # swap the transposed dimensions
        perm = np.arange(2 * n)  # identity permutation
        perm[r_[sys, sys + n]] = perm[r_[sys + n, sys]]

        # flat matrix into tensor, partial transpose, back into a flat matrix
        s.data = s.data.reshape(dim + dim).transpose(perm).reshape(orig_d)
        return s


    def reorder(self, perm, inplace=False):
        """Change the relative order of subsystems in a state.
        ::

          reorder([2, 1, 0])    reverse the order of three subsystems
          reorder([2, 5])       swap subsystems 2 and 5

        Reorders the subsystems of the state according to the permutation perm.

        The permutation vector may consist of either exactly two subsystem indices
        (to be swapped), or a full permutation of subsystem indices.
        """
        # Ville Bergholm 2010

        # this is just an adapter for lmap.reorder
        if self.is_ket():
            perm = (perm, None)
        else:
            perm = (perm, perm)
        return super(state, self).reorder(perm, inplace = inplace)


# physics methods

    def ev(self, A):
        """Expectation value of an observable in the state.

        Returns the expectation value of the observable A in the state.
        A has to be Hermitian.
        """
        # Ville Bergholm 2008

        # TODO for diagonal A, self.ev(A) == sum(A * self.prob())
        if self.is_ket():
            # state vector
            x = vdot(self.data, dot(A, self.data))
        else:
            # state operator
            x = trace(dot(A, self.data))
        return x.real # .real for a Hermitian observable and valid state


    def var(self, A):
        """Variance of an observable in the state.

        Returns the variance of the observable A in the state.
        A has to be Hermitian.
        """
        # Ville Bergholm 2009

        return self.ev(A**2) - self.ev(A)**2


    def prob(self):
        """Measurement probabilities of the state in the computational basis.

        Returns a vector of probabilities of finding a system
        in each of the different states of the computational basis.
        """
        # Ville Bergholm 2009

        if self.is_ket():
            temp = self.data.ravel() # into 1D array
            return (temp * temp.conj()).real  # == np.absolute(self.data) ** 2
        else:
            return diag(self.data).real # .real


    def projector(self):
        """Projection operator defined by the state.

        Returns the projection operator P defined by the state.
        """
        # Ville Bergholm 2009-2014
        if abs(self.purity() - 1) > tol:
            raise ValueError('The state is not pure, and thus does not correspond to a projector.')

        s = self.to_op()
        return lmap(s)


    def u_propagate(self, U):
        """Propagate the state using a unitary.

        Propagates the state using the unitary propagator U,
        returns the resulting state.
        """
        # Ville Bergholm 2009-2010

        if isinstance(U, lmap):
            if self.is_ket():
                return state(U * self)
            else:
                return state(U * self * U.ctranspose())
        elif isinstance(U, np.ndarray):
            # U is a matrix, dims do not change. could also construct an lmap here...
            if self.is_ket():
                return state(dot(U, self.data), self.dims())
            else:
                return state(dot(dot(U, self.data), U.conj().transpose()), self.dims())
        else:
            raise TypeError('States can only be propagated using lmaps and arrays.')


    def propagate(self, G, t, out_func=lambda x, h: deepcopy(x), **kwargs):
        r"""Propagate the state continuously in time.
        ::

          propagate(H, t)                     Hamiltonian
          propagate(L, t)                     Liouvillian
          propagate([H, A_1, A_2, ...], t)    Hamiltonian and Lindblad ops

        Propagates the state using the generator G for the time t,
        returns the resulting state.

        The generator G can either be a

        * Hamiltonian H: :math:`\text{out} = \exp(-i H t) |s\rangle` (or :math:`\exp(-i H t) \rho_s \exp(+i H t)`)
        * Liouvillian superoperator L: :math:`\text{out} = \text{inv\_vec}(\exp(L t) \text{vec}(\rho_s))`
        * list consisting of a Hamiltonian followed by Lindblad operators.
        
        For time-dependent cases, G can be a function G(t) which takes a time instance t
        as input and returns the corresponding generator(s).

        If t is a vector of increasing time instances, returns a list
        containing the propagated state for each time given in t.

        Optional parameters:
        out_func: If given, for each time instance t returns out_func(s(t), G(t)).
        Any unrecognized keyword args are passed on to the ODE solver.
        """
        # Ville Bergholm 2008-2011
        # James Whitfield 2009

        s = self._inplacer(False)
        if isscalar(t):
            t = [t]
        t = asarray(t)
        n = len(t) # number of time instances we are interested in
        out = []
        dim = s.data.shape[0]  # system dimension

        if isinstance(G, types.FunctionType):
            # time dependent
            t_dependent = True
            F = G
            H = G(0)
        else:
            # time independent
            t_dependent = False
            H = G

        if isinstance(H, np.ndarray):
            # matrix
            dim_H = H.shape[1]
            if dim_H == dim:
                gen = 'H'  # Hamiltonian
            elif dim_H == dim ** 2:
                gen = 'L'  # Liouvillian
                s.to_op(inplace = True)
            else:
                raise ValueError('Dimension of the generator does not match the dimension of the state.')
        elif isinstance(H, list):
            # list: Hamiltonian and the Lindblad operators
            dim_H = H[0].shape[1]
            if dim_H == dim:
                gen = 'A'
                s.to_op(inplace = True)

                # HACK, in this case we use an ODE solver anyway
                if not t_dependent:
                    t_dependent = True 
                    F = lambda t: H  # ops stay constant
            else:
                raise ValueError('Dimension of the Lindblad ops does not match the dimension of the state.')
        else:
            raise ValueError("""The second parameter has to be either a matrix, a list,
                             or a function that returns a matrix or a list.""")

        dim = s.data.shape  # may have been switched to operator representation

        if t_dependent:
            # time dependent case, use ODE solver

            # derivative functions for the solver TODO vectorization?
            # H, ket
            def pure_fun(t, y, F):
                "Derivative of a pure state, Hamiltonian."
                return -1j * dot(F(t), y)
            def pure_jac(t, y, F):
                "Jacobian of a pure state, Hamiltonian."
                return -1j * F(t)

            # H, state op
            def mixed_fun(t, y, F):
                "Derivative of a mixed state, Hamiltonian."
                H = -1j * F(t)
                if y.ndim == 1:
                    rho = inv_vec(y, dim)  # into a matrix
                    return vec(dot(H, rho) - dot(rho, H)) # back into a vector
                else:
                    # vectorization, rows of y 
                    d = empty(y.shape, complex)
                    for k in range(len(y)):
                        rho = inv_vec(y[k], dim) # into a matrix
                        d[k] = vec(dot(H, rho) - dot(rho, H)) # back into a vector
                    return d

            # L, state op, same as the H/ket ones, only without the -1j
            def liouvillian_fun(t, y, F):
                "Derivative of a state, Liouvillian."
                return dot(F(t), y)
            def liouvillian_jac(t, y, F):
                "Jacobian of a state, Liouvillian."
                return F(t)

            # A, state op
            def lindblad_fun(t, y, F):
                "Derivative of a mixed state, Lindbladian."
                X = F(t)  # X == [H, A_1, A_2, ..., A_n]
                H = -1j * X[0] # -1j * Hamiltonian
                Lind = X[1:]   # Lindblad ops
                if y.ndim == 1:
                    rho = inv_vec(y, dim)  # into a matrix
                    temp = dot(H, rho) - dot(rho, H)
                    for A in Lind:
                        ac = 0.5 * dot(A.conj().transpose(), A)
                        temp += dot(dot(A, rho), A.conj().transpose()) -dot(ac, rho) -dot(rho, ac)
                    return vec(temp) # back into a vector
                else:
                    # vectorization, rows of y
                    d = empty(y.shape, complex)
                    for k in range(len(y)):
                        rho = inv_vec(y[k], dim)  # into a matrix
                        temp = dot(H, rho) - dot(rho, H)
                        for A in Lind:
                            ac = 0.5 * dot(A.conj().transpose(), A)
                            temp += dot(dot(A, rho), A.conj().transpose()) -dot(ac, rho) -dot(rho, ac)
                        d[k] = vec(temp)  # back into a vector
                    return d

            # what kind of generator are we using?
            if gen == 'H':  # Hamiltonian
                if dim[1] == 1:
                    func, jac = pure_fun, pure_jac
                else:
                    func, jac = mixed_fun, None
            elif gen == 'L':  # Liouvillian
                func, jac = liouvillian_fun, liouvillian_jac
            else: # 'A'  # Hamiltonian and Lindblad operators in a list
                func, jac = lindblad_fun, None

            # do we want the initial state too? (the integrator can't handle t=0!)
            if t[0] == 0:
                out.append(out_func(s, F(0)))
                t = t[1:]

            # ODE solver default parameters
            odeopts = {'rtol' : 1e-4,
                       'atol' : 1e-6,
                       'method' : 'bdf', # 'adams' for non-stiff cases
                       'with_jacobian' : True}
            odeopts.update(kwargs) # user options

            # run the solver
            r = ode(func, jac).set_integrator('zvode', **odeopts)
            r.set_initial_value(vec(s.data), 0.0).set_f_params(F).set_jac_params(F)
            for k in t:
                r.integrate(k)  # times must be in increasing order, NOT include zero(!)
                if not r.successful():
                    raise RuntimeError('ODE integrator failed.')
                s.data = inv_vec(r.y, dim)
                out.append(out_func(s, F(k)))

        else:
            # time independent case
            if gen == 'H':
                if dim_H < 500:
                    # eigendecomposition
                    d, v = eigh(H)
                    for k in t:
                        # propagator
                        U = dot(dot(v, diag(exp(-1j * k * d))), v.conj().transpose())
                        out.append(out_func(s.u_propagate(U), H))
                else:
                    # Krylov subspace method
                    # FIXME imaginary time doesn't yet work
                    w, err, hump = expv(-1j * t, H, s.data)
                    for k in range(n):
                        s.data = w[k, :]  # TODO state ops
                        out.append(out_func(s, H))
            elif gen == 'L':
                # Krylov subspace method
                w, err, hump = expv(t, H, vec(s.data))
                for k in range(n):
                    s.data = inv_vec(w[k, :])
                    out.append(out_func(s, H))

        if len(out) == 1:
            return out[0] # don't bother to wrap a single output in a list
        else:
            return out


    def kraus_propagate(self, E):
        r"""Apply a quantum operation to the state.

        Applies the quantum operation E to the state.
        :math:`E = [E_1, E_2, \ldots]` is a set of Kraus operators.
        """
        # Ville Bergholm 2009

        # TODO allow the user to apply E only to some subsystems of s0
        n = len(E)
        # TODO: If n > prod(dims(s))^2, there is a simpler equivalent
        # operation. Should the user be notified?
        def test_kraus(E):
            "Check if E represents a physical quantum operation."
            temp = 0
            for k in E:
                temp += dot(k.conj().transpose(), k)
            if norm(temp.data - np.eye(temp.shape)) > tol:
                _warn('Unphysical quantum operation.')

        if self.is_ket():
            if n == 1:
                return self.u_propagate(E[0]) # remains a pure state

        s = self.to_op()
        q = state(zeros(s.data.shape, complex), s.dims())
        for k in E:
            q += s.u_propagate(k)
        return q


    def measure(self, M=None, do='R'):
        r"""Quantum measurement.
        ::

          (p, res, c)
            = measure()                 measure the entire system projectively
            = measure((1, 4))           measure subsystems 1 and 4 projectively
            = measure([M_1, M_2, ...])  perform a general measurement
            = measure(A)                measure a Hermitian observable A

        Performs a quantum measurement on the state.

        * If no M is given, a full projective measurement in the
          computational basis is performed.

        * If a list/tuple of subsystems is given as the second parameter, only
          those subsystems are measured, projectively, in the
          computational basis.

        * A general measurement may be performed by giving a complete set
          of measurement operators :math:`[M_1, M_2, \ldots]` as the second parameter.
          A POVM can be emulated using :math:`M_i = \sqrt{P_i}` and discarding the collapsed state.

        * Finally, if the second parameter is a single Hermitian matrix A, the
          corresponding observable is measured. In this case the second
          column of p contains the eigenvalue of A corresponding to each
          measurement result.

        p = measure(..., do='P') returns the vector p, where p[k] is the probability of
        obtaining result k in the measurement. For a projective measurement
        in the computational basis this corresponds to the ket :math:`|k\rangle`.

        (p, res) = measure(...) additionally returns the index of the result of the
        measurement, res, chosen at random from the probability distribution p.
 
        (p, res, c) = measure(..., do='C') additionally gives c, the collapsed state
        corresponding to the measurement result res.
        """
        # Ville Bergholm 2009-2010

        def rand_measure(p):
            """Result of a random measurement using the prob. distribution p."""
            return nonzero(rand() <= cumsum(p))[0][0]

        perform = True
        collapse = False
        do = do.upper()
        if do in ('C', 'D'):
            collapse = True
        elif do == 'P':
            perform = False

        d = self.dims()

        if M == None:
            # full measurement in the computational basis
            p = self.prob()  # probabilities 
            if perform:
                res = rand_measure(p)
                if collapse:
                    s = state(res, d) # collapsed state

        elif isinstance(M, np.ndarray):
            # M is a matrix TODO lmap?
            # measure the given Hermitian observable
            a, P = spectral_decomposition(M)
            m = len(a)  # number of possible results

            p = zeros((m, 2))
            for k in range(m):
                p[k, 0] = self.ev(P[k])  # probabilities
            p[:, 1] = a  # corresponding measurement results

            if perform:
                res = rand_measure(p)
                if collapse:
                    # collapsed state
                    ppp = P[res]  # Hermitian projector
                    s = deepcopy(self)
                    if self.is_ket():
                        s.data = dot(ppp, s.data) / sqrt(p[res, 0])
                    else:
                        s.data = dot(dot(ppp, s.data), ppp) / p[res, 0]

        elif isinstance(M, (list, tuple)):
            if isinstance(M[0], numbers.Number):
                # measure a set of subsystems in the computational basis
                sys = self.clean_selection(M)
                d = array(d)

                # dimensions of selected subsystems and identity ops between them
                # TODO sequential measured subsystems could be concatenated as well
                q = len(sys)
                pdims = []
                start = 0  # first sys not yet included
                for k in sys:
                    pdims.append(prod(d[start:k])) # identity
                    pdims.append(d[k]) # selected subsys
                    start = k+1

                pdims.append(prod(d[start:])) # last identity

                # index multipliers 
                muls = index_muls(d[sys])
                # now muls == [..., d_s{q-1}*d_s{q}, d_s{q}, 1]

                m = muls[0] * d[sys][0] # number of possible results == prod(d[sys])

                def build_stencil(j, q, pdims, muls):
                    """Projector to state j (diagonal because we project into the computational basis)"""
                    stencil = ones(pdims[0]) # first identity
                    for k in range(q):
                        # projector for system k
                        temp = zeros(pdims[2*k + 1])
                        temp[int(j / muls[k]) % pdims[2*k + 1]] = 1
                        stencil = kron(kron(stencil, temp), ones(pdims[2*k + 2])) # temp + next identity
                    return stencil

                # sum the probabilities
                p = zeros(m)
                born = self.prob()
                for j in range(m):
                    p[j] = dot(build_stencil(j, q, pdims, muls), born)

                if perform:
                    res = rand_measure(p)
                    if collapse:
                        # collapsed state
                        s = deepcopy(self)
                        R = build_stencil(res, q, pdims, muls) # diagonal of a diagonal projector (just zeros and ones)

                        if do == 'D':
                            # discard the measured subsystems from s
                            d = np.delete(d, sys)
                            keep = (R == 1)  # indices of elements to keep
        
                            if self.is_ket():
                                s.data = s.data[keep] / sqrt(p[res])
                            else:
                                s.data = s.data[:, keep][keep, :] / p[res]

                            s = state(s.data, d)
                        else:
                            if self.is_ket():
                                s.data = R.reshape(-1, 1) * s.data / sqrt(p[res]) # collapsed state
                            else:
                                s.data = np.outer(R, R) * s.data / p[res] # collapsed state, HACK
            else:
                # otherwise use set M of measurement operators (assumed complete!)
                m = len(M)

                # probabilities
                p = zeros(m)
                for k in range(m):
                    p[k] = self.ev(dot(M[k].conj().transpose(), M[k]))  #  M^\dagger M  is Hermitian
                    # TODO for kets, this is slightly faster:
                    #temp = dot(M[k], self.data)
                    #p[k] = vdot(temp, temp)

                if perform:
                    res = rand_measure(p)
                    if collapse:
                        s = deepcopy(self)
                        if self.is_ket():
                            s.data = dot(M[res], s.data) / sqrt(p[res])
                        else:
                            s.data = dot(dot(M[res], s.data), M[res].conj().transpose()) / p[res]
        else:
            raise ValueError('Unsupported input type.')
        if collapse:
            return p, res, s
        elif perform:
            return p, res
        else:
            return p



# quantum information methods

    def fidelity(self, r):
        r"""Fidelity of two states.

        Fidelity of two state operators \rho and \sigma is defined as
        :math:`F(\rho, \sigma) = \mathrm{Tr} \sqrt{\sqrt{\rho} \sigma \sqrt{\rho}}`.
        For state vectors this is equivalent to the overlap, :math:`F = |\langle a|b\rangle|`.

        Fidelity is symmetric in its arguments and bounded in the interval [0,1].
        TODO Uhlmann's theorem, Bures metric, monotonicity under TP maps
        See :cite:`NC`, chapter 9.2.2.
        """
        # Ville Bergholm 2009-2010

        if not isinstance(r, state):
            raise TypeError('Not a state.')

        if self.is_ket():
            if r.is_ket():
                return abs(vdot(self.data, r.data))
            else:
                return sqrt(vdot(self.data, dot(r.data, self.data)).real)  # .real
        else:
            if r.is_ket():
                return sqrt(vdot(r.data, dot(self.data, r.data)).real)  # .real
            else:
                temp = sqrtm(self.data)
                return trace(sqrtm(dot(dot(temp, r.data), temp))).real  # .real


    def trace_dist(self, r):
        r"""Trace distance of two states.

        Trace distance between state operators r and s is defined as
        :math:`D(r, s) = \frac{1}{2} \mathrm{Tr}(\sqrt{A^\dagger A})`, where A = r-s.

        Equivalently :math:`D(r, s) = \frac{1}{2} \sum_k |\lambda_k|`, where :math:`\lambda_k`
        are the eigenvalues of A (since A is Hermitian). See :cite:`NC`, chapter 9.2.1.

        TODO stuff in NC
        """
        # Ville Bergholm 2009

        if not isinstance(r, state):
            raise TypeError('Not a state.')

        # avoid copying state ops since we just do read-only stuff here
        S = self.to_op() if self.is_ket() else self
        R = r.to_op() if r.is_ket() else r

        A = R.data - S.data
        return 0.5 * sum(abs(eigvalsh(A)))
        #return 0.5*trace(sqrtm(A'*A))



    def schmidt(self, sys=None, full=False):
        r"""Schmidt decomposition.
        ::

          lambda       = schmidt(sys)
          lambda, u, v = schmidt(sys, full=True)

        Calculates the Schmidt decomposition of the (pure) state.
        Subsystems listed in vector sys constitute part A, the rest forming part B.
        Vector lambda will contain the Schmidt coefficients.

        If required, matrices u and v will contain the corresponding orthonormal
        Schmidt bases for A and B, respectively, as column vectors, i.e.

        .. math::

           |k\rangle_A = u[:, k],  \qquad |k\rangle_B = v[:, k].

        The state is then given by :math:`\sum_k \lambda_k |k\rangle_A \otimes |k\rangle_B`.

        See :cite:`NC`, chapter 2.5.
        """
        # Ville Bergholm 2009-2014

        dim = array(self.dims())
        n = self.subsystems()

        if sys == None:
            if n == 2:
                # reasonable choice
                sys = (0,)
            else:
                raise ValueError('Requires a state and a vector of subsystems.')

        try:
            s = self.to_ket()
        except ValueError:
            raise ValueError('Schmidt decomposition is only defined for pure states.')

        # complement of sys, dimensions of the partitions
        sys = s.clean_selection(sys)
        compl = s.invert_selection(sys)
        d1 = prod(dim[sys])
        d2 = prod(dim[compl])
        perm = r_[sys, compl]

        if all(perm == range(n)):
            # nothing to do
            pass
        else:
            # reorder the system according to the partitioning
            s.reorder(perm, inplace = True)

        # order the coefficients into a matrix, take an svd
        if not full:
            return svdvals(s.data.reshape(d1, d2))
        else:
            u, s, vh = svd(s.data.reshape(d1, d2), full_matrices = False)
            # note the definition of vh in svd
            return s, u, vh.transpose()



    def entropy(self, sys=None, alpha=1):
        r"""Von Neumann or Renyi entropy of the state.

        Returns the Renyi entropy of order :math:`\alpha`,
        :math:`S_\alpha(\rho) = \frac{1}{1-\alpha} \log_2 \mathrm{Tr}(\rho^\alpha)`.
        
        When :math:`\alpha = 1`, this coincides with the von Neumann entropy
        :math:`S(\rho) = -\mathrm{Tr}(\rho \log_2(\rho))`.

        If sys == None, returns the entropy of the state.
        If sys is a vector of subsystem indices, returns the
        entropy of entanglement of the state wrt. the partitioning
        defined by sys. Entropy of entanglement is only defined for pure states.
        """
        # Ville Bergholm 2009-2012

        if sys != None:
            s = self.to_ket().ptrace(sys) # partial trace over one partition
        else:
            s = self

        if s.is_ket():
            return 0

        p = eigvalsh(s.data)
        if alpha != 1:
            # Rényi entropy
            return np.log2(sum(p ** alpha)) / (1 - alpha)
        else:
            # Von Neumann entropy
            p[p == 0] = 1   # avoid trouble with the logarithm
            return -dot(p, np.log2(p))


    def concurrence(self, sys=None):
        """Concurrence of the state.

        Returns the concurrence of the state s wrt. the partitioning
        given by the listing of subsystems in the vector sys.

        See :cite:`Wootters`, :cite:`Horodecki`.
        """
        # Ville Bergholm 2006-2014
        # TODO rewrite, check
        if abs(self.trace() - 1) > tol:
            _warn('State not properly normalized.')

        dim = self.dims()

        if sys != None:
            # concurrence between a qubit and a larger system
            if not(len(sys) == 1 and dim[sys] == 2):
                raise ValueError('Concurrence only defined between a qubit and another system.')

            if abs(self.purity() - 1) > tol:
                raise ValueError('Not a pure state.')

            # pure state
            #n = len(dim)
            rho_A = self.ptrace(self.invert_selection(sys)) # trace over everything but sys
            return 2 * sqrt(det(rho_A.data).real) # = sqrt(2*(1-real(trace(temp*temp)))), .real

        # concurrence between two qubits
        if self.subsystems() != 2 or any(dim != array([2, 2])):
            # not a two-qubit state
            raise ValueError('Not a two-qubit state.')

        X = kron(sy, sy)
        p = self.data
        if self.is_ket():
            # ket
            return abs(dot(dot(p.transpose(), X), p))

            # find the coefficients a of the state ket in the magic base
            # phi+-, psi+-,  = triplet,singlet
            #bell = [1 i 0 0 0 0 i 1 0 0 i -1 1 -i 0 0]/sqrt(2)
            #a = bell'*p
            #C = abs(sum(a ** 2))
        else:
            # state operator
            temp = dot(p, X)  # X.conj() == X so this works
            temp = dot(temp, temp.conj())  # == p * X * conj(p) * X
            if abs(self.purity() - 1) > tol:
                L = sqrt(sort(np.linalg.eigvals(temp).real)[::-1]).real  # .real?
                return max(0, L[1] -L[2] -L[3] -L[4])
            else:
                return sqrt(trace(temp).real) # same formula as for state vecs, .real?


    def negativity(self, sys):
        """Negativity of the state.

        Returns the negativity of the state wrt. the partitioning
        given by the listing of subsystems in the vector sys.

        See :cite:`Peres`, :cite:`Horodecki1`
        """
        # Ville Bergholm 2008-2014

        s = self.ptranspose(sys)  # partial transpose the state
        x = svdvals(s.data)  # singular values
        return (sum(x) - 1) / 2


    def lognegativity(self, sys):
        """Logarithmic negativity of the state.

        Returns the logarithmic negativity of the state wrt. the partitioning
        given by the listing of subsystems in the vector sys.
        """
        # Ville Bergholm 2008-2014

        return np.log2(2 * self.negativity(sys) + 1)


    def scott(self, m):
        """Scott's average bipartite entanglement measure.

        Returns the vector Q containing the terms of the Scott entanglement measure
        of the system for partition size m.

        When m = 1 this is coincides with the Meyer-Wallach entanglement measure.

        See :cite:`Love`, :cite:`Scott`, :cite:`MW`.
        """
        # Jacob D. Biamonte 2008
        # Ville Bergholm 2008-2014

        import itertools

        dim = self.dims()
        n = self.subsystems()

        if m < 1 or m > n-1:
            raise ValueError('Partition size must be between 1 and n-1.')

        D = min(dim) # FIXME correct for arbitrary combinations of qudits??
        N = sp.misc.comb(n, m, exact=True)
        C = (D**m / (D**m - 1)) / N  # normalization

        Q = empty((N,))
        # Loop over all m-combinations of n subsystems, trace over everything except them.
        # reversed() fixes the order since we are actually looping over the complements.
        for k, sys in enumerate(reversed(list(itertools.combinations(range(n), n-m)))):
            temp = self.ptrace(sys)  # trace over everything except S_k
            # NOTE: For pure states, tr(\rho_S^2) == tr(\rho_{\bar{S}}^2)
            temp = 1 - trace(np.linalg.matrix_power(temp.data, 2))
            Q[k] = C * temp.real
        return Q



    def locc_convertible(self, t, sys):
        """LOCC convertibility of states.

        For bipartite pure states s and t, returns True if s can be converted to t
        using local operations and classical communication (LOCC).
        sys is a vector of subsystems defining the partition.

        See :cite:`NC`, chapter 12.5.1
        """
        # Ville Bergholm 2010

        if not equal_dims(self, t):
            raise ValueError('States must have equal dimensions.')

        try:
            s = self.to_ket()
            t = t.to_ket()
        except ValueError:
            raise ValueError('Not implemented for nonpure states.')

        s.ptrace(sys, inplace = True)
        t.ptrace(sys, inplace = True)
        return majorize(eigvalsh(s.data), eigvalsh(t.data))



    def plot(self, symbols=3):
        """State tomography plot.

        Plots the probabilities of finding a system in this state
        in the different computational basis states upon measurement.
        Relative phases are represented by the colors of the bars.

        If the state is nonpure, also plots the coherences.
        """
        # Ville Bergholm 2009-2010

        import matplotlib.pyplot as plt
        from matplotlib import cm, colors

        dim = self.dims()
        n = self.subsystems()

        # prepare labels
        m = min(n, symbols)  # at most three symbols
        d = dim[:m]
        nd = prod(d)
        rest = '0' * (n-m) # the rest is all zeros
        ticklabels = []
        for k in range(nd):
            temp = array_to_numstr(np.unravel_index(k, d))
            ticklabels.append(temp + rest)

        ntot = prod(dim)
        skip = ntot / nd  # only every skip'th state gets a label to avoid clutter
        ticks = r_[0 : ntot : skip]

        N = self.data.shape[0]
        fig = plt.gcf()
        fig.clf() # clear the figure

        # color normalization
        nn = colors.Normalize(vmin = -1, vmax = 1, clip = True)
        def phases(A):
            """Phase normalized to (-1,1]"""
            return np.angle(A) / np.pi

        if self.is_ket():
            s = self.fix_phase()
            c = phases(s.data.ravel())  # use phases as colors
            ax = fig.gca()

            width = 0.8
            bars = ax.bar(range(N), s.prob(), width)
            # color bars using phase data
            colormapper = cm.ScalarMappable(norm=nn, cmap=cm.hsv)
            colormapper.set_array(c)
            for b in range(N):
                bars[b].set_edgecolor('k')
                bars[b].set_facecolor(colormapper.to_rgba(c[b]))

            # add a colorbar
            cb = fig.colorbar(colormapper, ax = ax, ticks = linspace(-1, 1, 5))
            cb.ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])

            # the way it should work (using np.broadcast, ScalarMappable)
            #bars = ax.bar(range(N), s.prob(), color=c, cmap=cm.hsv, norm=whatever, align='center')
            #cb = fig.colorbar(bars, ax = ax, ticks = linspace(-1, 1, 5))

            ax.set_xlabel('Basis state')
            ax.set_ylabel('Probability')
            ax.set_xticks(ticks + width / 2) # shift by half the bar width
            ax.set_xticklabels(ticklabels)
        else:
            import mpl_toolkits.mplot3d

            c = phases(self.data)  # use phases as colors
            ax = fig.add_subplot(111, projection = '3d')
            ax.view_init(40, -115)

            width = 0.6  # bar width
            temp = np.arange(-width/2, N-1) # center the labels
            x, y = meshgrid(temp, temp[::-1])
            x = x.ravel()
            y = y.ravel()
            z = abs(self.data.ravel())
            dx = width * ones(x.shape)
            ax.bar3d(x, y, zeros(x.shape), dx, dx, z, color='b')

            # now the colors
            pcol = ax.get_children()[3]  # FIXME we need a robust way of getting the poly3Dcollection
            pcol.set_norm(nn)
            pcol.set_cmap(cm.hsv)
            pcol.set_array(kron(c.ravel(), (1,)*6))  # six faces per bar

            # add a colorbar
            cb = fig.colorbar(pcol, ax = ax, ticks = linspace(-1, 1, 5))
            cb.ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])

            # the way it should work (using np.broadcast, ScalarMappable)
            #x, y = meshgrid(temp, temp)
            #pcol = ax.bar3d(x, y, 0, width, width, abs(self.data), color=c, cmap=cm.hsv, norm=whatever, align='center')

            ax.set_xlabel('Col state')
            ax.set_ylabel('Row state')
            ax.set_zlabel('$|\\rho|$')
            #ax.set_xticks(ticks)
            ax.set_xticklabels(ticklabels)
            #ax.set_yticks(ticks)
            ax.set_yticklabels(ticklabels[::-1])
            #ax.set_alpha(0.8)
            # TODO ticks, ticklabels, alpha

        plt.show()
        return ax


# other state representations

    def bloch_vector(self):
        r"""Generalized Bloch vector.

        Returns the generalized Bloch vector A corresponding to the state.

        For an n-subsystem state the generalized Bloch vector is an order-n correlation
        tensor defined in terms of the standard Hermitian tensor basis B
        corresponding to state dimensions:

        .. math::

           A_{ijk\ldots} = \sqrt{D} \mathrm{Tr}(\rho_s  B_{ijk\ldots}),

        where D = prod(self.dims()). A is always real since :math:`\rho_s` is Hermitian.
        For valid, normalized states

           self.purity() <= 1   implies   norm(A, 'fro')  <= sqrt(D)

           self.trace()   = 1   implies   A[0, 0, ..., 0] == 1

        E.g. for a single-qubit system norm(A, 'fro') <= sqrt(2).
        """
        # Ville Bergholm 2009-2011

        dim = self.dims()
        G = tensorbasis(dim)
        a = []
        for g in G:
            a.append(self.ev(g))
        a = array(a) * sqrt(prod(dim)) # to match the usual Bloch vector normalization

        # into an array, one dimension per subsystem
        return a.reshape(array(dim) ** 2)




    def tensor(*arg):
        """Tensor product of states.

        Returns the tensor product state of states s1, s2, ...
        """
        # Ville Bergholm 2009-2010

        # if all states are kets, keep the result state a ket
        pure = True
        for k in arg:
            if not k.is_ket():
                pure = False
                break

        if not pure:
            # otherwise convert all states to state ops before tensoring
            temp = []
            for k in arg:
                temp.append(k.to_op())
            arg = temp

        return state(lmap_tensor(*arg))


    @staticmethod
    def bloch_state(A, dim=None):
        r"""State corresponding to a generalized Bloch vector.
        s = bloch_state(A)       assume 
        s = bloch_state(A, dim)  give state dimensions explicitly

        Returns the state s corresponding to the generalized Bloch vector A.

        A is defined in terms of the standard Hermitian tensor basis B
        corresponding to the dimension vector dim. If dim is not given, it is assumed
        to be sqrt(A.shape).

        .. math::

           \rho_s = \sum_{ijk\ldots} A_{ijk\ldots} B_{ijk\ldots} / \sqrt{D},

        where D = prod(dim). For valid states norm(A, 2) <= sqrt(D).
        """
        # Ville Bergholm 2009-2011

        if dim == None:
            dim = tuple(sqrt(A.shape).astype(int))  # s == dim ** 2

        G = tensorbasis(dim)
        d = prod(dim)
        rho = zeros((d, d), complex)
        for k, a in enumerate(A.flat):
            rho += a * G[k]

        C = 1/sqrt(d) # to match the usual Bloch vector normalization
        return state(C * rho, dim)



# wrappers

def fidelity(s, t):
    """Wrapper for state.fidelity."""
    return s.fidelity(t)


def trace_dist(s, t):
    """Wrapper for state.trace_dist."""
    return s.trace_dist(t)
