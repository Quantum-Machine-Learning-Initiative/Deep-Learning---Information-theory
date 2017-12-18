# -*- coding: utf-8 -*-
"""
Linear maps (:mod:`qit.lmap`)
=============================

Bounded finite-dimensional linear maps are represented using :class:`lmap` class instances.
In addition to the matrix representing the map, they contain
the dimension vectors of the domain and codomain vector spaces.
All the usual scalar-map and map-map arithmetic operators are
provided, including the exponentiation of maps by integers.


.. currentmodule:: qit.lmap.lmap


Utilities
---------

.. autosummary::

   remove_singletons
   is_compatible
   is_ket


Linear algebra
--------------

.. autosummary::

   conj
   transpose
   ctranspose
   trace
   norm
   reorder


Non-member functions:

.. currentmodule:: qit.lmap

.. autosummary::
   tensor
""" 
# Ville Bergholm 2008-2011

from __future__ import division, absolute_import, print_function, unicode_literals

from copy import copy, deepcopy

import numpy as np
import scipy.sparse as ssp

from .base import tol

__all__ = ['numstr_to_array', 'array_to_numstr', 'lmap', 'tensor']



def numstr_to_array(s):
    """Utility, converts a numeric string to the corresponding array."""
    return np.array([ord(x) -ord('0') for x in s])


def array_to_numstr(s):
    """Utility, converts an integer array to the corresponding numeric string."""
    return "".join([chr(x +ord('0')) for x in s])



class lmap(object):
    """Bounded linear maps between tensor products of finite-dimensional Hilbert spaces.

    Contains both the order-2 tensor and the dimensional information.

    TODO Another possible interpretation of lmap would be to
    treat each subsystem as an index, with the subsystems within dim{1} and dim{2}
    corresponding to contravariant and covariant indices, respectively?

    Variables:
    data:  ndarray of tensor data
    dim:   tuple of input and output dimension tuples, big-endian: ((out), (in))

    Base class of state.
    """

# TODO def __format__(self, format_spec)
# TODO linalg efficiency: copy vs. view

    def __init__(self, s, dim=None):
        """Construct an lmap.

        s:    ndarray OR valid initializer for ndarray OR lmap instance
              A copy is made unless s is an ndarray.

        dim:  2-tuple containing the output and input subsystem dimensions
              stored in tuples:  dim == ((out), (in)).
              If dim, (out) or (in) is None, the corresponding dimensions
              are inferred from s.
        
        calling syntax                     resulting dim
        ==============                     =============
        lmap(rand(a))                      ((a,), (1,))      1D array default: ket vector
        lmap(rand(a), ((1,), None))        ((1,), (a,))      bra vector given as a 1D array
        lmap(rand(a,b))                    ((a,), (b,))      2D array, all dims inferred
        lmap(rand(4,b), ((2, 2), None))    ((2, 2), (b,))    2D array, output: two qubits 
        lmap(rand(a,6), (None, (3, 2)))    ((a,), (3, 2))    2D array, input: qutrit+qubit
        lmap(rand(6,6), ((3, 2), (2, 3)))  ((3, 2), (2, 3))  2D array, all dims given

        lmap(A)             (A is an lmap) copy constructor
        lmap(A, dim)        (A is an lmap) copy constructor, redefine the dimensions
        """
        # initialize the ndarray part
        if isinstance(s, lmap):
            # copy constructor
            self.data = deepcopy(s.data)
            defdim = s.dim  # copy the dimensions too, unless redefined
        else:
            if ssp.isspmatrix(s):
                # TODO FIXME handle sparse matrices properly
                # TODO lmap constructor, mul/add, tensor funcs must be able to handle both dense and sparse arrays.
                s = s.todense()

            # valid array initializer
            self.data = np.asarray(s) # NOTE that if s is an ndarray it is not copied here

            # into a 2d array
            if self.data.ndim == 0:
                # scalar
                self.data.resize((1, 1))
            elif self.data.ndim == 1:
                # vector, ket by default
                self.data.resize((self.data.size, 1))
            elif self.data.ndim > 2:
                raise ValueError('Array dimension must be <= 2.')
            # now self.data.ndim == 2, always

            # is it a bra given as a 1D array?
            if dim and dim[0] == (1,):
                self.data.resize((1, self.data.size))

            # infer default dims from data (wrap them in tuples!)
            defdim = tuple([(k,) for k in self.data.shape])

        # set the dimensions
        if dim == None:
            # infer both dimensions from s
            dim = (None, None)

        self.dim = []
        for k in range(len(dim)):
            if dim[k] == None:
                # not specified, use default
                self.dim.append(defdim[k])
            else:
                self.dim.append(tuple(dim[k]))
        self.dim = tuple(self.dim)

        # check dimensions
        if self.data.shape != tuple(map(np.prod, self.dim)):
            raise ValueError('Dimensions of the array do not match the combined dimensions of the subsystems.')


    def __repr__(self):
        """Display the lmap in a neat format."""
        out = ''
        # is it a vector? (a map with a singleton domain or codomain dimension)
        sh = self.data.shape
        if 1 in sh:
            # vector 
            # ket or bra?
            if sh[1] == 1:
                # let scalars be kets too
                dim = self.dim[0]
                is_ket = True
            else:
                dim = self.dim[1]
                is_ket = False

            # loop over all vector elements
            printed = 0
            d = np.prod(dim)
            for ind in range(d):
                # TODO with sparse arrays we could do better
                # sanity check, do not display lmaps with hundreds of terms
                if ind >= 128 or printed >= 20:
                    out += ' ...'
                    break

                temp = self.data.flat[ind]
                # make sure there is something to print
                if abs(temp) < tol:
                    continue

                printed += 1
                if abs(temp.imag) < tol:
                    # just the real part
                    out += ' {0:+.4g}'.format(temp.real)
                elif abs(temp.real) < tol:
                    # just the imaginary part
                    out += ' {0:+.4g}j'.format(temp.imag)
                else:
                    # both
                    out += ' +({0:.4g}{1:+.4g}j)'.format(temp.real, temp.imag) #' +' + str(temp)

                # ket or bra symbol
                temp = array_to_numstr(np.unravel_index(ind, dim))
                if is_ket:
                    out += ' |' + temp + '>'
                else:
                    out += ' <' + temp + '|'
        else:
            # matrix
            out = self.data.__repr__()

        out += '\ndim: ' + str(self.dim[0]) + ' <- ' + str(self.dim[1])
        return out


# utilities

    def _inplacer(self, inplace):
        """Utility for implementing inplace operations.

        Functions using this should begin with s = self._inplacer(inplace)
        and end with return s
        """
        if inplace:
            return self
        else:
            return deepcopy(self)


    def remove_singletons(self):
        """Eliminate unnecessary singleton dimensions.

        NOTE: changes the object itself!
        """
        dd = []
        for d in self.dim:
            temp = tuple([x for x in d if x > 1])
            if len(temp) == 0:
                temp = (1,)
            dd.append(temp)
        self.dim = tuple(dd)
        return


    def is_compatible(self, t):
        """True iff the lmaps have equal dimensions and can thus be added."""
        if not isinstance(t, lmap):
            raise TypeError('t is not an lmap.')
        return self.dim == t.dim


    def is_ket(self):
        """True if the lmap is a ket."""
        return self.data.shape[1] == 1


# linear algebra

    def conj(self):
        """Complex conjugate."""
        s = copy(self)  # preserves the type, important for subclasses
        s.data = np.conj(self.data) # copy
        return s


    def transpose(self):
        """Transpose."""
        s = copy(self)
        s.dim = (s.dim[1], s.dim[0]) # swap dims
        s.data = self.data.transpose().copy()
        return s


    def ctranspose(self):
        """Hermitian conjugate."""
        s = copy(self)
        s.dim = (s.dim[1], s.dim[0]) # swap dims
        s.data = np.conj(self.data).transpose() # view to a copy
        return s


    def __mul__(self, t):
        """Multiplication of lmaps by lmaps and scalars."""
        # must be able to handle sparse data
        if isinstance(t, lmap):
            if self.dim[1] != t.dim[0]:
                raise ValueError('The dimensions do not match.')
            else:
                s = copy(self)
                s.dim = (self.dim[0], t.dim[1])
                s.data = self.data.dot(t.data)
        else:
            # t is a scalar
            s = copy(self)
            s.data = self.data * t
        return s


    def __rmul__(self, t):
        """Multiplication of lmaps by scalars, reverse."""
        # scalars commute, lmaps already handled by __mul__
        return self.__mul__(t)


    def __div__(self, t):
        """Division of lmaps by scalars from the right."""
        s = copy(self)
        s.data = self.data / t
        return s


    def __truediv__(self, t):
        """Division of lmaps by scalars from the right."""
        s = copy(self)
        s.data = self.data / t
        return s


    def __add__(self, t):
        """Addition of lmaps."""
        if not self.is_compatible(t):
            raise ValueError('The lmaps are not compatible.')
        s = copy(self)
        s.data = self.data + t.data
        return s


    def __sub__(self, t):
        """Subtraction of lmaps."""
        if not self.is_compatible(t):
            raise ValueError('The lmaps are not compatible.')
        s = copy(self)
        s.data = self.data - t.data
        return s


    def __pow__(self, n):
        """Exponentiation of lmaps by integer scalars."""
        if self.dim[0] != self.dim[1]:
            raise ValueError('The dimensions do not match.')
        s = copy(self)
        s.data = np.linalg.matrix_power(self.data, n)
        return s


    def __imul__(self, t):
        """In-place multiplication of lmaps by scalars from the right."""
        self.data *= t
        return self


    def __itruediv__(self, t):
        """In-place division of lmaps by scalars from the right."""
        self.data /= t
        return self


    def __iadd__(self, t):
        """In-place addition of lmaps."""
        if not self.is_compatible(t):
            raise ValueError('The lmaps are not compatible.')
        self.data += t.data
        return self


    def __isub__(self, t):
        """In-place subtraction of lmaps."""
        if not self.is_compatible(t):
            raise ValueError('The lmaps are not compatible.')
        self.data -= t.data
        return self


    def trace(self):
        """Trace of the lmap.

        The trace is only properly defined if self.dim[0] == self.dim[1].
        """
        if not np.array_equal(self.dim[0], self.dim[1]):
            raise ValueError('Trace not defined for non-endomorphisms.')

        return np.trace(self.data)


    def norm(self):
        """Matrix norm of the lmap."""
        return np.linalg.norm(self.data)


# subsystem ordering

    def reorder(self, perm, inplace=False):
        """Change the relative order of the input and/or output subsystems.

        Returns a copy of the lmap with permuted subsystem order.

        A permutation can be either None (do nothing), a pair (a, b) of subsystems to be swapped,
        or a tuple containing a full permutation of the subsystems.
        Two subsystems to be swapped must be in decreasing order so as not
        to mistake the full identity permutation (0, 1) for a swap.

        reorder((None, (2, 1, 0)))   ignore first index, reverse the order of subsystems in the second
        reorder(((5, 2), None))      swap the subsystems 2 and 5 in the first index, ignore the second

        NOTE: The full permutations are interpreted in the same sense as
        numpy.transpose() understands them, i.e. the permutation
        tuple is the new ordering of the old subsystem indices.
        This is the inverse of the mathematically more common "one-line" notation.
        """
        s = self._inplacer(inplace)

        orig_d = s.data.shape  # original dimensions
        total_d = []
        total_perm = []
        last_used_index = 0
        newdim = list(s.dim)

        # loop over indices
        for k, this_perm in enumerate(perm):
            # avoid a subtle problem with the input syntax, (0, 1) must not be understood as swap!
            if this_perm != None and tuple(this_perm) == (0, 1):
                this_perm = None

            # requested permutation for this index
            if this_perm == None:
                # no change
                # let the dimensions vector be, lump all subsystems in this index into one
                this_dim = (orig_d[k],)
                this_perm = np.array([0])
                this_n = 1
            else:
                this_dim  = np.array(s.dim[k])  # subsystem dims
                this_perm = np.array(this_perm) # requested permutation for this index
                this_n = len(this_dim)  # number of subsystems

                temp = np.arange(this_n) # identity permutation

                if len(this_perm) == 2:
                    # swap two subsystems
                    temp[this_perm] = this_perm[::-1]
                    this_perm = temp
                else:
                    # full permutation
                    if len(set(temp) ^ set(this_perm)) != 0:
                        raise ValueError('Invalid permutation.')

                # reorder the dimensions vector
                newdim[k] = tuple(this_dim[this_perm])

            # big-endian ordering
            total_d.extend(this_dim)
            total_perm.extend(last_used_index + this_perm)
            last_used_index += this_n

        # tensor into another tensor which has one index per subsystem, permute dimensions, back into a tensor with the original number of indices
        s.dim = tuple(newdim)
        s.data = s.data.reshape(total_d).transpose(total_perm).reshape(orig_d)
        return s



def tensor(*arg):
    """Tensor product of lmaps."""
    data = 1
    dout = []
    din  = []

    for k in arg:
        # concatenate dimensions
        dout += k.dim[0]
        din  += k.dim[1]
        # kronecker product of the data
        data = np.kron(data, k.data)

    s = lmap(data, (tuple(dout), tuple(din)))
    return s
