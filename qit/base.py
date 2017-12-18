# -*- coding: utf-8 -*-
"""Basic definitions."""
# Ville Bergholm 2008-2014

from __future__ import division, absolute_import, print_function, unicode_literals

from numpy import array, eye, sqrt, finfo

__all__ = ['I', 'sx', 'sy', 'sz', 'p0', 'p1', 'H',
           'Q_Bell', 'tol']


# Pauli matrices
I  = eye(2)
sx = array([[0, 1], [1, 0]])
sy = array([[0, -1j], [1j, 0]])
sz = array([[1, 0], [0, -1]])

# qubit projectors
p0 = array([[1, 0], [0, 0]])
p1 = array([[0, 0], [0, 1]])

# easy Hadamard
H = array([[1, 1], [1, -1]]) / sqrt(2)

# magic basis
Q_Bell = array([[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]]) / sqrt(2)


# error tolerance
tol = max(1e-8, finfo(float).eps)
