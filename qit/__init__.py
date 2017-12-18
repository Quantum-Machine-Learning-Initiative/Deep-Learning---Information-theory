# -*- coding: utf-8 -*-
"""Python Quantum Information Toolkit

See the README.txt file included in this distribution
or the project website at http://qit.sourceforge.net/
"""
# Ville Bergholm 2011-2014

from __future__ import division, absolute_import, print_function, unicode_literals


# toolkit version number
__version__ = '0.11.0'

def version():
    """Returns the QIT version number (as a string)."""
    return __version__


import numpy as np
import scipy as sp
import scipy.constants as const
import matplotlib.pyplot as plt

from .base import *
from .lmap import *
from .utils import *
from .state import *
from .plot import *
from . import gate, hamiltonian, ho, invariant, markov, seq, examples

#print('Python Quantum Information Toolkit, version {0}.'.format(__version__))
