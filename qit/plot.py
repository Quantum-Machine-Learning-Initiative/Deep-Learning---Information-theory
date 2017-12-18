# -*- coding: utf-8 -*-
"""Plots."""
# Ville Bergholm 2011-2014

from __future__ import division, absolute_import, print_function, unicode_literals

import numpy as np
from numpy import array, zeros, ones, sin, cos, tanh, sort, pi, r_, c_, linspace, outer
from numpy.linalg import eigvalsh
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .state import *
from .utils import copy_memoize, eighsort

__all__ = ['adiabatic_evolution', 'state_trajectory', 'bloch_sphere', 'correlation_simplex', 'pcolor',
           'asongoficeandfire', 'sphere']


def sphere(N=15):
    """X, Y, Z coordinate meshes for a unit sphere."""
    theta = linspace(0, pi, N)
    phi = linspace(0, 2*pi, 2*N)
    X = outer(sin(theta), cos(phi))
    Y = outer(sin(theta), sin(phi))
    Z = outer(cos(theta), ones(phi.shape))
    return X, Y, Z


def adiabatic_evolution(t, st, H_func, n=4):
    """Adiabatic evolution plot.

    Input: vector t of time instances, cell vector st of states corresponding
    to the times and time-dependant Hamiltonian function handle H_func.

    Plots the energies of the eigenstates of H_func(t(k)) as a function of t(k),
    and the overlap of st{k} with the n lowest final Hamiltonian eigenstates. 
    Useful for illustrating adiabatic evolution.
    """
    # Jacob D. Biamonte 2008
    # Ville Bergholm 2009-2010

    T = t[-1]  # final time
    H = H_func(T)

    n = min(n, H.shape[0])
    m = len(t)

    # find the n lowest eigenstates of the final Hamiltonian
    #d, v = scipy.sparse.linalg.eigs(H, n, which = 'SR')
    #ind = d.argsort()  # increasing real part
    d, v = eighsort(H)
    lowest = []
    for j in range(n):
        #j = ind[j]
        lowest.append(state(v[:, -j-1]))
    # TODO with degenerate states these are more or less random linear combinations of the basis states... overlaps are not meaningful

    energies = zeros((m, H.shape[0]))
    overlaps = zeros((m, n))
    for k in range(m):
        tt = t[k]
        H = H_func(tt)
        energies[k, :] = sort(eigvalsh(H).real)
        for j in range(n):
            overlaps[k, j] = lowest[j].fidelity(st[k]) ** 2 # squared overlap with lowest final states

    plt.subplot(2, 1, 1)
    plt.plot(t/T, energies)
    plt.grid(True)
    plt.title('Energy spectrum')
    plt.xlabel('Adiabatic time')
    plt.ylabel('Energy')
    plt.axis([0, 1, np.min(energies), np.max(energies)])


    plt.subplot(2, 1, 2)
    plt.plot(t/T, overlaps) #, 'LineWidth', 1.7)
    plt.grid(True)
    plt.title('Squared overlap of current state and final eigenstates')
    plt.xlabel('Adiabatic time')
    plt.ylabel('Probability')
    temp = []
    for k in range(n):
        temp.append('$|{0}\\rangle$'.format(k))
    plt.legend(temp)
    plt.axis([0, 1, 0, 1])
    # axis([0, 1, 0, max(overlaps)])


def bloch_sphere(ax=None):
    """Bloch sphere plot.

    Plots a Bloch sphere, a geometrical representation of the state space of a single qubit.
    Pure states are on the surface of the sphere, nonpure states inside it.
    The states |0> and |1> lie on the north and south poles of the sphere, respectively.

    s is a two dimensional state to be plotted.
    """
    # Ville Bergholm  2005-2012
    # James Whitfield 2010

    if ax == None:
        ax = plt.subplot(111, projection='3d')

    ax.hold(True)
    # surface
    X, Y, Z = sphere()
    ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, color = 'g', alpha = 0.2, linewidth = 0) #cmap = xxx
    ax.axis('tight')
    # poles
    coords = array([[0, 0, 1], [0, 0, -1]]).transpose()  # easier to read this way
    ax.scatter(*coords, c = 'r', marker = 'o')
    ax.text(0, 0,  1.1, '$|0\\rangle$')
    ax.text(0, 0, -1.2, '$|1\\rangle$')

    # TODO equator?
    #phi = linspace(0, 2*pi, 40);
    #plot3(cos(phi), sin(phi), zeros(size(phi)), 'k-');
    # labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    return ax


def correlation_simplex(ax=None, labels='diagonal'):
    """Plots the correlations simplexes for two-qubit states.

    Plots the geometrical representation of the set of allowed
    correlations in a two-qubit state. For each group of three
    correlation variables the set is a tetrahedron.

    The groups are 'diagonal', 'pos' and 'neg'.
    For diagonal correlations the vertices correspond to the four Bell states.

    Returns the Axes instance and a vector of three linear
    indices denoting the correlations to be plotted as the x, y and z coordinates.

    NOTE the strange logic in the ordering of the pos and neg
    correlations follows the logic of the Bell state labeling convention, kind of.
    """
    # Ville Bergholm 2011-2012

    import mpl_toolkits.mplot3d as mplot3

    if ax == None:
        ax = plt.subplot(111, projection='3d')

    ax.hold(True)
    ax.grid(True)
    ax.view_init(20, -105)

    # tetrahedron
    # vertices and faces
    v = array([[-1, -1, -1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]])
    f = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [3, 2, 1]]
    polys = [v[k, :] for k in f]
    polyc = mplot3.art3d.Poly3DCollection(polys, color = 'g', alpha = 0.2)
    ax.add_collection(polyc)
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)

    # mark vertices
    ax.scatter([0], [0], [0], c = 'r', marker = '.')  # center
    ax.scatter(v[:, 0], v[:, 1], v[:, 2], c = 'r', marker = '.')  # vertices

    # label axes and vertices
    if labels == 'diagonal':
        ax.set_title('diagonal correlations')
        ax.set_xlabel('XX')
        ax.set_ylabel('YY')
        ax.set_zlabel('ZZ')
        ax.text(1.1, 1.1, -1.1, r'$|\Psi^+\rangle$')
        ax.text(1.1, -1.1, 1.1, r'$|\Phi^+\rangle$')
        ax.text(-1.1, 1.1, 1.1, r'$|\Phi^-\rangle$')
        ax.text(-1.2, -1.2, -1.2, r'$|\Psi^-\rangle$')
        ind = [5, 10, 15]

    elif labels == 'pos':
        ax.set_title('pos correlations')
        ax.set_xlabel('ZX')
        ax.set_ylabel('XY')
        ax.set_zlabel('YZ')
        ax.text(1.1, -1.1, 1.1, r'$|y+,0\rangle +|y-,1\rangle$')
        ax.text(-1.1, 1.1, 1.1, r'$|y+,0\rangle -|y-,1\rangle$')
        ax.text(1.1, 1.1, -1.1, r'$|y-,0\rangle +|y+,1\rangle$')
        ax.text(-1.2, -1.2, -1.2, r'$|y-,0\rangle -|y+,1\rangle$')
        ind = [7, 9, 14]

    elif labels == 'neg':
        ax.set_title('neg correlations')
        ax.set_xlabel('XZ')
        ax.set_ylabel('YX')
        ax.set_zlabel('ZY')
        ax.text(1.1, 1.1, -1.1, r'$|0,y-\rangle +|1,y+\rangle$')
        ax.text(-1.1, 1.1, 1.1, r'$|0,y+\rangle -|1,y-\rangle$')
        ax.text(1.1, -1.1, 1.1, r'$|0,y+\rangle +|1,y-\rangle$')
        ax.text(-1.2, -1.2, -1.2, r'$|0,y-\rangle -|1,y+\rangle$')
        ind = [13, 6, 11]

    elif labels == 'none':
        ind = []

    else:
        raise ValueError('Unknown set of correlations.')

    plt.show()
    return ax, ind


def state_trajectory(traj, reset=True, ax=None, color='b'):
    """Plot a state trajectory in the correlation representation.

    For a single-qubit system, plots the trajectory in the Bloch sphere.

    For a two-qubit system, plots the reduced single-qubit states (in
    Bloch spheres), as well as the interqubit correlations.

    traj is a list of generalized Bloch vectors.
    It can be obtained e.g. by using one of the continuous-time
    state propagation functions and feeding the results to
    bloch_vector.

    If reset is false, adds another trajectory to current plot
    without erasing it.

    Example 1: trajectory of s under the Hamiltonian H
      out = propagate(s, H, t, @(s,H) bloch_vector(s))
      bloch_trajectory(out)

    Example 2: just a single state s
      bloch_trajectory({bloch_vector(s)})
    """
    # Ville Bergholm  2006-2012

    if ax == None:
        ax = plt.subplot(111, projection='3d')

    def plot_traj(ax, A, ind):
        """Plots the trajectory formed by the correlations given in ind."""

        ax.scatter(A[0,  ind[0]],  A[0,  ind[1]],  A[0,  ind[2]], c = color, marker = 'x')
        # if we only have a single point, do not bother with these
        if len(A) > 1:
            ax.plot(A[:,  ind[0]],  A[:,  ind[1]],  A[:,  ind[2]], c = color)
            ax.scatter(A[-1, ind[0]],  A[-1, ind[1]],  A[-1, ind[2]], c = color, marker = 'o')


    if isinstance(traj, list):
        d = traj[0].size
    else:
        d = traj.size
    A = array(traj).reshape((-1, d), order='F')  # list index becomes the first dimension

    if len(A[0]) == 4:
        # single qubit
        if reset:
            bloch_sphere(ax)
        plot_traj(ax, A, [1, 2, 3])
  
    elif len(A[0]) == 16:
        # two qubits (or a single ququat...)

        # TODO split ax into subplots...
        fig = plt.gcf()
        if reset:
            gs = GridSpec(2, 3)

            ax = fig.add_subplot(gs[0, 0], projection = '3d')
            bloch_sphere(ax)
            ax.set_title('qubit A')
    
            ax = fig.add_subplot(gs[0, 1], projection = '3d')
            bloch_sphere(ax)
            ax.set_title('qubit B')

            ax = fig.add_subplot(gs[1, 0], projection = '3d')
            correlation_simplex(ax, labels = 'diagonal')

            ax = fig.add_subplot(gs[1, 1], projection = '3d')
            correlation_simplex(ax, labels = 'pos')

            ax = fig.add_subplot(gs[1, 2], projection = '3d')
            correlation_simplex(ax, labels = 'neg')

        # update existing axes instances
        qqq = fig.get_axes()
        plot_traj(qqq[0], A, [1, 2, 3])
        plot_traj(qqq[1], A, [4, 8, 12])
        plot_traj(qqq[2], A, [5, 10, 15])
        plot_traj(qqq[3], A, [7, 9, 14])
        plot_traj(qqq[4], A, [13, 6, 11])
    
    else:
        raise ValueError('At the moment only plots one- and two-qubit trajectories.')



def pcolor(W, a, b, clim=(0, 1)):
    """Easy pseudocolor plot.

    Plots the 2D function given in the matrix W.
    The vectors x and y define the coordinate grid.
    clim is an optional parameter for color limits.

    Returns the plot object.
    """
    # Ville Bergholm 2010

    # a and b are quad midpoint coordinates but pcolor wants quad vertices, so
    def to_quad(x):
        "Quad midpoints to vertices."
        return (r_[x, x[-1]] + r_[x[0], x]) / 2

    plt.gcf().clf()  # clear the figure
    p = plt.pcolor(to_quad(a), to_quad(b), W, clim = clim, cmap = asongoficeandfire())
    plt.axis('equal')
    plt.axis('tight')
    #shading('interp')
    plt.colorbar()
    return p


def makemovie(filename, frameset, plot_func, *arg):
    """Create an AVI movie. FIXME
    aviobj = makemovie(filename, frameset, plot_func [, ...])

    Creates an AVI movie file named 'filename.avi' in the current directory.
    Frame k in the movie is obtained from the contents of the
    current figure after calling plot_func(frameset[k]).
    The optional extra parameters are passed directly to avifile.

    Returns the closed avi object handle.

    Example: makemovie('test', cell_vector_of_states, @(x) plot(x))
    """
    # James D. Whitfield 2009
    # Ville Bergholm 2009-2010

    # create an AVI object
    aviobj = avifile(filename, arg)

    fig = figure('Visible', 'off')
    for k in frameset:
        plot_func(k)
        aviobj = addframe(aviobj, fig)
        #  F = getframe(fig)   
        #  aviobj = addframe(aviobj, F)

    close(fig)
    aviobj = close(aviobj)


@copy_memoize
def asongoficeandfire(n=127):
    """Colormap with blues and reds. Wraps.

    Returns a matplotlib.colors.Colormap object.
    n is the number of color definitions in the map.
    """
    # Ville Bergholm 2010-2011

    from matplotlib import colors
    # exponent
    d = 3.1
    p = linspace(-1, 1, n)
    # negative values: reds
    x = p[p < 0]
    c = c_[1 -((1+x) ** d), 0.5*(tanh(4*(-x -0.5)) + 1), (-x) ** d]
    # positive values: blues
    x = p[p >= 0]
    c = r_[c, c_[x ** d, 0.5*(tanh(4*(x -0.5)) + 1), 1 -((1-x) ** d)]]
    return colors.ListedColormap(c, name='asongoficeandfire')
    # TODO colors.LinearSegmentedColormap(name, segmentdata, N=256, gamma=1.0)
