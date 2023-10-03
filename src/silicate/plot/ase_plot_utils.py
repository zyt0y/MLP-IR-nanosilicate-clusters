import numpy as np
from math import sqrt
from itertools import islice

from ase.io.formats import string2index
from ase.utils import rotate
from ase.data import covalent_radii, atomic_numbers
from ase.data.colors import jmol_colors


class PlottingVariables:
    # removed writer - self
    def __init__(self, atoms, rotation='', show_unit_cell=2, show_bonds=1, show_vector=0,
                 color_scheme = None, plot_bond_radii=1.1,
                 bonds_color_map=None, bonds_norm=None,
                 radii=None, bbox=None, colors=None, scale=20,
                 maxwidth=500, extra_offset=(0., 0.)):
        self.numbers = atoms.get_atomic_numbers()
        self.colors = colors

        if color_scheme is None:
            self.color_scheme = np.zeros((len(atoms), 4), dtype=object)
            self.color_scheme[:, 0] = 1
            self.color_scheme[:, 2] = 1 # alpha values for atoms color
            self.color_scheme[:, 3] = 0 # dashed for atoms Circle boundaries
        else:
            self.color_scheme = color_scheme
        if colors is None:
            ncolors = len(jmol_colors)
            self.colors = jmol_colors[self.numbers.clip(max=ncolors - 1)]

        if radii is None:
            radii = covalent_radii[self.numbers]
        elif isinstance(radii, float):
            radii = covalent_radii[self.numbers] * radii
        else:
            radii = np.array(radii)

        natoms = len(atoms)

        if isinstance(rotation, str):
            rotation = rotate(rotation)

        cell = atoms.get_cell()
        disp = atoms.get_celldisp().flatten()

        if show_unit_cell > 0:
            L, T, D = cell_to_lines(self, cell)
            cell_vertices = np.empty((2, 2, 2, 3))
            for c1 in range(2):
                for c2 in range(2):
                    for c3 in range(2):
                        cell_vertices[c1, c2, c3] = np.dot([c1, c2, c3],
                                                           cell) + disp
            cell_vertices.shape = (8, 3)
            cell_vertices = np.dot(cell_vertices, rotation)
        else:
            L = np.empty((0, 3))
            T = None
            D = None
            cell_vertices = None

        if show_bonds:
            from ase.io.pov import get_bondpairs
            
            pairs = get_bondpairs(atoms, plot_bond_radii)
            '''
            if self.support_indices is not None:
                self.bonds = []
                for p in pairs:
                    a,b,c = p
                    if a in self.support_indices or b in self.support_indices:
                        #print(a, b)
                        self.bonds.append(p)
            else:
            '''
            self.bonds = pairs
            self.bonds_color_map = bonds_color_map
            self.bonds_norm = bonds_norm
            #radii *= 0.7

        else:
            self.bonds = None

        nlines = len(L)

        positions = np.empty((natoms + nlines, 3))

        R = atoms.get_positions()
        positions[:natoms] = R
        positions[natoms:] = L

        r2 = radii**2
        for n in range(nlines):
            d = D[T[n]]
            if ((((R - L[n] - d)**2).sum(1) < r2) &
                (((R - L[n] + d)**2).sum(1) < r2)).any():
                T[n] = -1

        positions = np.dot(positions, rotation)
        R = positions[:natoms]

        if bbox is None:
            X1 = (R - radii[:, None]).min(0)
            X2 = (R + radii[:, None]).max(0)
            if show_unit_cell == 2:
                X1 = np.minimum(X1, cell_vertices.min(0))
                X2 = np.maximum(X2, cell_vertices.max(0))
            M = (X1 + X2) / 2
            S = 1.05 * (X2 - X1)
            w = scale * S[0]
            if w > maxwidth:
                w = maxwidth
                scale = w / S[0]
            h = scale * S[1]
            offset = np.array([scale * M[0] - w / 2, scale * M[1] - h / 2, 0])
        else:
            w = (bbox[2] - bbox[0]) * scale
            h = (bbox[3] - bbox[1]) * scale
            offset = np.array([bbox[0], bbox[1], 0]) * scale

        offset[0] = offset[0] - extra_offset[0]
        offset[1] = offset[1] - extra_offset[1]
        self.w = w + extra_offset[0]
        self.h = h + extra_offset[1]

        positions *= scale
        positions -= offset

        if nlines > 0:
            D = np.dot(D, rotation)[:, :2] * scale

        if cell_vertices is not None:
            cell_vertices *= scale
            cell_vertices -= offset

        cell = np.dot(cell, rotation)
        cell *= scale

        self.cell = cell
        self.positions = positions
        self.D = D
        self.T = T
        self.cell_vertices = cell_vertices
        self.natoms = natoms
        self.d = 2 * scale * radii

        if show_vector and atoms.get_velocities() is not None:
            velocities = np.dot(atoms.get_velocities(), rotation)
            self.velocities = velocities
        else:
            self.velocities = None

        # extension for partial occupancies
        self.frac_occ = False
        self.tags = None
        self.occs = None

        try:
            self.occs = atoms.info['occupancy']
            self.tags = atoms.get_tags()
            self.frac_occ = True
        except KeyError:
            pass


def cell_to_lines(writer, cell):
    # XXX this needs to be updated for cell vectors that are zero.
    # Cannot read the code though!  (What are T and D? nn?)
    nlines = 0
    nsegments = []
    for c in range(3):
        d = sqrt((cell[c]**2).sum())
        n = max(2, int(d / 0.3))
        nsegments.append(n)
        nlines += 4 * n

    positions = np.empty((nlines, 3))
    T = np.empty(nlines, int)
    D = np.zeros((3, 3))

    n1 = 0
    for c in range(3):
        n = nsegments[c]
        dd = cell[c] / (4 * n - 2)
        D[c] = dd
        P = np.arange(1, 4 * n + 1, 4)[:, None] * dd
        T[n1:] = c
        for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            n2 = n1 + n
            positions[n1:n2] = P + i * cell[c - 2] + j * cell[c - 1]
            n1 = n2

    return positions, T, D


def make_patch_list(writer):
    try:
        from matplotlib.path import Path
    except ImportError:
        Path = None
        from matplotlib.patches import Circle, Polygon, Wedge, FancyArrow, Rectangle
    else:
        from matplotlib.patches import Circle, PathPatch, Wedge, FancyArrow, Rectangle

    #print(writer.D)
    #print(writer.T)
    #print(writer.bonds)
    indices = writer.positions[:, 2].argsort()
    patch_list = []
    #print(indices)
    for a in indices:
        xy = writer.positions[a, :2]
        #print(writer.natoms, a, xy)
        if a < writer.natoms:
            if writer.color_scheme[a, 0]:
                if writer.velocities is None:
                    vxy = 0
                else:
                    vxy = writer.velocities[a, :2]
                r = writer.d[a] / 2
                if writer.frac_occ:
                    site_occ = writer.occs[writer.tags[a]]
                    # first an empty circle if a site is not fully occupied
                    if (np.sum([v for v in site_occ.values()])) < 1.0:
                        # fill with white
                        fill = '#ffffff'
                        patch = Circle(xy, r, facecolor=fill,
                                    edgecolor='black')
                        patch_list.append(patch)

                    start = 0
                    # start with the dominant species
                    for sym, occ in sorted(site_occ.items(),
                                        key=lambda x: x[1],
                                        reverse=True):
                        if np.round(occ, decimals=4) == 1.0:
                            patch = Circle(xy, r, facecolor=writer.colors[a],
                                        edgecolor='black')
                            patch_list.append(patch)
                        else:
                            # jmol colors for the moment
                            extent = 360. * occ
                            patch = Wedge(
                                xy, r, start, start + extent,
                                facecolor=jmol_colors[atomic_numbers[sym]],
                                edgecolor='black')
                            patch_list.append(patch)
                            start += extent

                else:
                    if ((xy[1] + r > 0) and (xy[1] - r < writer.h) and
                        (xy[0] + r > 0) and (xy[0] - r < writer.w)):
                        alpha = writer.color_scheme[a, 2]
                        if writer.color_scheme[a, 3]:
                            ls = '--'
                        else:
                            ls = '-'
                        patch = Circle(xy, r, facecolor=writer.colors[a], alpha=alpha, ls=ls,
                                    edgecolor='black') # plotting atoms boundaries
                        patch_list.append(patch)

                        #label = 'H'
                        #patch = Rectangle(xy, 0.5, 0.5, label=label)
                        #patch_list.append(patch)
                    if np.linalg.norm(vxy) > 0.1:
                        patch = FancyArrow(*xy, *vxy, width=0.05, head_width=0.3, head_length=0.2, color='blue')
                        patch_list.append(patch)
        else:
            #print(xy)
            a -= writer.natoms
            c = writer.T[a]
            if c != -1:
                hxy = writer.D[c]
                #hxy = 0
                #print(xy + hxy, xy - hxy)
                #print(hxy)
                if Path is None:
                    #print('Polygon')
                    patch = Polygon((xy + hxy, xy - hxy), color='red')
                else:
                    # this is for plotting cell
                    # plot every short dotted lines
                    patch = PathPatch(Path((xy + hxy, xy - hxy)), color='black')
                patch_list.append(patch)
    
    if writer.bonds is not None:
        for bond_pair in writer.bonds:
            a, b, c = bond_pair
            #print(writer.positions.shape)
            p = writer.positions[[a, b]]
            #print(p)
            bond_width = 0.2
            bond_offset = np.array([bond_width, bond_width, 0])

            #p0 = p[0] + bond_offset*0.5
            #p1 = p[1] + bond_offset*0.5
            p0 = p[0]
            p1 = p[1]
            
            #plot_bonds = writer.color_scheme[a, 1] and writer.color_scheme[b, 1]
            #if (c == 0).all() and plot_bonds:
                #print(c)
            #    color = writer.color_scheme[a, 3]
                #print(color)
            #    patch = PathPatch(Path((p0[:2], p1[:2])), color=color, zorder=0, lw=1.5)
            #    patch_list.append(patch)

            #p0 = p[0] - bond_offset*0.5
            #p1 = p[1] - bond_offset*0.5
            bond_length = np.linalg.norm(p0-p1)


            if writer.bonds_color_map is not None and np.isin([a, b], np.arange(16)).all():
                color = writer.bonds_color_map(writer.bonds_norm([bond_length]))[0]
            else:
                color = 'black'
 
            patch = PathPatch(Path((p0[:2], p1[:2])), color=color, zorder=0, lw=2)
            patch_list.append(patch)
    
    return patch_list


class ImageChunk:
    """Base Class for a file chunk which contains enough information to
    reconstruct an atoms object."""

    def build(self, **kwargs):
        """Construct the atoms object from the stored information,
        and return it"""
        pass


class ImageIterator:
    """Iterate over chunks, to return the corresponding Atoms objects.
    Will only build the atoms objects which corresponds to the requested
    indices when called.
    Assumes ``ichunks`` is in iterator, which returns ``ImageChunk``
    type objects. See extxyz.py:iread_xyz as an example.
    """
    def __init__(self, ichunks):
        self.ichunks = ichunks

    def __call__(self, fd, index=None, **kwargs):
        if isinstance(index, str):
            index = string2index(index)

        if index is None or index == ':':
            index = slice(None, None, None)

        if not isinstance(index, (slice, str)):
            index = slice(index, (index + 1) or None)

        for chunk in self._getslice(fd, index):
            yield chunk.build(**kwargs)

    def _getslice(self, fd, indices):
        try:
            iterator = islice(self.ichunks(fd),
                              indices.start, indices.stop,
                              indices.step)
        except ValueError:
            # Negative indices.  Go through the whole thing to get the length,
            # which allows us to evaluate the slice, and then read it again
            if not hasattr(fd, 'seekable') or not fd.seekable():
                raise ValueError(('Negative indices only supported for '
                                  'seekable streams'))

            startpos = fd.tell()
            nchunks = 0
            for chunk in self.ichunks(fd):
                nchunks += 1
            fd.seek(startpos)
            indices_tuple = indices.indices(nchunks)
            iterator = islice(self.ichunks(fd), *indices_tuple)
        return iterator
