"""@package docstring
Iso2Mesh for Python - Primitive shape meshing functions

Copyright (c) 2024 Edward Xu <xu.ed at neu.edu>
              2024-2025 Qianqian Fang <q.fang at neu.edu>
"""

__all__ = [
    "meshgrid5",
    "meshgrid6",
    "latticegrid",
    "meshabox",
    "meshacylinder",
    "meshcylinders",
    "meshanellip",
    "meshunitsphere",
    "meshasphere",
    "extrudecurve",
    "extrudesurf",
]

##====================================================================================
## dependent libraries
##====================================================================================

import numpy as np
import sys
from itertools import permutations, combinations
from iso2mesh.core import surf2mesh, vol2restrictedtri
from iso2mesh.trait import meshreorient, volface, surfedge
from iso2mesh.utils import *
from iso2mesh.modify import removeisolatednode

# _________________________________________________________________________________________________________


def meshabox(p0, p1, maxvol=None, nodesize=1, **kwargs):
    """
    Create the surface and tetrahedral mesh of a box geometry.

    Parameters:
    p0: Coordinates (x, y, z) for one end of the box diagonal
    p1: Coordinates (x, y, z) for the other end of the box diagonal
    opt: Maximum volume of the tetrahedral elements
    nodesize: (Optional) Size of the elements near each vertex.
              Can be a scalar or an 8x1 array.

    Returns:
    node: Node coordinates, 3 columns for x, y, and z respectively
    face: Surface mesh faces, each row represents a face element
    elem: Tetrahedral elements, each row represents a tetrahedron
    """
    if nodesize is None:
        nodesize = 1

    # Call to surf2mesh function to generate the surface mesh and volume elements
    node, elem, ff = surf2mesh(
        np.array([]),
        np.array([]),
        p0,
        p1,
        1,
        maxvol,
        regions=None,
        holes=None,
        dobbx=nodesize,
        **kwargs,
    )

    # Reorient the mesh elements
    elem, _, _ = meshreorient(node, elem[:, :4])

    # Extract the surface faces from the volume elements
    face = volface(elem)[0]

    return node, face, elem


# _________________________________________________________________________________________________________


def meshunitsphere(tsize, maxvol=None):
    dim = 60
    esize = tsize * dim
    thresh = dim / 2 - 1

    xi, yi, zi = np.meshgrid(
        np.arange(0, dim + 0.5, 0.5),
        np.arange(0, dim + 0.5, 0.5),
        np.arange(0, dim + 0.5, 0.5),
    )
    dist = thresh - np.sqrt((xi - 30) ** 2 + (yi - 30) ** 2 + (zi - 30) ** 2)
    dist[dist < 0] = 0

    # Call a vol2restrictedtri equivalent in Python here (needs a custom function)
    node, face = vol2restrictedtri(
        dist, 1, (dim, dim, dim), dim**3, 30, esize, esize, 40000
    )

    node = (node - 0.5) * 0.5
    node, face, _ = removeisolatednode(node, face)

    node = (node - 30) / 28
    r0 = np.sqrt(np.sum(node**2, axis=1))
    node = node / r0[:, None]

    if not maxvol:
        maxvol = tsize**3

    # Call a surf2mesh equivalent in Python here (needs a custom function)
    node, elem, face = surf2mesh(
        node, face, np.array([-1, -1, -1]) * 1.1, np.array([1, 1, 1]) * 1.1, 1, maxvol
    )

    return node, face, elem


# _________________________________________________________________________________________________________


def meshasphere(c0, r, tsize, maxvol=None):
    if maxvol is None:
        maxvol = tsize**3

    if maxvol is not None:
        node, face, elem = meshunitsphere(tsize / r, maxvol=maxvol / (r**3))
    else:
        node, face, elem = meshunitsphere(tsize / r)

    node = node * r + np.tile(np.array(c0).reshape(1, -1), (node.shape[0], 1))

    return node, face, elem  # if maxvol is not None else (node, face)


# _________________________________________________________________________________________________________


def meshacylinder(c0, c1, r, tsize=None, maxvol=None, ndiv=20):
    if len(np.array([r])) == 1:
        r = np.array([r, r])

    if np.any(np.array(r) <= 0) or np.all(c0 == c1):
        raise ValueError("Invalid cylinder parameters")

    c0 = np.array(c0).reshape(-1, 1)
    c1 = np.array(c1).reshape(-1, 1)
    v0 = c1 - c0
    len_axis = np.linalg.norm(v0)

    if tsize is None:
        tsize = min(np.append(r, len_axis)) / 10

    if maxvol is None:
        maxvol = tsize**3 / 5

    dt = 2 * np.pi / ndiv
    theta = np.arange(dt, 2 * np.pi + dt, dt)
    cx = np.outer(np.array(r), np.cos(theta))
    cy = np.outer(np.array(r), np.sin(theta))

    p0 = np.column_stack((cx[0, :], cy[0, :], np.zeros(ndiv)))
    p1 = np.column_stack((cx[1, :], cy[1, :], len_axis * np.ones(ndiv)))

    pp = np.vstack((p0, p1))
    no = rotatevec3d(pp, v0.T) + np.tile(c0.T, (pp.shape[0], 1))

    # face = np.empty((0,4))
    face = []
    for i in range(1, ndiv):
        # face = np.vstack((face, np.array([i, i + ndiv, i + ndiv + 1, i + 1])))
        face.append([[[i, i + ndiv, i + ndiv + 1, i + 1]], [1]])

    face.append([[[ndiv, 2 * ndiv, ndiv + 1, 1]], [1]])
    face.append([[list(range(1, ndiv + 1))], [2]])
    face.append([[list(range(ndiv + 1, 2 * ndiv + 1))], [3]])

    if tsize == 0.0 and maxvol == 0.0:
        return no, face

    node, elem, *_ = surf2mesh(
        no,
        face,
        np.min(no, axis=0),
        np.max(no, axis=0),
        1,
        maxvol,
        regions=np.array([[0, 0, 1]]),
        holes=np.array([]),
    )
    face, *_ = volface(elem[:, 0:4])

    return node, face, elem


# _________________________________________________________________________________________________________


def meshgrid5(*args):
    args = list(args)

    n = len(args)
    if n != 3:
        raise ValueError("only works for 3D case!")

    for i in range(n):
        v = args[i]
        if len(v) % 2 == 0:
            args[i] = np.linspace(v[0], v[-1], len(v) + 1)

    # create a single n-d hypercube
    cube8 = np.array(
        [
            [1, 4, 5, 13],
            [1, 2, 5, 11],
            [1, 10, 11, 13],
            [11, 13, 14, 5],
            [11, 13, 1, 5],
            [2, 3, 5, 11],
            [3, 5, 6, 15],
            [15, 11, 12, 3],
            [15, 11, 14, 5],
            [11, 15, 3, 5],
            [4, 5, 7, 13],
            [5, 7, 8, 17],
            [16, 17, 13, 7],
            [13, 17, 14, 5],
            [5, 7, 17, 13],
            [5, 6, 9, 15],
            [5, 8, 9, 17],
            [17, 18, 15, 9],
            [17, 15, 14, 5],
            [17, 15, 5, 9],
            [10, 13, 11, 19],
            [13, 11, 14, 23],
            [22, 19, 23, 13],
            [19, 23, 20, 11],
            [13, 11, 19, 23],
            [11, 12, 15, 21],
            [11, 15, 14, 23],
            [23, 21, 20, 11],
            [23, 24, 21, 15],
            [23, 21, 11, 15],
            [16, 13, 17, 25],
            [13, 17, 14, 23],
            [25, 26, 23, 17],
            [25, 22, 23, 13],
            [13, 17, 25, 23],
            [17, 18, 15, 27],
            [17, 15, 14, 23],
            [26, 27, 23, 17],
            [27, 23, 24, 15],
            [23, 27, 17, 15],
        ]
    ).T

    # build the complete lattice
    nodecount = [len(arg) for arg in args]

    if any(count < 2 for count in nodecount):
        raise ValueError("Each dimension must be of size 2 or more.")

    node = lattice(*args)

    ix, iy, iz = np.meshgrid(
        np.arange(1, nodecount[0] - 1, 2),
        np.arange(1, nodecount[1] - 1, 2),
        np.arange(1, nodecount[2] - 1, 2),
        indexing="ij",
    )
    ind = np.ravel_multi_index(
        (ix.flatten() - 1, iy.flatten() - 1, iz.flatten() - 1), nodecount, order="F"
    )

    nodeshift = np.array(
        [
            0,
            1,
            2,
            nodecount[0],
            nodecount[0] + 1,
            nodecount[0] + 2,
            2 * nodecount[0],
            2 * nodecount[0] + 1,
            2 * nodecount[0] + 2,
        ]
    )
    nodeshift = np.concatenate(
        (
            nodeshift,
            nodeshift + nodecount[0] * nodecount[1],
            nodeshift + 2 * nodecount[0] * nodecount[1],
        )
    )

    nc = len(ind)
    elem = np.zeros((nc * 40, 4), dtype=int)
    for i in range(nc):
        elem[np.arange(0, 40) + (i * 40), :] = (
            np.reshape(nodeshift[cube8.flatten() - 1], (4, 40)).T + ind[i]
        )

    elem = elem + 1
    elem = meshreorient(node[:, :3], elem[:, :4])[0]

    return node, elem


# _________________________________________________________________________________________________________


def meshgrid6(*args):
    # dimension of the lattice
    n = len(args)

    # create a single n-d hypercube     # list of node of the cube itself
    vhc = (
        np.array(list(map(lambda x: list(bin(x)[2:].zfill(n)), range(2**n)))) == "1"
    ).astype(int)

    # permutations of the integers 1:n
    p = list(permutations(range(1, n + 1)))
    p = p[::-1]
    nt = len(p)
    thc = np.zeros((nt, n + 1), dtype=int)

    for i in range(nt):
        thc[i, :] = np.where(
            np.all(np.diff(vhc[:, np.array(p[i]) - 1], axis=1) >= 0, axis=1)
        )[0]

    # build the complete lattice
    nodecount = np.array([len(arg) for arg in args])
    if np.any(nodecount < 2):
        raise ValueError("Each dimension must be of size 2 or more.")
    node = lattice(*args)

    # unrolled index into each hyper-rectangle in the lattice
    ind = [np.arange(nodecount[i] - 1) for i in range(n)]
    ind = np.meshgrid(*ind, indexing="ij")
    ind = np.array(ind).reshape(n, -1).T
    k = np.cumprod([1] + nodecount[:-1].tolist())

    ind = 1 + ind @ k.T  # k[:-1].reshape(-1, 1)
    nind = len(ind)
    offset = vhc @ k.T
    elem = np.zeros((nt * nind, n + 1), dtype=int)
    L = np.arange(1, nind + 1).reshape(-1, 1)

    for i in range(nt):
        elem[L.flatten() - 1, :] = np.tile(ind, (n + 1, 1)).T + np.tile(
            offset[thc[i, :]], (nind, 1)
        )
        L += nind

    elem = meshreorient(node[:, :3], elem[:, :4])[0]

    return node, elem


# _________________________________________________________________________________________________________


def lattice(*args):
    n = len(args)
    sizes = [len(arg) for arg in args]
    grids = np.meshgrid(*args, indexing="ij")
    grid = np.zeros((np.prod(sizes), n))
    for i in range(n):
        grid[:, i] = grids[i].ravel(order="F")
    return grid


# _________________________________________________________________________________________________________


def latticegrid(*args):
    """
    node, face, centroids = latticegrid(xrange, yrange, zrange, ...)

    Generate a 3D lattice.

    Parameters:
        *args: 1D arrays specifying the range of each dimension.

    Returns:
        node: (N, D) array of node coordinates.
        face: list of faces (each a list of indices starting from 1).
        centroids: (M, D) array of centroid coordinates of each lattice cell.
    """
    n = len(args)
    p = np.meshgrid(*args, indexing="ij")
    node = np.zeros((p[0].size, n))
    for i in range(n):
        node[:, i] = p[i].ravel(order="F")

    if n == 1:
        return node

    dim = p[0].shape
    dd = [dim[0], dim[0] * dim[1]]

    onecube = np.array(
        [
            [0, dd[0], dd[0] + 1, 1],
            [0, 1, dd[1] + 1, dd[1]],
            [0, dd[1], dd[1] + dd[0], dd[0]],
        ]
    )
    onecube = np.vstack(
        [
            onecube,
            onecube + np.array([[dd[1]], [dd[0]], [1]]) @ np.ones((1, 4), dtype=int),
        ]
    )

    len_cube = np.prod(np.array(dim[:3]) - 1)
    face = np.tile(onecube, (len_cube, 1))

    xx, yy, zz = np.meshgrid(
        np.arange(1, dim[0]), np.arange(1, dim[1]), np.arange(1, dim[2]), indexing="ij"
    )

    # Convert subscript to linear index in column-major order (MATLAB-style)
    idx = (
        np.ravel_multi_index(
            (xx.ravel(order="F") - 1, yy.ravel(order="F") - 1, zz.ravel(order="F") - 1),
            dim,
            order="F",
        )
        + 1
    )  # 1-based index for face construction
    orig = np.tile(idx, (onecube.shape[0], 1))

    for i in range(onecube.shape[1]):
        face[:, i] = face[:, i] + orig.ravel(order="F")

    # Convert to 1-based row-unique face list (like MATLAB)
    face = np.unique(face, axis=0)
    face = np.array([list(row) for row in face])

    centroids = None
    if len(args) >= 3:
        diffvec = [np.diff(arg) for arg in args]
        xx, yy, zz = np.meshgrid(*diffvec, indexing="ij")
        centroids = (
            node[idx - 1, :]
            + 0.5
            * np.vstack(
                [xx.ravel(order="F"), yy.ravel(order="F"), zz.ravel(order="F")]
            ).T
        )

    return node, face.tolist(), centroids


# _________________________________________________________________________________________________________


def extrudecurve(c0, c1, curve, ndiv):
    if len(c0) != len(c1) or len(c0) != 3:
        raise ValueError("c0 and c1 must be 3D points of the same dimension!")

    if ndiv < 1:
        raise ValueError("ndiv must be at least 1!")

    curve = np.array(curve)
    if curve.shape[1] != 3:
        raise ValueError("curve must be a Nx3 array!")

    ncurve = curve.shape[0]
    nodes = np.zeros((ndiv * ncurve, 3))
    for i in range(ndiv):
        alpha = i / (ndiv - 1)  # linear interpolation factor
        point = (1 - alpha) * c0 + alpha * c1
        nodes[i * ncurve : (i + 1) * ncurve, :] = curve + point

    elem = np.zeros((ncurve * (ndiv - 1) * 2, 4), dtype=int)
    for i in range(ndiv - 1):
        for j in range(ncurve):
            if j < ncurve - 1:
                elem[i * ncurve * 2 + j * 2, :] = [
                    i * ncurve + j,
                    (i + 1) * ncurve + j,
                    (i + 1) * ncurve + (j + 1),
                    i * ncurve + (j + 1),
                ]
                elem[i * ncurve * 2 + j * 2 + 1, :] = [
                    (i + 1) * ncurve + j,
                    (i + 1) * ncurve + (j + 1),
                    i * ncurve + (j + 1),
                    i * ncurve + j,
                ]

    return nodes, elem


# _________________________________________________________________________________________________________


def meshcylinders(c0, c1, r, tsize=0, maxvol=0, ndiv=20):
    if np.any(np.array(r) <= 0):
        raise ValueError("Radius must be greater than zero.")

    if np.array(c0).shape != (3,) or np.array(c1).shape != (3,):
        raise ValueError("c0 and c1 must be 3D points.")

    if len(r) == 1:
        r = [r[0], r[0]]

    r = np.array(r).flatten()

    if len(r) == 2:
        r = np.array([r[0], r[0], r[1]])

    len_axis = np.linalg.norm(np.array(c1) - np.array(c0))

    if tsize == 0:
        tsize = min(r) * 0.1

    if maxvol == 0:
        maxvol = tsize**3 * 0.2

    node, face, elem = meshacylinder(c0, c1, r, tsize, maxvol, ndiv)

    return node, face, elem


# _________________________________________________________________________________________________________


def extrudesurf(no, fc, vec):
    """
    Create an enclosed surface mesh by extruding an open surface.

    Parameters:
    no : ndarray
        2D array containing the 3D node coordinates of the original surface.
    fc : ndarray
        2D array representing the triangular faces of the original surface.
        Each row corresponds to a triangle defined by indices of 3 nodes.
    vec : array or scalar
        If an array, defines the extrusion direction. If scalar, the normal vector
        is used and multiplied by this scalar for extrusion.

    Returns:
    node : ndarray
        3D node coordinates for the generated surface mesh.
    face : ndarray
        Triangular face patches of the generated surface mesh.
    """

    nlen = no.shape[0]  # Number of nodes in the original surface

    if len(vec) > 1:  # Extrude using a specified vector
        node = np.vstack([no, no + np.tile(vec, (nlen, 1))])
    else:  # Extrude along the surface normal
        node = np.vstack([no, no + vec * nodesurfnorm(no, fc)])

    face = np.vstack([fc, fc + nlen])  # Create top and bottom faces

    # Find surface edges and create side faces
    edge = surfedge(fc)
    sideface = np.hstack([edge, edge[:, [0]] + nlen])
    sideface = np.vstack([sideface, edge + nlen, edge[:, [1]]])

    face = np.vstack([face, sideface])  # Combine all faces

    # Perform mesh repair (fix degenerate elements, etc.)
    node, face = meshcheckrepair(node, face)

    return node, face


# _________________________________________________________________________________________________________


def meshanellip(c0, rr, tsize, maxvol=None):
    """
    Create the surface and tetrahedral mesh of an ellipsoid.

    Parameters:
    c0 : list or ndarray
        Center coordinates [x0, y0, z0] of the ellipsoid.
    rr : list or ndarray
        Radii of the ellipsoid. If rr is:
            - Scalar: a sphere with radius rr.
            - 1x3 or 3x1 vector: specifies the ellipsoid radii [a, b, c].
            - 1x5 or 5x1 vector: specifies [a, b, c, theta, phi], where theta and phi are rotation angles along the z and x axes.
    tsize : float
        Maximum surface triangle size on the ellipsoid.
    maxvol : float, optional
        Maximum volume of the tetrahedral elements.

    Returns:
    node : ndarray
        Node coordinates, 3 columns for x, y, and z respectively.
    face : ndarray
        Surface mesh face elements (each row has 3 vertices).
    elem : ndarray, optional
        Tetrahedral mesh elements (each row has 4 vertices).
    """

    rr = np.asarray(rr).flatten()

    if len(rr) == 1:
        rr = [rr[0], rr[0], rr[0]]  # Sphere case
    elif len(rr) == 3:
        pass  # Already in ellipsoid format
    elif len(rr) != 5:
        raise ValueError("Invalid rr length. See help for details.")

    rmax = min(rr[:3])

    if maxvol is None:
        maxvol = tsize**3  # Set maxvol based on tsize if not provided

    # Call meshunitsphere to generate unit sphere mesh
    if maxvol:
        node, face, elem = meshunitsphere(tsize / rmax, maxvol=maxvol / (rmax**3))
    else:
        node, face = meshunitsphere(tsize / rmax)

    # Scale the unit sphere to the ellipsoid
    node = node @ np.diag(rr[:3])

    if len(rr) == 5:
        theta = rr[3]
        phi = rr[4]

        # Rotation matrices for theta (z-axis) and phi (x-axis)
        Rz = np.array(
            [
                [np.cos(theta), np.sin(theta), 0],
                [-np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )

        Rx = np.array(
            [[1, 0, 0], [0, np.cos(phi), np.sin(phi)], [0, -np.sin(phi), np.cos(phi)]]
        )

        # Apply rotation to the node coordinates
        node = (Rz @ (Rx @ node.T)).T

    # Translate the ellipsoid to the center c0
    node += np.array(c0).reshape(1, 3)

    return node, face, elem if maxvol else (node, face)
