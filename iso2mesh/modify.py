"""@package docstring
Iso2Mesh for Python - Mesh data queries and manipulations

Copyright (c) 2024-2025 Qianqian Fang <q.fang at neu.edu>
"""
__all__ = [
    "sms",
    "smoothsurf",
    "qmeshcut",
    "meshcheckrepair",
    "removedupelem",
    "removedupnodes",
    "removeisolatednode",
    "removeisolatedsurf",
    "surfaceclean",
    "removeedgefaces",
    "getintersecttri",
    "delendelem",
    "surfreorient",
    "sortmesh",
    "cart2sph",
    "sortrows",
    "mergemesh",
    "meshrefine",
    "mergesurf",
    "surfboolean",
    "meshresample",
    "domeshsimplify",
    "remeshsurf",
]

##====================================================================================
## dependent libraries
##====================================================================================

import numpy as np
import os
import re
import platform
import subprocess
from iso2mesh.utils import *
from iso2mesh.io import saveoff, readoff
from iso2mesh.trait import meshconn, mesheuler, finddisconnsurf

##====================================================================================
## implementations
##====================================================================================


def sms(node, face, iter=10, alpha=0.5, method="laplacianhc"):
    """
    Simplified version of surface mesh smoothing.

    Parameters:
    node: node coordinates of a surface mesh
    face: face element list of the surface mesh
    iter: smoothing iteration number (default is 10)
    alpha: scaler, smoothing parameter, v(k+1)=alpha*v(k)+(1-alpha)*mean(neighbors) (default is 0.5)
    method: smoothing method, same as in smoothsurf (default is 'laplacianhc')

    Returns:
    newnode: the smoothed node coordinates
    """

    # Compute mesh connectivity
    conn = meshconn(face, node.shape[0])[0]

    # Smooth surface mesh nodes
    newnode = smoothsurf(node[:, :3], None, conn, iter, alpha, method, alpha)

    return newnode


# _________________________________________________________________________________________________________


def smoothsurf(
    node, mask, conn0, iter, useralpha=0.5, usermethod="laplacian", userbeta=0.5
):
    """
    Smoothing a surface mesh.

    Parameters:
    node: node coordinates of a surface mesh
    mask: flag whether a node is movable (0 for movable, 1 for non-movable).
          If mask is None, all nodes are considered movable.
    conn: a list where each element contains a list of neighboring node IDs for a node
    iter: number of smoothing iterations
    useralpha: scalar smoothing parameter, v(k+1) = (1-alpha)*v(k) + alpha*mean(neighbors) (default 0.5)
    usermethod: smoothing method, 'laplacian', 'laplacianhc', or 'lowpass' (default 'laplacian')
    userbeta: scalar smoothing parameter for 'laplacianhc' (default 0.5)

    Returns:
    p: smoothed node coordinates
    """

    p = np.copy(node)
    conn = [None] * len(conn0)
    for i in range(len(conn0)):
        conn[i] = [x - 1 for x in conn0[i]]

    # If mask is empty, all nodes are considered movable
    if mask is None:
        idx = np.arange(node.shape[0])
    else:
        idx = np.where(mask == 0)[0]

    nn = len(idx)

    alpha = useralpha
    method = usermethod
    beta = userbeta

    ibeta = 1 - beta
    ialpha = 1 - alpha

    # Remove nodes without neighbors
    idx = np.array(
        [i for i in idx if (hasattr(conn[i], "__iter__") and len(conn[i]) > 0)]
    )
    nn = len(idx)

    if method == "laplacian":
        for j in range(iter):
            for i in range(nn):
                p[idx[i], :] = ialpha * p[idx[i], :] + alpha * np.mean(
                    node[conn[idx[i]], :], axis=0
                )
            node = np.copy(p)

    elif method == "laplacianhc":
        for j in range(iter):
            q = np.copy(p)
            for i in range(nn):
                p[idx[i], :] = np.mean(q[conn[idx[i]], :], axis=0)
            b = p - (alpha * node + ialpha * q)
            for i in range(nn):
                p[idx[i], :] -= beta * b[idx[i], :] + ibeta * np.mean(
                    b[conn[idx[i]], :], axis=0
                )

    elif method == "lowpass":
        beta = -1.02 * alpha
        ibeta = 1 - beta
        for j in range(iter):
            for i in range(nn):
                p[idx[i], :] = ialpha * node[idx[i], :] + alpha * np.mean(
                    node[conn[idx[i]], :], axis=0
                )
            node = np.copy(p)
            for i in range(nn):
                p[idx[i], :] = ibeta * node[idx[i], :] + beta * np.mean(
                    node[conn[idx[i]], :], axis=0
                )
            node = np.copy(p)

    return p


def qmeshcut(elem, node, value, cutat):
    """
    Fast tetrahedral mesh slicer. Intersects a 3D mesh with a plane or isosurface.

    Parameters:
    elem: Integer array (Nx4), indices of nodes forming tetrahedra
    node: Node coordinates (Nx3 array for x, y, z)
    value: Scalar array of values at each node or element
    cutat: Can define the cutting plane or isosurface using:
           - 3x3 matrix (plane by 3 points)
           - Vector [a, b, c, d] for plane (a*x + b*y + c*z + d = 0)
           - Scalar for isosurface at value=cutat
           - String expression for an implicit surface

    Returns:
    cutpos: Coordinates of intersection points
    cutvalue: Interpolated values at the intersection
    facedata: Indices forming the intersection polygons
    elemid: Tetrahedron indices where intersection occurs
    nodeid: Interpolation info for intersection points
    """

    if (
        value.shape[0] != node.shape[0]
        and value.shape[0] != elem.shape[0]
        and value.size != 0
    ):
        raise ValueError("the length of value must be either that of node or elem")

    if value.size == 0:
        cutvalue = []

    if isinstance(cutat, str) or (
        isinstance(cutat, list) and len(cutat) == 2 and isinstance(cutat[0], str)
    ):
        x, y, z = node[:, 0], node[:, 1], node[:, 2]
        if isinstance(cutat, str):
            match = re.match(r"(.+)=([^=]+)", cutat)
            if not match:
                raise ValueError('single expression must contain a single "=" sign')
            expr1, expr2 = match.groups()
            dist = eval(expr1) - eval(expr2)
        else:
            dist = eval(cutat[0]) - cutat[1]
        asign = np.where(dist <= 0, -1, 1)
    elif not isinstance(cutat, (int, float)) and (len(cutat) == 9 or len(cutat) == 4):
        if len(cutat) == 9:
            a, b, c, d = getplanefrom3pt(np.array(cutat).reshape(3, 3))
        else:
            a, b, c, d = cutat
        dist = np.dot(node, np.array([a, b, c])) + d
        asign = np.where(dist >= 0, 1, -1)
    else:
        if value.shape[0] != node.shape[0]:
            raise ValueError(
                "must use nodal value list when cutting mesh at an isovalue"
            )
        dist = value - cutat
        asign = np.where(dist > 0, 1, -1)

    esize = elem.shape[1]
    if esize == 4:
        edges = np.vstack(
            [
                elem[:, [0, 1]],
                elem[:, [0, 2]],
                elem[:, [0, 3]],
                elem[:, [1, 2]],
                elem[:, [1, 3]],
                elem[:, [2, 3]],
            ]
        )
    elif esize == 3:
        edges = np.vstack([elem[:, [0, 1]], elem[:, [0, 2]], elem[:, [1, 2]]])
    elif esize == 10:
        edges = np.vstack(
            [
                elem[:, [0, 4]],
                elem[:, [0, 7]],
                elem[:, [0, 6]],
                elem[:, [1, 4]],
                elem[:, [1, 5]],
                elem[:, [1, 8]],
                elem[:, [2, 5]],
                elem[:, [2, 6]],
                elem[:, [2, 9]],
                elem[:, [3, 7]],
                elem[:, [3, 8]],
                elem[:, [3, 9]],
            ]
        )

    edgemask = np.sum(asign[edges - 1], axis=1)
    cutedges = np.where(edgemask == 0)[0]

    cutweight = dist[edges[cutedges] - 1]
    totalweight = np.diff(cutweight, axis=1)[:, 0]
    cutweight = np.abs(cutweight / totalweight[:, np.newaxis])

    nodeid = edges[cutedges] - 1
    nodeid = np.column_stack([nodeid, cutweight[:, 1]])

    cutpos = (
        node[edges[cutedges, 0] - 1] * cutweight[:, [1]]
        + node[edges[cutedges, 1] - 1] * cutweight[:, [0]]
    )

    if value.shape[0] == node.shape[0]:
        if isinstance(cutat, (str, list)) or (
            not isinstance(cutat, (int, float)) and len(cutat) in [4, 9]
        ):
            cutvalue = (
                value[edges[cutedges, 0] - 1] * cutweight[:, [1]]
                + value[edges[cutedges, 1] - 1] * cutweight[:, [0]]
            )
        elif np.isscalar(cutat):
            cutvalue = np.full((cutpos.shape[0], 1), cutat)

    emap = np.zeros(edges.shape[0], dtype=int)
    emap[cutedges] = np.arange(1, len(cutedges) + 1)
    emap = emap.reshape((elem.shape[0], -1), order="F")

    etag = np.sum(emap > 0, axis=1)
    if esize == 3:
        linecut = np.where(etag == 2)[0]
        lineseg = emap[linecut, :]
        facedata = lineseg[lineseg > 0].reshape((2, len(linecut)), order="F").T
        elemid = linecut
        if value.shape[0] == elem.shape[0] and "cutvalue" not in locals():
            cutvalue = value[elemid]
        return cutpos, cutvalue, facedata, elemid, nodeid

    tricut = np.where(etag == 3)[0]
    quadcut = np.where(etag == 4)[0]
    elemid = np.concatenate([tricut, quadcut])

    if value.shape[0] == elem.shape[0] and "cutvalue" not in locals():
        cutvalue = value[elemid]

    tripatch = emap[tricut, :]
    tripatch = tripatch[tripatch > 0].reshape((3, len(tricut)), order="F").T

    quadpatch = emap[quadcut, :]
    quadpatch = quadpatch[quadpatch > 0].reshape((4, len(quadcut)), order="F").T

    facedata = np.vstack([tripatch[:, [0, 1, 2, 2]], quadpatch[:, [0, 1, 3, 2]]])

    return cutpos, cutvalue, facedata, elemid, nodeid


def meshcheckrepair(node, elem, opt=None, **kwargs):
    """
    Check and repair a surface mesh.

    Parameters:
    node : ndarray
        Input/output, surface node list (nn x 3).
    elem : ndarray
        Input/output, surface face element list (be x 3).
    opt : str, optional
        Options include:
            'dupnode'   : Remove duplicated nodes.
            'dupelem'   : Remove duplicated elements.
            'dup'       : Both remove duplicated nodes and elements.
            'isolated'  : Remove isolated nodes.
            'open'      : Abort if open surface is found.
            'deep'      : Call external jmeshlib to remove non-manifold vertices.
            'meshfix'   : Repair closed surface using meshfix (removes self-intersecting elements, fills holes).
            'intersect' : Test for self-intersecting elements.

    Returns:
    node : ndarray
        Repaired node list.
    elem : ndarray
        Repaired element list.
    """

    if opt in (None, "dupnode", "dup"):
        l1 = node.shape[0]
        node, elem = removedupnodes(node, elem, kwargs.get("tolerance", 0))
        l2 = node.shape[0]
        if l2 != l1:
            print(f"{l1 - l2} duplicated nodes were removed")

    if opt in (None, "duplicated", "dupelem", "dup"):
        l1 = elem.shape[0]
        elem = removedupelem(elem)
        l2 = elem.shape[0]
        if l2 != l1:
            print(f"{l1 - l2} duplicated elements were removed")

    if opt in (None, "isolated"):
        l1 = len(node)
        node, elem, _ = removeisolatednode(node, elem)
        l2 = len(node)
        if l2 != l1:
            print(f"{l1 - l2} isolated nodes were removed")

    if opt == "open":
        eg = surfedge(elem)
        if eg:
            raise ValueError(
                "Open surface found. You need to enclose it by padding zeros around the volume."
            )

    if opt in (None, "deep"):
        exesuff = fallbackexeext(getexeext(), "jmeshlib")
        deletemeshfile(mwpath("post_sclean.off"))
        saveoff(node[:, :3], elem[:, :3], mwpath("pre_sclean.off"))

        exesuff = getexeext()
        exesuff = fallbackexeext(exesuff, "jmeshlib")
        jmeshlib_path = mcpath("jmeshlib") + exesuff

        command = f'"{jmeshlib_path}" "{mwpath("pre_sclean.off")}" "{mwpath("post_sclean.off")}"'

        if ".exe" not in exesuff:
            status, output = subprocess.getstatusoutput(command)
        else:
            status, output = subprocess.getstatusoutput(
                f'"{mcpath("jmeshlib")}" "{mwpath("pre_sclean.off")}" "{mwpath("post_sclean.off")}"'
            )
        if status:
            raise RuntimeError(f"jmeshlib command failed: {output}")
        node, elem = readoff(mwpath("post_sclean.off"))

    if opt == "meshfix":
        exesuff = fallbackexeext(getexeext(), "meshfix")
        moreopt = kwargs.get("meshfixparam", " -q -a 0.01 ")
        deletemeshfile(mwpath("pre_sclean.off"))
        deletemeshfile(mwpath("pre_sclean_fixed.off"))
        saveoff(node, elem, mwpath("pre_sclean.off"))
        status = subprocess.call(
            f'"{mcpath("meshfix")}{exesuff}" "{mwpath("pre_sclean.off")}" {moreopt}',
            shell=True,
        )
        if status:
            raise RuntimeError("meshfix command failed")
        node, elem = readoff(mwpath("pre_sclean_fixed.off"))

    if opt == "intersect":
        moreopt = f' -q --no-clean --intersect -o "{mwpath("pre_sclean_inter.msh")}"'
        deletemeshfile(mwpath("pre_sclean.off"))
        deletemeshfile(mwpath("pre_sclean_inter.msh"))
        saveoff(node, elem, mwpath("pre_sclean.off"))
        subprocess.call(
            f'"{mcpath("meshfix")}{exesuff}" "{mwpath("pre_sclean.off")}" {moreopt}',
            shell=True,
        )
    return node, elem


def removedupelem(elem):
    """
    Remove doubly duplicated (folded) elements from the element list.

    Parameters:
    elem : ndarray
        List of elements (node indices).

    Returns:
    elem : ndarray
        Element list after removing the duplicated elements.
    """
    # Sort elements and remove duplicates (folded elements)
    sorted_elem = np.sort(elem, axis=1)

    # Find unique rows and their indices
    sort_elem, idx, counts = np.unique(
        sorted_elem, axis=0, return_index=True, return_inverse=True
    )

    # Histogram of element occurrences
    bins = np.bincount(counts, minlength=elem.shape[0])

    # Elements that are duplicated and their indices
    cc = bins[counts]

    # Remove folded elements
    elem = np.delete(elem, np.where((cc > 0) & (cc % 2 == 0)), axis=0)

    return elem


def removedupnodes(node, elem, tol=0):
    """
    Remove duplicated nodes from a mesh.

    Parameters:
    node : ndarray
        Node coordinates, with 3 columns for x, y, and z respectively.
    elem : ndarray or list
        Element list where each row contains the indices of nodes for each tetrahedron.
    tol : float, optional
        Tolerance for considering nodes as duplicates. Default is 0 (no tolerance).

    Returns:
    newnode : ndarray
        Node list without duplicates.
    newelem : ndarray or list
        Element list with only unique nodes.
    """

    if tol != 0:
        node = np.round(node / tol) * tol

    # Find unique rows (nodes) and map them back to elements
    newnode, I, J = np.unique(node, axis=0, return_index=True, return_inverse=True)

    if isinstance(elem, list):
        newelem = [J[e - 1] for e in elem]
    else:
        newelem = J[elem - 1]
    newelem = newelem + 1

    return newnode, newelem


def removeisolatednode(node, elem, face=None):
    """
    Remove isolated nodes: nodes that are not included in any element.

    Parameters:
    node : ndarray
        List of node coordinates.
    elem : ndarray or list
        List of elements of the mesh, can be a regular array or a list for PLCs (piecewise linear complexes).
    face : ndarray or list, optional
        List of triangular surface faces.

    Returns:
    no : ndarray
        Node coordinates after removing the isolated nodes.
    el : ndarray or list
        Element list of the resulting mesh.
    fa : ndarray or list, optional
        Face list of the resulting mesh.
    """

    oid = np.arange(node.shape[0])  # Old node indices
    elem = elem - 1

    if not isinstance(elem, list):
        idx = np.setdiff1d(oid, elem.ravel(order="F"))  # Indices of isolated nodes
    else:
        el = np.concatenate(elem)
        idx = np.setdiff1d(oid, el)

    idx = np.sort(idx)
    delta = np.zeros_like(oid)
    delta[idx] = 1
    delta = -np.cumsum(
        delta
    )  # Calculate the new node index after removal of isolated nodes

    oid = oid + delta  # Map to new index

    if not isinstance(elem, list):
        el = oid[elem]  # Update element list with new indices
    else:
        el = [oid[e] for e in elem]

    if face is not None:
        if not isinstance(face, list):
            fa = oid[face - 1]  # Update face list with new indices
        else:
            fa = [oid[f - 1] for f in face]
        fa = fa + 1
    else:
        fa = None

    el = el + 1

    no = np.delete(node, idx, axis=0)  # Remove isolated nodes

    return no, el, fa


def removeisolatedsurf(v, f, maxdiameter):
    """
    Removes disjointed surface fragments smaller than a given maximum diameter.

    Args:
    v: List of vertices (nodes) of the input surface.
    f: List of faces (triangles) of the input surface.
    maxdiameter: Maximum bounding box size for surface removal.

    Returns:
    fnew: New face list after removing components smaller than maxdiameter.
    """
    fc = finddisconnsurf(f)
    for i in range(len(fc)):
        xdia = v[fc[i] - 1, 0]
        if np.max(xdia) - np.min(xdia) <= maxdiameter:
            fc[i] = []
            continue

        ydia = v[fc[i] - 1, 1]
        if np.max(ydia) - np.min(ydia) <= maxdiameter:
            fc[i] = []
            continue

        zdia = v[fc[i] - 1, 2]
        if np.max(zdia) - np.min(zdia) <= maxdiameter:
            fc[i] = []
            continue

    fnew = np.vstack([fc[i] for i in range(len(fc)) if len(fc[i]) > 0])

    if fnew.shape[0] != f.shape[0]:
        print(
            f"Removed {f.shape[0] - fnew.shape[0]} elements of small isolated surfaces"
        )

    return fnew


def surfaceclean(f, v):
    """
    Removes surface patches that are located inside the bounding box faces.

    Args:
    f: Surface face element list (be, 3).
    v: Surface node list (nn, 3).

    Returns:
    f: Faces free of those on the bounding box.
    """
    pos = v
    mi = np.min(pos, axis=0)
    ma = np.max(pos, axis=0)

    idx0 = np.where(np.abs(pos[:, 0] - mi[0]) < 1e-6)[0]
    idx1 = np.where(np.abs(pos[:, 0] - ma[0]) < 1e-6)[0]
    idy0 = np.where(np.abs(pos[:, 1] - mi[1]) < 1e-6)[0]
    idy1 = np.where(np.abs(pos[:, 1] - ma[1]) < 1e-6)[0]
    idz0 = np.where(np.abs(pos[:, 2] - mi[2]) < 1e-6)[0]
    idz1 = np.where(np.abs(pos[:, 2] - ma[2]) < 1e-6)[0]

    f = removeedgefaces(f, v, idx0)
    f = removeedgefaces(f, v, idx1)
    f = removeedgefaces(f, v, idy0)
    f = removeedgefaces(f, v, idy1)
    f = removeedgefaces(f, v, idz0)
    f = removeedgefaces(f, v, idz1)

    return f


def removeedgefaces(f, v, idx1):
    """
    Helper function to remove edge faces based on node indices.

    Args:
    f: Surface face element list.
    v: Surface node list.
    idx1: Node indices that define the bounding box edges.

    Returns:
    f: Faces with edge elements removed.
    """
    mask = np.zeros(len(v), dtype=bool)
    mask[idx1] = True
    mask_sum = np.sum(mask[f], axis=1)
    f = f[mask_sum < 3, :]
    return f


def getintersecttri(tmppath):
    """
    Get the IDs of self-intersecting elements from TetGen.

    Args:
    tmppath: Working directory where TetGen output is stored.

    Returns:
    eid: An array of all intersecting surface element IDs.
    """
    exesuff = getexeext()
    exesuff = fallbackexeext(exesuff, "tetgen")
    tetgen_path = mcpath("tetgen") + exesuff

    command = f'"{tetgen_path}" -d "{os.path.join(tmppath, "post_vmesh.poly")}"'
    status, str_output = subprocess.getstatusoutput(command)

    eid = []
    if status == 0:
        ids = re.findall(r" #([0-9]+) ", str_output)
        eid = [int(id[0]) for id in ids]

    eid = np.unique(eid)
    return eid


def delendelem(elem, mask):
    """
    Deletes elements whose nodes are all edge nodes.

    Args:
    elem: Surface/volumetric element list (2D array).
    mask: 1D array of length equal to the number of nodes, with 0 for internal nodes and 1 for edge nodes.

    Returns:
    elem: Updated element list with edge-only elements removed.
    """
    # Find elements where all nodes are edge nodes
    badidx = np.sum(mask[elem], axis=1)

    # Remove elements where all nodes are edge nodes
    elem = elem[badidx != elem.shape[1], :]

    return elem


def surfreorient(node, face):
    """
    Reorients the normals of all triangles in a closed surface mesh to point outward.

    Args:
    node: List of nodes (coordinates).
    face: List of faces (each row contains indices of nodes for a triangle).

    Returns:
    newnode: The output node list (same as input node in most cases).
    newface: The face list with consistent ordering of vertices.
    """
    newnode, newface = meshcheckrepair(node[:, :3], face[:, :3], "deep")
    return newnode, newface


def sortmesh(origin, node, elem, ecol=None, face=None, fcol=None):
    """
    Sort nodes and elements in a mesh so that indexed nodes and elements
    are closer to each other (potentially reducing cache misses during calculations).

    Args:
        origin: Reference point for sorting nodes and elements based on distance and angles.
                If None, it defaults to node[0, :].
        node: List of nodes (coordinates).
        elem: List of elements (each row contains indices of nodes that form an element).
        ecol: Columns in elem to participate in sorting. If None, all columns are used.
        face: List of surface triangles (optional).
        fcol: Columns in face to participate in sorting (optional).

    Returns:
        no: Node coordinates in the sorted order.
        el: Element list in the sorted order.
        fc: Surface triangle list in the sorted order (if face is provided).
        nodemap: New node mapping order. no = node[nodemap, :]
    """

    # Set default origin if not provided
    if origin is None:
        origin = node[0, :]

    # Compute distances relative to the origin
    sdist = node - np.tile(origin, (node.shape[0], 1))

    # Convert Cartesian to spherical coordinates
    theta, phi, R = cart2sph(sdist[:, 0], sdist[:, 1], sdist[:, 2])
    sdist = np.column_stack((R, phi, theta))

    # Sort nodes based on spherical distance
    nval, nodemap = sortrows(sdist)
    no = node[nodemap, :]

    # Sort elements based on nodemap
    nval, nidx = sortrows(nodemap)
    el = elem.copy()

    # If ecol is not provided, sort all columns
    if ecol is None:
        ecol = np.arange(elem.shape[1])

    # Update elements with sorted node indices
    el[:, ecol] = np.sort(nidx[elem[:, ecol] - 1] + 1, axis=1)
    el = sortrows(el, ecol)[0]

    # If face is provided, sort it as well
    fc = None
    if face is not None and fcol is not None:
        fc = face.copy()
        fc[:, fcol] = np.sort(nidx[face[:, fcol] - 1] + 1, axis=1)
        fc = sortrows(fc, fcol)[0]

    return no, el, fc, nodemap


def cart2sph(x, y, z):
    """Convert Cartesian coordinates to spherical (R, phi, theta)."""
    R = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    idx = R > 0.0
    phi = np.copy(R)
    phi[idx] = z[idx] / R[idx]
    return theta, phi, R


import numpy as np


def sortrows(A, cols=None):
    """
    Sort rows of a 2D NumPy array like MATLAB's sortrows(A, cols).

    Parameters:
        A (ndarray): 2D array to sort.
        cols (list or None): List of columns to sort by.
                             Positive for ascending, negative for descending.
                             If None, sort by all columns ascending (left to right).

    Returns:
        sorted_A (ndarray): Sorted array.
        row_indices (ndarray): Indices of original rows in sorted order.
    """
    A = np.asarray(A)

    if A.ndim == 1:
        A = A[:, np.newaxis]

    n_cols = A.shape[1]

    if cols is None:
        # Default: sort by all columns, ascending
        cols = list(range(n_cols))
        ascending = [True] * n_cols
    else:
        ascending = [c > 0 for c in cols]
        cols = [abs(c) - 1 for c in cols]  # MATLAB-style (1-based to 0-based)

    # Build sort keys in reverse order (last key first)
    keys = []
    for col, asc in reversed(list(zip(cols, ascending))):
        key = A[:, col]
        if not asc:
            key = -key  # For descending sort
        keys.append(key)

    row_indices = np.lexsort(keys)
    sorted_A = A[row_indices]
    return sorted_A, row_indices


def mergemesh(node, elem, *args):
    """
    Concatenate two or more tetrahedral meshes or triangular surfaces.

    Args:
        node: Node coordinates, dimension (nn, 3).
        elem: Tetrahedral element or triangle surface, dimension (nn, 3) to (nn, 5).
        *args: Pairs of node and element arrays for additional meshes.

    Returns:
        newnode: The node coordinates after merging.
        newelem: The elements after merging.

    Note:
        Use meshcheckrepair on the output to remove duplicated nodes or elements.
        To remove self-intersecting elements, use mergesurf() instead.
    """
    # Initialize newnode and newelem with input mesh
    newnode = np.copy(node)
    newelem = np.copy(elem)

    # Check if the number of extra arguments is valid
    if len(args) > 0 and len(args) % 2 != 0:
        raise ValueError("You must give node and element in pairs")

    # Compute the Euler characteristic
    X = mesheuler(newelem)[0]

    # Add a 5th column to tetrahedral elements if not present
    if newelem.shape[1] == 4 and X >= 0:
        newelem = np.column_stack((newelem, np.ones((newelem.shape[0], 1), dtype=int)))

    # Add a 4th column to triangular elements if not present
    if newelem.shape[1] == 3:
        newelem = np.column_stack((newelem, np.ones((newelem.shape[0], 1), dtype=int)))

    # Iterate over pairs of additional meshes and merge them
    for i in range(0, len(args), 2):
        no = args[i]  # node array
        el = args[i + 1]  # element array
        baseno = newnode.shape[0]

        # Ensure consistent node dimensions
        if no.shape[1] != newnode.shape[1]:
            raise ValueError("Input node arrays have inconsistent columns")

        # Update element indices and append nodes/elements to the merged mesh
        if el.shape[1] == 5 or el.shape[1] == 4:
            el[:, :4] += baseno

            if el.shape[1] == 4 and X >= 0:
                el = np.column_stack(
                    (el, np.ones((el.shape[0], 1), dtype=int) * (i // 2 + 1))
                )
            newnode = np.vstack((newnode, no))
            newelem = np.vstack((newelem, el))
        elif el.shape[1] == 3 and newelem.shape[1] == 4:
            el[:, :3] += baseno
            el = np.column_stack(
                (el, np.ones((el.shape[0], 1), dtype=int) * (i // 2 + 1))
            )
            newnode = np.vstack((newnode, no))
            newelem = np.vstack((newelem, el))
        else:
            raise ValueError("Input element arrays have inconsistent columns")

    return newnode, newelem


def meshrefine(node, elem, *args):
    """
    Refine a tetrahedral mesh by adding new nodes or constraints.

    Args:
        node: Existing tetrahedral mesh node list.
        elem: Existing tetrahedral element list.
        args: Optional parameters for mesh refinement. This can include a face array or an options struct.

    Returns:
        newnode: Node coordinates of the refined tetrahedral mesh.
        newelem: Element list of the refined tetrahedral mesh.
        newface: Surface element list of the tetrahedral mesh.
    """
    # Default values
    sizefield = None
    newpt = None

    # If the node array has a 4th column, treat it as sizefield and reduce node array to 3 columns
    if node.shape[1] == 4:
        sizefield = node[:, 3]
        node = node[:, :3]

    # Parse optional arguments
    face = None
    opt = {}

    if len(args) == 1:
        if isinstance(args[0], dict):
            opt = args[0]
        elif len(args[0]) == len(node) or len(args[0]) == len(elem):
            sizefield = args[0]
        else:
            newpt = args[0]
    elif len(args) >= 2:
        face = args[0]
        if isinstance(args[1], dict):
            opt = args[1]
        elif len(args[1]) == len(node) or len(args[1]) == len(elem):
            sizefield = args[1]
        else:
            newpt = args[1]
    else:
        raise ValueError("meshrefine requires at least 3 inputs")

    # Check if options struct contains new nodes or sizefield
    if isinstance(opt, dict):
        if "newnode" in opt:
            newpt = opt["newnode"]
        if "sizefield" in opt:
            sizefield = opt["sizefield"]

    # Call mesh refinement functions (external tools are required here for actual mesh refinement)
    # Placeholders for calls to external mesh generation/refinement tools such as TetGen

    newnode, newelem, newface = (
        node,
        elem,
        face,
    )  # Placeholder, actual implementation needs external tools

    return newnode, newelem, newface


def mergesurf(node, elem, *args):
    """
    Merge two or more triangular meshes and split intersecting elements.

    Args:
        node: Node coordinates, dimension (nn, 3).
        elem: Triangle surface element list (nn, 3).
        *args: Additional node-element pairs for further surfaces to be merged.

    Returns:
        newnode: The node coordinates after merging, dimension (nn, 3).
        newelem: Surface elements after merging, dimension (nn, 3).
    """
    # Initialize newnode and newelem with input node and elem
    newnode = node
    newelem = elem

    # Ensure valid number of input pairs (node, elem)
    if len(args) > 0 and len(args) % 2 != 0:
        raise ValueError("You must give node and element in pairs")

    # Iterate over each pair of node and element arrays
    for i in range(0, len(args), 2):
        no = args[i]
        el = args[i + 1]
        # Perform boolean surface merge
        newnode, newelem = surfboolean(newnode, newelem, "all", no, el)

    return newnode, newelem


def surfboolean(node, elem, *varargin):
    """
    Perform boolean operations on triangular meshes and resolve intersecting elements.

    Parameters:
    node : ndarray
        Node coordinates (nn x 3)
    elem : ndarray
        Triangle surfaces (ne x 3)
    varargin : list
        Additional parameters including operators and meshes (op, node, elem)

    Returns:
    newnode : ndarray
        Node coordinates after the boolean operations.
    newelem : ndarray
        Elements after boolean operations (nn x 4) or (nhn x 5).
    newelem0 : ndarray (optional)
        For 'self' operator, returns the intersecting element list in terms of the input node list.
    """

    len_varargin = len(varargin)
    newnode = node
    newelem = elem

    if len_varargin > 0 and len_varargin % 3 != 0:
        raise ValueError(
            "You must provide operator, node, and element in a triplet form."
        )

    try:
        exename = os.environ.get("ISO2MESH_SURFBOOLEAN", "cork")
    except KeyError:
        exename = "cork"

    exesuff = fallbackexeext(getexeext(), exename)
    randseed = int("623F9A9E", 16)  # Random seed

    # Check if ISO2MESH_RANDSEED is available
    iso2mesh_randseed = os.environ.get("ISO2MESH_RANDSEED")
    if iso2mesh_randseed is not None:
        randseed = int(iso2mesh_randseed, 16)

    for i in range(0, len_varargin, 3):
        op = varargin[i]
        no = varargin[i + 1]
        el = varargin[i + 2]
        opstr = op

        # Map operations to proper string values
        op_map = {
            "or": "union",
            "xor": "all",
            "and": "isct",
            "-": "diff",
            "self": "solid",
        }
        opstr = op_map.get(op, op)

        tempsuff = "off"
        deletemeshfile(mwpath(f"pre_surfbool*.{tempsuff}"))
        deletemeshfile(mwpath("post_surfbool.off"))

        if opstr == "all":
            deletemeshfile(mwpath("s1out2.off"))
            deletemeshfile(mwpath("s1in2.off"))
            deletemeshfile(mwpath("s2out1.off"))
            deletemeshfile(mwpath("s2in1.off"))

        if op == "decouple":
            if "node1" not in locals():
                node1 = node
                elem1 = elem
                newnode[:, 3] = 1
                newelem[:, 3] = 1
            opstr = " --decouple-inin 1 --shells 2"
            saveoff(node1[:, :3], elem1[:, :3], mwpath("pre_decouple1.off"))
            if no.shape[1] != 3:
                opstr = f"-q --shells {no}"
                cmd = f'cd "{mwpath()}" && "{mcpath("meshfix")}{exesuff}" "{mwpath("pre_decouple1.off")}" {opstr}'
            else:
                saveoff(no[:, :3], el[:, :3], mwpath("pre_decouple2.off"))
                cmd = f'cd "{mwpath()}" && "{mcpath("meshfix")}{exesuff}" "{mwpath("pre_decouple1.off")}" "{mwpath("pre_decouple2.off")}" {opstr}'
        else:
            saveoff(newnode[:, :3], newelem[:, :3], mwpath(f"pre_surfbool1.{tempsuff}"))
            saveoff(no[:, :3], el[:, :3], mwpath(f"pre_surfbool2.{tempsuff}"))
            cmd = f'cd "{mwpath()}" && "{mcpath(exename)}{exesuff}" -{opstr} "{mwpath(f"pre_surfbool1.{tempsuff}")}" "{mwpath(f"pre_surfbool2.{tempsuff}")}" "{mwpath("post_surfbool.off")}" -{randseed}'

        status, outstr = subprocess.getstatusoutput(cmd)
        if status != 0 and op != "self":
            raise RuntimeError(
                f"surface boolean command failed:\n{cmd}\nERROR: {outstr}\n"
            )

        if op == "self":
            if "NOT SOLID" not in outstr:
                print("No self-intersection was found!")
                return None, None
            else:
                print("Input mesh is self-intersecting")
                return np.array([1]), np.array([])

    # Further processing based on the operation 'all'
    if opstr == "all":
        nnode, nelem = readoff(mwpath("s1out2.off"))
        newelem = np.hstack([nelem, np.ones((nelem.shape[0], 1))])
        newnode = np.hstack([nnode, np.ones((nnode.shape[0], 1))])
        nnode, nelem = readoff(mwpath("s1in2.off"))
        newelem = np.vstack(
            [
                newelem,
                np.hstack([nelem + newnode.shape[0], np.ones((nelem.shape[0], 1)) * 3]),
            ]
        )
        newnode = np.vstack(
            [newnode, np.hstack([nnode, np.ones((nnode.shape[0], 1)) * 3])]
        )
        nnode, nelem = readoff(mwpath("s2out1.off"))
        newelem = np.vstack(
            [
                newelem,
                np.hstack([nelem + newnode.shape[0], np.ones((nelem.shape[0], 1)) * 2]),
            ]
        )
        newnode = np.vstack(
            [newnode, np.hstack([nnode, np.ones((nnode.shape[0], 1)) * 2])]
        )
        nnode, nelem = readoff(mwpath("s2in1.off"))
        newelem = np.vstack(
            [
                newelem,
                np.hstack([nelem + newnode.shape[0], np.ones((nelem.shape[0], 1)) * 4]),
            ]
        )
        newnode = np.vstack(
            [newnode, np.hstack([nnode, np.ones((nnode.shape[0], 1)) * 4])]
        )
    else:
        newnode, newelem = readoff(mwpath("post_surfbool.off"))

    return newnode, newelem


def meshresample(v, f, keepratio):
    """
    Resample mesh using the CGAL mesh simplification utility.

    Parameters:
    v : ndarray
        List of nodes.
    f : ndarray
        List of surface elements (each row representing a triangle).
    keepratio : float
        Decimation rate, a number less than 1 representing the percentage of elements to keep after sampling.

    Returns:
    node : ndarray
        Node coordinates of the resampled surface mesh.
    elem : ndarray
        Element list of the resampled surface mesh.
    """

    node, elem = domeshsimplify(v, f, keepratio)

    if len(node) == 0:
        print(
            "Input mesh contains topological defects. Attempting to repair with meshcheckrepair..."
        )
        vnew, fnew = meshcheckrepair(v, f)
        node, elem = domeshsimplify(vnew, fnew, keepratio)

    # Remove duplicate nodes
    node, I, J = np.unique(node, axis=0, return_index=True, return_inverse=True)
    elem = J[elem - 1] + 1

    saveoff(node, elem, mwpath("post_remesh.off"))

    return node, elem


def domeshsimplify(v, f, keepratio):
    """
    Perform the actual mesh resampling using CGAL's simplification utility.

    Parameters:
    v : ndarray
        List of nodes.
    f : ndarray
        List of surface elements.
    keepratio : float
        Decimation rate, a number less than 1.

    Returns:
    node : ndarray
        Node coordinates after simplification.
    elem : ndarray
        Element list after simplification.
    """

    exesuff = getexeext()
    exesuff = fallbackexeext(exesuff, "cgalsimp2")

    # Save the input mesh in OFF format
    saveoff(v, f, mwpath("pre_remesh.off"))

    # Delete the old remeshed file if it exists
    deletemeshfile(mwpath("post_remesh.off"))

    # Build and execute the command for CGAL simplification
    cmd = f'"{mcpath("cgalsimp2")}{exesuff}" "{mwpath("pre_remesh.off")}" {keepratio} "{mwpath("post_remesh.off")}"'
    status = subprocess.call(cmd, shell=True)

    if status != 0:
        raise RuntimeError("cgalsimp2 command failed")

    # Read the resampled mesh
    node, elem = readoff(mwpath("post_remesh.off"))

    return node, elem


def remeshsurf(node, face, opt):
    """
    remeshsurf(node, face, opt)

    Remesh a triangular surface, output is guaranteed to be free of self-intersecting elements.
    This function can both downsample or upsample a mesh.

    Parameters:
        node: list of nodes on the input surface mesh, 3 columns for x, y, z
        face: list of triangular elements on the surface, [n1, n2, n3, region_id]
        opt: function parameters
            opt.gridsize: resolution for the voxelization of the mesh
            opt.closesize: if there are openings, set the closing diameter
            opt.elemsize: the size of the element of the output surface
            If opt is a scalar, it defines the elemsize and gridsize = opt / 4

    Returns:
        newno: list of nodes on the resulting surface mesh, 3 columns for x, y, z
        newfc: list of triangular elements on the surface, [n1, n2, n3, region_id]
    """

    # Step 1: convert the old surface to a volumetric image
    p0 = np.min(node, axis=0)
    p1 = np.max(node, axis=0)

    if isinstance(opt, dict):
        dx = opt.get("gridsize", None)
    else:
        dx = opt / 4

    x_range = np.arange(p0[0] - dx, p1[0] + dx, dx)
    y_range = np.arange(p0[1] - dx, p1[1] + dx)
    z_range = np.arange(p0[2] - dx, p1[2] + dx)

    img = surf2vol(node, face, x_range, y_range, z_range)

    # Compute surface edges
    eg = surfedge(face)

    closesize = 0
    if eg.size > 0 and isinstance(opt, dict):
        closesize = opt.get("closesize", 0)

    # Step 2: fill holes in the volumetric binary image
    img = fillholes3d(img, closesize)

    # Step 3: convert the filled volume to a new surface
    if isinstance(opt, dict):
        if "elemsize" in opt:
            opt["radbound"] = opt["elemsize"] / dx
            newno, newfc, _, _ = v2s(img, 0.5, opt, "cgalsurf")
    else:
        opt = {"radbound": opt / dx}
        newno, newfc, _, _ = v2s(img, 0.5, opt, "cgalsurf")

    # Adjust new nodes to match original coordinates
    newno[:, 0:3] *= dx
    newno[:, 0] += p0[0]
    newno[:, 1] += p0[1]
    newno[:, 2] += p0[2]

    return newno, newfc
