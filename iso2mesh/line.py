"""@package docstring
Iso2Mesh for Python - Polyline processing utilities

Copyright (c) 2024-2025 Qianqian Fang <q.fang at neu.edu>
"""

__all__ = [
    "linextriangle",
    "getplanefrom3pt",
    "polylinelen",
    "closestnode",
    "polylineinterp",
    "polylinesimplify",
]
##====================================================================================
## dependent libraries
##====================================================================================

import numpy as np

##====================================================================================
## implementations
##====================================================================================


def linextriangle(p0, p1, plane):
    """
    Calculate the intersection of a 3D line (defined by two points) with a plane (defined by 3 points).

    Parameters:
        p0 : array_like
            A 3D point in the form of (x, y, z)
        p1 : array_like
            Another 3D point, p0 and p1 determine the line
        plane : ndarray
            A 3x3 matrix, each row is a 3D point defining the triangle on the plane

    Returns:
        isinside : bool
            True if the intersection is within the triangle defined by the plane; False otherwise
        pt : ndarray
            Coordinates of the intersection point
        coord : ndarray
            Barycentric coordinates of the intersection point (if inside); otherwise all zeros.

    -- this function is part of iso2mesh toolbox (http://iso2mesh.sf.net)
    """

    a, b, c, d = getplanefrom3pt(plane)

    if a**2 + b**2 + c**2 == 0.0:
        raise ValueError("Degenerated plane")

    dl_n = np.dot([a, b, c], np.array(p1) - np.array(p0))

    if dl_n == 0.0:
        raise ValueError("Degenerated line")

    # Solve for the intersection point
    t = -(a * p0[0] + b * p0[1] + c * p0[2] + d) / dl_n
    pt = np.array(p0) + t * (np.array(p1) - np.array(p0))

    dist = np.sum(np.abs(np.diff(plane, axis=0)), axis=0)
    imax = np.argsort(dist)
    if dist[imax[1]] == 0.0:
        raise ValueError("Degenerated triangle")

    goodidx = imax[1:]

    ptproj = pt[goodidx]
    mat0 = np.vstack([plane[:, goodidx].T, ptproj, np.ones(4)])

    isinside = False
    coord = np.array([0.0, 0.0, 0.0])

    det1 = np.linalg.det(mat0[:, [3, 1, 2]])
    det2 = np.linalg.det(mat0[:, [0, 3, 2]])
    if det1 * det2 < 0:
        return isinside, pt, coord

    det3 = np.linalg.det(mat0[:, [0, 1, 3]])
    if det2 * det3 < 0:
        return isinside, pt, coord
    if det1 * det3 < 0:
        return isinside, pt, coord

    isinside = True
    det0 = np.linalg.det(mat0[:, [0, 1, 2]])
    coord = np.array([det1, det2, det3]) / det0

    return isinside, pt, coord


def getplanefrom3pt(plane):
    """
    Define a plane equation ax + by + cz + d = 0 from three 3D points.

    Parameters:
        plane : ndarray
            A 3x3 matrix with each row specifying a 3D point (x, y, z).

    Returns:
        a, b, c, d : float
            The coefficients of the plane equation ax + by + cz + d = 0.

    -- this function is part of iso2mesh toolbox (http://iso2mesh.sf.net)
    """

    x = plane[:, 0]
    y = plane[:, 1]
    z = plane[:, 2]

    # Compute the plane equation a*x + b*y + c*z + d = 0
    a = y[0] * (z[1] - z[2]) + y[1] * (z[2] - z[0]) + y[2] * (z[0] - z[1])
    b = z[0] * (x[1] - x[2]) + z[1] * (x[2] - x[0]) + z[2] * (x[0] - x[1])
    c = x[0] * (y[1] - y[2]) + x[1] * (y[2] - y[0]) + x[2] * (y[0] - y[1])
    d = -np.linalg.det(plane)

    return a, b, c, d


def closestnode(node, p):
    """
    Find the closest point in a node list and return its index.

    Parameters:
        node : ndarray
            Each row is an N-D node coordinate.
        p : ndarray
            A given position in the same space (1D array).

    Returns:
        idx : int
            The index of the position in the node list that has the shortest
            Euclidean distance to the position p.
        dist : float
            The squared distance between p and the closest node.

    -- this function is part of brain2mesh toolbox (http://mcx.space/brain2mesh)
       License: GPL v3 or later, see LICENSE.txt for details
    """

    dd = node - np.tile(p, (node.shape[0], 1))
    dist_sq = np.sum(dd * dd, axis=1)
    idx = np.argmin(dist_sq)
    dist = dist_sq[idx]
    return idx + 1, dist


def polylinelen(node, p0=None, p1=None, pmid=None):
    """
    Calculate the polyline line segment length vector in sequential order

    Parameters:
        node : ndarray
            An N x 3 array defining each vertex of the polyline in sequential order.
        p0 : int or ndarray, optional
            A given node to define the start of the polyline (index or coordinate).
            If not defined, start position is assumed to be the first node.
        p1 : int or ndarray, optional
            A given node to define the end of the polyline (index or coordinate).
            If not defined, end position is assumed to be the last node.
        pmid : int or ndarray, optional
            A given node that sits between p0 and p1. If not defined, the floor of the middle index is used.

    Returns:
        len : ndarray
            The length of each segment between the start and end points.
        node : ndarray
            The node list between the start and end points of the polyline.
        inputreversed : bool
            1 if the input node is reversed from p0 to pmid to p1, 0 otherwise.

    -- this function is part of brain2mesh toolbox (http://mcx.space/brain2mesh)
       License: GPL v3 or later, see LICENSE.txt for details
    """

    if p1 is None:
        p1 = node.shape[0]
        if p0 is None:
            p0 = 1

    if pmid is None:
        pmid = np.floor((p0 + p1) * 0.5)

    if isinstance(p0, (list, np.ndarray)) and np.shape(p0)[-1] == 3:
        p0, _ = closestnode(node, np.asarray(p0))
    if isinstance(p1, (list, np.ndarray)) and np.shape(p1)[-1] == 3:
        p1, _ = closestnode(node, np.asarray(p1))
    if isinstance(pmid, (list, np.ndarray)) and np.shape(pmid)[-1] == 3:
        pmid, _ = closestnode(node, np.asarray(pmid))

    # convert to 0-based indices
    p0 -= 1
    p1 -= 1
    pmid -= 1

    if p0 < pmid and pmid < p1:
        inputreversed = 0
        node = node[range(p0, p1 + 1), :]
    elif p0 < pmid and p1 < pmid:
        inputreversed = min(p0, p1) == p0
        idx_range = list(range(min(p0, p1), -1, -1)) + list(
            range(node.shape[0] - 1, max(p0, p1) - 1, -1)
        )
        node = node[idx_range, :]
        if not inputreversed:
            node = np.flipud(node)
    elif p0 > pmid and pmid > p1:
        inputreversed = 1
        node = node[range(p0, p1 - 1, -1), :]
    elif p0 > pmid and p1 > pmid:
        inputreversed = max(p0, p1) == p1
        idx_range = list(range(max(p0, p1), node.shape[0])) + list(
            range(0, min(p0, p1) + 1)
        )
        node = node[idx_range, :]
        if inputreversed:
            node = np.flipud(node)

    seg = node[:-1, :] - node[1:, :]
    length = np.sqrt(np.sum(seg * seg, axis=1))
    return length, node, inputreversed


def polylineinterp(polylen, length, nodes=None):
    """
    Find the polyline segment indices and interpolation weights for a
    specified total length or a set of lengths.

    Parameters:
        polylen : array_like
            A 1D vector sequentially recording the length of each segment
            of a polyline. The first number is the length of the 1st segment, and so on.
        length : float or array_like
            A scalar or array specifying the total length(s) to interpolate.
        nodes : ndarray, optional
            If provided, should be an array with shape (len(polylen)+1, N),
            where each row is a coordinate for a node along the polyline.

    Returns:
        idx : ndarray
            The indices (1-based) of the polyline segments where each length ends.
            NaN if length > sum(polylen). Indexing starts at 1 (like MATLAB).
        weight : ndarray
            Interpolation weights (0-1) toward the end node of the segment.
        newnodes : ndarray, optional
            Interpolated node positions corresponding to `length`.

    Example:
        polylen = [2, 2, 1, 7, 10]
        idx, weight = polylineinterp(polylen, [3, 12, 7])
    """

    polylen = np.asarray(polylen, dtype=float).flatten()
    length = np.atleast_1d(length).astype(float)
    cumlen = np.concatenate(([0], np.cumsum(polylen)))

    idx = np.full(length.shape, np.nan)
    weight = np.zeros(length.shape)

    compute_nodes = nodes is not None and nodes.shape[0] == len(polylen) + 1
    newnodes = np.zeros((len(length), nodes.shape[1])) if compute_nodes else None

    for i in range(len(length)):
        pos = np.histogram(length[i], bins=cumlen)[0]
        if np.any(pos == 1):
            seg = np.searchsorted(cumlen, length[i], side="right")
            if seg == len(cumlen):
                idx[i] = seg - 1
                weight[i] = 1.0
                if compute_nodes:
                    newnodes[i, :] = nodes[-1, :]
            elif seg <= len(polylen):
                idx[i] = seg
                weight[i] = (length[i] - cumlen[seg - 1]) / polylen[seg - 1]
                if compute_nodes:
                    newnodes[i, :] = (1 - weight[i]) * nodes[seg - 1, :] + weight[
                        i
                    ] * nodes[seg, :]

    idx[idx > len(polylen)] = np.nan

    return idx, weight, newnodes


def polylinesimplify(nodes, minangle=None):
    """
    Calculate a simplified polyline by removing nodes where two adjacent
    segments have an angle less than a specified limit.

    Parameters:
        nodes : ndarray
            An N x 3 array defining each vertex of the polyline in sequential order.
        minangle : float, optional
            Minimum segment angle in radians. If not given, defaults to 0.75 * pi.

    Returns:
        newnodes : ndarray
            The updated node list; start/end will not be removed.
        len : ndarray
            The length of each segment between the start and end points.

    -- this function is part of brain2mesh toolbox (http://mcx.space/brain2mesh)
    """

    if minangle is None:
        minangle = 0.75 * np.pi

    def segvec(n1, n2):
        v = n2 - n1
        normals = np.linalg.norm(v, axis=1, keepdims=True)
        return v / normals

    v = segvec(nodes[:-1], nodes[1:])
    dotprod = np.sum(-v[:-1] * v[1:], axis=1)
    ang = np.arccos(np.clip(dotprod, -1.0, 1.0))

    newnodes = nodes.copy()
    newv = v.copy()
    newang = ang.copy()

    idx = np.where(newang < minangle)[0]

    while len(idx) > 0:
        newnodes = np.delete(newnodes, idx + 1, axis=0)
        newv = np.delete(newv, idx + 1, axis=0)
        newang = np.delete(newang, idx, axis=0)

        idx = idx - np.arange(len(idx))
        idx = np.unique(idx)

        idx1 = idx[idx < newnodes.shape[0] - 1]
        if len(idx1) > 0:
            newv[idx1, :] = segvec(newnodes[idx1], newnodes[idx1 + 1])
        idx1 = idx[idx < newv.shape[0] - 1]
        if len(idx1) > 0:
            newang[idx1] = np.arccos(np.sum(-newv[idx1] * newv[idx1 + 1], axis=1))
        idx0 = idx[idx > 0]
        if len(idx0) > 0:
            newang[idx0 - 1] = np.arccos(np.sum(-newv[idx0 - 1] * newv[idx0], axis=1))

        idx = np.where(newang < minangle)[0]

    if newnodes.shape[0] > 1:
        lenvec = newnodes[:-1] - newnodes[1:]
        length = np.sqrt(np.sum(lenvec**2, axis=1))
    else:
        length = np.array([])

    return newnodes, length
