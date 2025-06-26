"""@package docstring
Iso2Mesh for Python - File I/O module

Copyright (c) 2024-2025 Qianqian Fang <q.fang at neu.edu>
"""

__all__ = [
    "affinemap",
    "meshinterp",
    "meshremap",
    "proj2mesh",
    "dist2surf",
    "regpt2surf",
]

##====================================================================================
## dependent libraries
##====================================================================================

import numpy as np
from iso2mesh.trait import nodesurfnorm
from iso2mesh.line import linextriangle

##====================================================================================
## implementations
##====================================================================================


def affinemap(pfrom, pto):
    """
    Calculate an affine transform (A matrix and b vector) to map 3D vertices
    from one space to another using least square solutions.

    Parameters:
    pfrom : numpy array (n x 3), points in the original space
    pto : numpy array (n x 3), points in the mapped space

    Returns:
    A : numpy array (3 x 3), the affine transformation matrix
    b : numpy array (3 x 1), the translation vector
    """
    ptnum = pfrom.shape[0]

    if pto.shape[0] != ptnum:
        raise ValueError("Two inputs should have the same size")

    bsubmat = np.eye(3)
    amat = np.zeros((ptnum * 3, 9))

    for i in range(ptnum):
        amat[i * 3 : (i + 1) * 3, :] = np.kron(bsubmat, pfrom[i, :])

    amat = np.hstack([amat, np.tile(bsubmat, (ptnum, 1))])
    bvec = pto.T.flatten()

    x, _, _, _ = np.linalg.lstsq(amat, bvec, rcond=None)
    A = x[:9].reshape(3, 3).T
    b = x[-3:]

    return A, b


def meshinterp(fromval, elemid, elembary, fromelem, initval=None):
    """
    Interpolate nodal values from the source mesh to the target mesh based on a linear interpolation.

    Args:
        fromval: Values defined at the source mesh nodes. The row or column number
                 must be the same as the source mesh node count (matching elemid length).
        elemid: IDs of the source mesh element that encloses the target mesh nodes; a vector of length
                equal to the target mesh node count.
        elembary: Barycentric coordinates of each target mesh node within the source mesh elements.
                  The sum of each row is 1, with 3 or 4 columns.
        fromelem: The element list of the source mesh.
        initval: Optional initial values for the target nodes.

    Returns:
        newval: A 2D array where rows equal the target mesh nodes, and columns equal
                the value numbers defined at each source mesh node.
    """

    # If fromval is a single row, convert it to a column vector
    if fromval.shape[0] == 1:
        fromval = fromval.T

    # Find valid indices (non-NaN element IDs)
    idx = np.where(~np.isnan(elemid))[0]

    # Reshape fromval to match the size of elembary and fromelem
    allval = np.reshape(
        fromval[fromelem[elemid[idx], :], :],
        (len(idx), elembary.shape[1], fromval.shape[1]),
    )

    # Perform linear interpolation using barycentric coordinates
    tmp = np.array(
        [np.sum(elembary[idx, :] * x, axis=1) for x in np.rollaxis(allval, 2)]
    )

    # Initialize newval with initval or NaN
    if initval is not None:
        newval = initval
    else:
        newval = np.full((len(elemid), fromval.shape[1]), np.nan)

    # Assign interpolated values to newval at valid indices
    newval[idx, :] = np.squeeze(np.stack(tmp, axis=2))

    return newval


def meshremap(fromval, elemid, elembary, toelem, nodeto):
    """
    Redistribute nodal values from the source mesh to the target mesh
    so that the sum of each property on each mesh is the same.

    Parameters:
    fromval: Values defined at the source mesh nodes; should be a 1D or 2D array
             with the same number of rows or columns as the source mesh node count.
    elemid: IDs of the target mesh element that encloses the source mesh nodes; a vector.
    elembary: Barycentric coordinates of each source mesh node within the target mesh elements;
              sum of each row is 1, expect 3 or 4 columns (or N-D).
    toelem: Element list of the target mesh.
    nodeto: Total number of target mesh nodes.

    Returns:
    newval: A 2D array with rows equal to the target mesh nodes and columns equal to
            the value numbers defined at each source mesh node.
    """

    # Ensure fromval is a column vector if it is a row vector
    if fromval.shape[0] == 1:
        fromval = fromval.T

    # Ensure fromval's number of columns matches the length of elemid
    if fromval.shape[1] == len(elemid):
        fromval = fromval.T

    # Initialize output array newval
    newval = np.zeros((nodeto, fromval.shape[1]))

    # Ignore NaN elements
    idx = ~np.isnan(elemid)
    fromval = fromval[idx, :]
    elembary = elembary[idx, :]
    idx = elemid[idx]

    # Compute nodal values weighted by barycentric coordinates
    nodeval = np.repeat(
        fromval[:, np.newaxis, :], elembary.shape[1], axis=1
    ) * np.repeat(elembary[:, np.newaxis, :], fromval.shape[1], axis=1).transpose(
        0, 2, 1
    )

    # Accumulate contributions to target mesh nodes
    for i in range(elembary.shape[1]):
        ix, iy = np.meshgrid(toelem[idx, i], np.arange(fromval.shape[1]))
        nval = nodeval[:, :, i].T
        newval += np.add.at(newval, (ix.flatten(), iy.flatten()), nval.flatten())

    return newval


def proj2mesh(v, f, pt, nv=None, cn=None, radmax=None):
    """
    Project a point cloud onto the surface mesh (triangular surface only).

    Parameters:
        v : ndarray
            Node coordinates of the surface mesh (nn x 3)
        f : ndarray
            Element list of the surface mesh (triangular or cubic, use only 3 columns for triangle)
        pt : ndarray
            Points to be projected, with 3 columns for x, y, and z respectively
        nv : ndarray, optional
            Nodal normals (size: v.shape[0] x 3), calculated by nodesurfnorm
        cn : ndarray, optional
            Integer vector of closest surface node indices for each point in pt (from dist2surf)
        radmax : float, optional
            If specified, limits search for elements to those within a bounding box centered at the point

    Returns:
        newpt : ndarray
            Projected points from pt
        elemid : ndarray
            Indices of the surface triangle that contains each projected point
        weight : ndarray
            Barycentric coordinates (weights) for each projected point

    -- this function is part of iso2mesh toolbox (http://iso2mesh.sf.net)
    """
    cent = np.mean(v, axis=0)
    enum = len(f)
    ec = np.reshape(v[f[:, :3] - 1].transpose(1, 2, 0), (3, 3, enum))
    centroid = np.mean(ec, axis=1)
    newpt = np.zeros_like(pt)
    elemid = np.zeros(pt.shape[0], dtype=int)
    weight = np.zeros((pt.shape[0], 3))

    idoldmesh = np.any(np.all(pt[:, None, :] == v[None, :, :], axis=2), axis=1)
    idnode = np.where(idoldmesh)[0]
    if idnode.size > 0:
        for idx in idnode:
            matches = np.where(
                np.all(
                    f[:, None, :] == np.where((v == pt[idx]).all(axis=1))[0][0] + 1,
                    axis=2,
                )
            )
            if matches[0].size > 0:
                newpt[idx, :] = pt[idx, :]
                elemid[idx] = matches[0][0] + 1
                weight[idx, matches[1][0]] = 1

    if nv is not None and cn is not None:
        direction = nv[cn - 1, :]
        if radmax is not None:
            radlimit = radmax
        else:
            radlimit = -1
    else:
        direction = pt - cent
        radlimit = -1

    for t in range(pt.shape[0]):
        if idoldmesh[t]:
            continue

        maxdist = np.linalg.norm(pt[t, :] - cent)
        if radlimit > 0:
            maxdist = radlimit

        mask = np.all(np.abs(centroid.T - pt[t]) < maxdist, axis=1)
        idx = np.where(mask)[0]
        dist = centroid[:, idx] - pt[t, :, None]
        c0 = np.sum(dist**2, axis=0)

        sorted_idx = np.argsort(c0)

        for i in range(len(idx)):
            inside, p, w = linextriangle(
                pt[t, :],
                pt[t, :] + direction[t, :],
                v[f[idx[sorted_idx[i]], :3] - 1, :],
            )
            if inside:
                newpt[t, :] = p
                weight[t, :] = w
                elemid[t] = idx[sorted_idx[i]] + 1
                break

    return newpt, elemid, weight


def dist2surf(node, nv, p, cn=None):
    """
    Calculate the distances from a point cloud to a surface, and return
    the indices of the closest surface node.

    Parameters:
        node : ndarray
            Node coordinates of the surface mesh (nn x 3).
        nv : ndarray
            Nodal normals (vector) calculated from nodesurfnorm(), shape (nn x 3).
        p : ndarray
            Points to be calculated, shape (N x 3).
        cn : ndarray, optional
            If provided, an integer vector of indices of the closest surface nodes.

    Returns:
        d2surf : ndarray
            Distances from each point in p to the surface.
        cn : ndarray
            Indices of the closest surface nodes.

    -- this function is part of "metch" toolbox, see COPYING for license
    """

    if cn is None:
        nn = node.shape[0]
        pnum = p.shape[0]
        mindist = np.zeros(pnum)
        cn = np.zeros(pnum, dtype=int)
        for i in range(pnum):
            d0 = node - np.tile(p[i, :], (nn, 1))
            d0 = np.sum(d0 * d0, axis=1)
            cn[i] = np.argmin(d0)
            mindist[i] = d0[cn[i]]
    d2surf = np.abs(np.sum(nv[cn, :] * (p - node[cn, :]), axis=1))

    return d2surf, cn


def regpt2surf(node, elem, p, pmask, A0, b0, cmask, maxiter):
    """
    Perform point cloud registration to a triangular surface
    (surface can be either triangular or cubic), using Gauss-Newton method.

    Parameters:
        node : ndarray
            Node coordinates of the surface mesh (nn x 3).
        elem : ndarray
            Element list of the surface mesh (triangular: 3 columns, cubic: 4 columns).
        p : ndarray
            Points to be registered (N x 3).
        pmask : ndarray
            Mask vector of same length as p. If pmask[i] == -1, the point is free;
            if 0, it is fixed; if n > 0, its distance to node[n-1] is minimized.
        A0 : ndarray
            Initial guess for affine A matrix (3x3).
        b0 : ndarray
            Initial guess for affine b vector (3,).
        cmask : ndarray
            Binary vector of length 12, indicates which of [A.flatten(); b] to optimize.
        maxiter : int
            Maximum number of optimization iterations.

    Returns:
        A : ndarray
            Updated affine transformation matrix (3x3).
        b : ndarray
            Updated translation vector (3,).
        newpos : ndarray
            Transformed positions of input points.
    """

    A = A0.copy()
    b = b0.reshape(-1)

    # Wrap A and b into single vector C
    C = np.concatenate([A.flatten(), b])
    delta = 1e-4

    newpos = (C[:9].reshape(3, 3) @ p.T + C[9:].reshape(3, 1)).T
    nv = nodesurfnorm(node, elem)

    clen = len(C)
    cuplist = np.where(cmask == 1)[0]
    pfree = np.where(pmask < 0)[0]
    pfix = np.where(pmask > 0)[0]

    for iter in range(maxiter):
        dist0 = np.zeros(len(pfree) + len(pfix))

        if len(pfree) > 0:
            dist0[pfree], cn0 = dist2surf(node, nv, newpos[pfree])
        else:
            cn0 = []

        if len(pfix) > 0:
            fixdist = node[pmask[pfix] - 1] - newpos[pfix]
            dist0[pfix] = np.sqrt(np.sum(fixdist**2, axis=1))

        print(f"iter={iter+1} error={np.sum(np.abs(dist0))}")

        J = np.zeros((len(dist0), clen))
        for i in range(clen):
            if cmask[i] == 0:
                continue
            dC = C.copy()
            dC[i] = C[i] * (1 + delta) if C[i] != 0 else C[i] + delta
            newp = (dC[:9].reshape(3, 3) @ p.T + dC[9:].reshape(3, 1)).T

            dist = np.zeros(len(dist0))
            if len(pfree) > 0:
                if len(cn0) == len(pfree):
                    dist[pfree], _ = dist2surf(node, nv, newp[pfree], cn0)
                else:
                    dist[pfree], _ = dist2surf(node, nv, newp[pfree])
            if len(pfix) > 0:
                fixdist = node[pmask[pfix] - 1] - newp[pfix]
                dist[pfix] = np.sqrt(np.sum(fixdist**2, axis=1))

            J[:, i] = (dist - dist0) / (dC[i] - C[i])

        wj = np.sqrt(np.sum(J**2, axis=0))
        J[:, cuplist] = J[:, cuplist] / wj[cuplist]

        dC = np.linalg.lstsq(J[:, cuplist], dist0, rcond=None)[0] / wj[cuplist]
        C[cuplist] -= 0.5 * dC

        newpos = (C[:9].reshape(3, 3) @ p.T + C[9:].reshape(3, 1)).T

    A = C[:9].reshape(3, 3)
    b = C[9:]
    return A, b, newpos
