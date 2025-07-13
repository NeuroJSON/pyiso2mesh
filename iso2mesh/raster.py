"""@package docstring
Iso2Mesh for Python - Mesh-to-volume mesh rasterization

Copyright (c) 2024-2025 Qianqian Fang <q.fang at neu.edu>
"""

__all__ = ["m2v", "mesh2vol", "mesh2mask", "barycentricgrid"]

##====================================================================================
## dependent libraries
##====================================================================================

import numpy as np
import matplotlib.pyplot as plt
from iso2mesh.modify import qmeshcut

##====================================================================================
## implementations
##====================================================================================


def m2v(*args, **kwargs):
    """
    Shortcut for mesh2vol, rasterizing a tetrahedral mesh to a volume.

    Parameters:
    Same as mesh2vol function.

    Returns:
    Volumetric representation of the mesh.
    """
    return mesh2vol(*args, **kwargs)


def mesh2vol(node, elem, xi, yi=None, zi=None, **kwargs):
    """
    mesh2vol(node, elem, xi, yi=None, zi=None)

    Fast rasterization of a 3D tetrahedral mesh into a volumetric label image.

    Parameters:
        node : ndarray
            Node coordinates (N x 3) or (N x 4, with values in 4th column)
        elem : ndarray
            Tetrahedral elements (M x 4 or M x >4)
        xi, yi, zi : array-like or scalar
            Grid definitions. Supports:
              - scalar: voxel resolution
              - [Nx, Ny, Nz]: volume size
              - xi, yi, zi: actual grid vectors

    Returns:
        mask : 3D ndarray
            Voxelized volume with element labels
        weight : 4 x Nx x Ny x Nz array (if requested or values present)

    Author:
        Qianqian Fang <q.fang at neu.edu>
    """

    node = np.array(node, dtype=np.float64)
    elem = np.array(elem, dtype=np.int32)

    nodeval = None
    if node.shape[1] == 4:
        nodeval = node[:, 3].copy()
        node = node[:, :3]

    if yi is None and zi is None:
        if isinstance(xi, (int, float)):
            mn = np.min(node, axis=0)
            mx = np.max(node, axis=0)
            df = (mx - mn) / xi
        elif isinstance(xi, (list, tuple, np.ndarray)) and len(xi) == 3:
            mn = np.min(node, axis=0)
            mx = np.max(node, axis=0)
            df = (mx - mn) / np.array(xi)
        else:
            raise ValueError(
                "xi must be scalar or 3-element vector if yi and zi are not provided"
            )
        xi = np.arange(mn[0], mx[0] + df[0], df[0])
        yi = np.arange(mn[1], mx[1] + df[1], df[1])
        zi = np.arange(mn[2], mx[2] + df[2], df[2])
    else:
        xi = np.array(xi)
        yi = np.array(yi)
        zi = np.array(zi)
        df = [np.min(np.diff(xi)), np.min(np.diff(yi)), np.min(np.diff(zi))]

    if node.shape[1] != 3 or elem.shape[1] < 4:
        raise ValueError("node must have 3 columns; elem must have 4 or more columns")

    nx, ny, nz = len(xi), len(yi), len(zi)
    mask = np.zeros((nx, ny, nz))
    weight = np.zeros((4, nx, ny, nz)) if nodeval is not None else None

    for i, zval in enumerate(zi[:-1]):
        if nodeval is not None:
            cutpos, cutvalue, facedata, elemid, _ = qmeshcut(
                elem, node, nodeval, f"z={zval}"
            )
        else:
            cutpos, cutvalue, facedata, elemid, _ = qmeshcut(
                elem, node, node[:, 0], f"z={zval}"
            )
        if cutpos is None or len(cutpos) == 0:
            continue

        if weight is not None:
            maskz, weightz = mesh2mask(cutpos, facedata, xi, yi, **kwargs)
            weight[:, :, :, i] = weightz
        else:
            maskz = mesh2mask(cutpos, facedata, xi, yi, **kwargs)[0]

        idx = ~np.isnan(maskz)
        if nodeval is not None:
            eid = facedata[maskz[idx].astype(int) - 1]  # 1-based to 0-based
            maskz_flat = (
                cutvalue[eid[:, 0]] * weightz[0, idx]
                + cutvalue[eid[:, 1]] * weightz[1, idx]
                + cutvalue[eid[:, 2]] * weightz[2, idx]
                + cutvalue[eid[:, 3]] * weightz[3, idx]
            )
            maskz[idx] = maskz_flat
        else:
            maskz[idx] = elemid[(maskz[idx] - 1).astype(int)]  # adjust 1-based index

        mask[:, :, i] = maskz

    return mask, weight


def mesh2mask(node, face, xi, yi=None, hf=None, **kwargs):
    """
    Fast rasterization of a 2D mesh to an image with triangle index labels.

    Parameters:
    node: Node coordinates (N by 2 or N by 3 array)
    face: Triangle surface (N by 3 or N by 4 array)
    xi: Grid or number of divisions along x-axis
    yi: (Optional) Grid along y-axis
    hf: (Optional) Handle to a figure for faster rendering

    Returns:
    mask: 2D image where pixel values correspond to the triangle index
    weight: (Optional) Barycentric weights for each triangle
    """
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon
    import matplotlib.cm as cm

    # Determine grid size from inputs
    if isinstance(xi, (int, float)) and yi is None:
        mn = np.min(node, axis=0)
        mx = np.max(node, axis=0)
        df = (mx[:2] - mn[:2]) / xi
    elif len(xi) == 2 and yi is None:
        mn = np.min(node, axis=0)
        mx = np.max(node, axis=0)
        df = (mx[:2] - mn[:2]) / xi
    elif yi is not None:
        mx = [np.max(xi), np.max(yi)]
        mn = [np.min(xi), np.min(yi)]
        df = [np.min(np.diff(xi)), np.min(np.diff(yi))]
    else:
        raise ValueError("At least xi input is required")

    # Error checking for input sizes
    if node.shape[1] <= 1 or face.shape[1] <= 2:
        raise ValueError(
            "node must have 2 or 3 columns; face must have at least 3 columns"
        )

    fig = (
        plt.figure(figsize=(xi.size * 0.01, yi.size * 0.01), dpi=100)
        if hf is None
        else hf
    )
    ax = fig.add_subplot(111)
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(mn[0], mx[0])
    ax.set_ylim(mn[1], mx[1])
    ax.set_axis_off()

    colors = cm.jet(np.linspace(0, 1, len(face)))

    patches = []
    for i, f in enumerate(face[:, :3]):
        polygon = Polygon(node[f - 1, :2], closed=True, zorder=1)
        patches.append(polygon)

    collection = PatchCollection(
        patches,
        facecolors=colors,
        linewidths=0,
        edgecolor="face",
        antialiased=(not kwargs.get("edge", True)),
    )
    ax.add_collection(collection)

    plt.draw()
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())

    mask = np.zeros(img.shape[:2], dtype=np.int32) * np.nan
    color_vals = np.floor(colors[:, :3] * 255 + 0.5).astype(np.uint8)

    for idx, cval in enumerate(color_vals):
        match = np.all(img[:, :, :3] == cval, axis=-1)
        mask[match] = idx + 1

    mask = mask[: len(yi), : len(xi)].T
    weight = barycentricgrid(node, face, xi, yi, mask)

    if hf is None:
        plt.close(fig)
    return mask, weight


def barycentricgrid(node, face, xi, yi, mask):
    """
    Compute barycentric weights for a 2D triangle mesh over a pixel grid.

    Parameters:
        node : ndarray (N, 2 or 3)
            Node coordinates.
        face : ndarray (M, 3)
            Triangle face indices (1-based).
        xi, yi : 1D arrays
            Grid coordinate vectors.
        mask : 2D ndarray
            Label image where each pixel contains the triangle index (1-based), NaN if outside.

    Returns:
        weight : ndarray (3, H, W)
            Barycentric coordinate weights for each pixel inside a triangle.
    """
    xx, yy = np.meshgrid(xi, yi, indexing="ij")  # shape: (H, W)
    mask = mask.astype(float)
    valid_idx = ~np.isnan(mask)

    # 1-based to 0-based index
    eid = mask[valid_idx].astype(int) - 1

    # triangle vertices (all triangles)
    t1 = node[face[:, 0] - 1]
    t2 = node[face[:, 1] - 1]
    t3 = node[face[:, 2] - 1]

    # denominator (twice the area of each triangle)
    tt = (t2[:, 1] - t3[:, 1]) * (t1[:, 0] - t3[:, 0]) + (t3[:, 0] - t2[:, 0]) * (
        t1[:, 1] - t3[:, 1]
    )

    # numerator for w1 and w2 (barycentric weights)
    w1 = (t2[eid, 1] - t3[eid, 1]) * (xx[valid_idx] - t3[eid, 0]) + (
        t3[eid, 0] - t2[eid, 0]
    ) * (yy[valid_idx] - t3[eid, 1])
    w2 = (t3[eid, 1] - t1[eid, 1]) * (xx[valid_idx] - t3[eid, 0]) + (
        t1[eid, 0] - t3[eid, 0]
    ) * (yy[valid_idx] - t3[eid, 1])

    w1 = w1 / tt[eid]
    w2 = w2 / tt[eid]
    w3 = 1 - w1 - w2

    # Assemble the weight volume
    weight = np.zeros((3, *mask.shape), dtype=np.float32)
    ww = np.zeros_like(mask, dtype=np.float32)

    ww[valid_idx] = w1
    weight[0, :, :] = ww
    ww[valid_idx] = w2
    weight[1, :, :] = ww
    ww[valid_idx] = w3
    weight[2, :, :] = ww

    return weight
