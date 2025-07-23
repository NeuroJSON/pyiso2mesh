"""@package docstring
Iso2Mesh for Python - Primitive shape meshing functions

Copyright (c) 2024-2025 Qianqian Fang <q.fang at neu.edu>
"""

__all__ = [
    "plotsurf",
    "plotasurf",
    "plotmesh",
    "plotedges",
    "plottetra",
]

##====================================================================================
## dependent libraries
##====================================================================================

import numpy as np
import matplotlib.pyplot as plt
from iso2mesh.trait import volface, meshcentroid

COLOR_OFFSET = 3
# _________________________________________________________________________________________________________


def plotsurf(node, face, *args, **kwargs):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.colors import Normalize

    rngstate = np.random.get_state()

    randseed = int("623F9A9E", 16) + COLOR_OFFSET

    if "ISO2MESH_RANDSEED" in globals():
        randseed = globals()["ISO2MESH_RANDSEED"]
    np.random.seed(randseed)

    sc = np.random.rand(10, 3)

    ax = plt.gca()

    h = {"fig": [], "ax": [], "obj": []}
    h["fig"].append(plt.gcf())
    h["ax"].append(ax)

    if not "color" in kwargs and not "cmap" in kwargs:
        kwargs["cmap"] = plt.get_cmap("jet")

    if isinstance(face, list):  # polyhedral facets
        newsurf = {}
        colormap = []

        for fc in face:
            if (
                isinstance(fc, (list, tuple))
                and len(fc) >= 2
                and isinstance(fc[0], (list, tuple))
            ):
                group_id = fc[1][0]
                if group_id + 1 > sc.shape[0]:
                    sc = np.vstack([sc, np.random.rand(group_id + 1 - sc.shape[0], 3)])
                newsurf.setdefault(group_id, []).append(np.asarray(fc[0]) - 1)
            else:
                newsurf.setdefault(1, []).append(np.asarray(fc) - 1)

        polydata = [
            node[np.array(subf).flatten(), :3]
            for subface in newsurf.values()
            for subf in subface
        ]
        if node.shape[1] > 3:
            node_values = node[:, 3]
            face_values = np.array(
                [
                    np.mean(node_values[np.array(subf).flatten()])
                    for subface in newsurf.values()
                    for subf in subface
                ]
            )
            norm = Normalize(vmin=face_values.min(), vmax=face_values.max())
            colormap = kwargs["cmap"](norm(face_values))
        else:
            colormap = [
                sc[i - 1, :] for i, subface in newsurf.items() for subf in subface
            ]

    elif face.shape[1] == 2:
        h = plotedges(node, face, *args, **kwargs)
        return h
    elif face.shape[1] == 4:
        tag = face[:, 3]
        types = np.unique(tag)

        if len(types) > sc.shape[0]:
            sc = np.vstack([sc, np.random.rand(len(types) - sc.shape[0], 3)])

        # plt.hold(True)
        polydata = []
        colormap = []
        for i in range(len(types)):
            pdata, _ = plotasurf(
                node,
                face[tag == types[i], 0:3],
                *args,
                **kwargs,
            )
            polydata.extend(pdata)
            colormap.extend([sc[i].tolist()] * len(pdata))
    else:
        polydata, colormap = plotasurf(node, face, *args, **kwargs)

    if "colormap" in locals() and len(colormap) > 0 and not "facecolors" in kwargs:
        kwargs["facecolors"] = colormap

    if "cmap" in kwargs and not "facecolors" in kwargs and face:
        node_values = node[:, 3] if node.shape[1] > 3 else node[:, 2]
        face_values = np.array([np.mean(node_values[f]) for f in face[:, :3] - 1])
        norm = Normalize(vmin=face_values.min(), vmax=face_values.max())
        kwargs["facecolors"] = kwargs["cmap"](norm(face_values))

    if not "linewidth" in kwargs:
        kwargs["linewidth"] = 0.3

    patch = Poly3DCollection(polydata, edgecolors="k", **kwargs)

    ax.add_collection3d(patch)
    _autoscale_3d(ax, node)
    h["obj"].append(patch)

    np.random.set_state(rngstate)
    # plt.axis("equal")

    return h


# _________________________________________________________________________________________________________


def plotasurf(node, face, *args, **kwargs):
    from matplotlib.colors import Normalize

    poly3d = [[node[i, :3] for i in p] for p in face[:, :3] - 1]
    node_values = node[:, 3] if node.shape[1] > 3 else node[:, 2]
    face_values = np.array([np.mean(node_values[f]) for f in face[:, :3] - 1])
    norm = Normalize(vmin=face_values.min(), vmax=face_values.max())
    colmap = []
    if "cmap" in kwargs:
        colmap = kwargs["cmap"](norm(face_values))
    return poly3d, colmap


# _________________________________________________________________________________________________________


def plottetra(node, elem, *args, **kwargs):
    """
    hm = plottetra(node, elem, *args, **kwargs)

    Plot 3D surface meshes.

    Parameters:
        node: (N, 3) or (N, 4) array of node coordinates (last column optional for color).
        elem: (M, 4) or (M, 5) array of tetrahedra (last column optional for tags).
        args, kwargs: Additional plotting options passed to plotsurf.

    Returns:
        hm: list of plot handles.
    """

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.colors import Normalize

    # Save current RNG state
    rngstate = np.random.get_state()

    # Set deterministic seed for consistent coloring
    randseed = int("623F9A9E", 16) + COLOR_OFFSET

    if "ISO2MESH_RANDSEED" in globals():
        randseed = globals()["ISO2MESH_RANDSEED"]

    np.random.seed(randseed)

    ax = plt.gca()

    h = {"fig": [], "ax": [], "obj": []}
    h["fig"].append(plt.gcf())
    h["ax"].append(ax)

    if not "color" in kwargs and not "cmap" in kwargs:
        kwargs["cmap"] = plt.get_cmap("jet")

    polydata = []
    colormap = []

    if isinstance(elem, list):
        elem = np.array(elem)

    if elem.shape[1] > 4:
        tag = elem[:, 4]  # 1-based -> column 5 in MATLAB
        types = np.unique(tag)
        for t in types:
            idx = np.where(tag == t)[0]
            face = volface(elem[idx, :4])[0]
            pdata, _ = plotasurf(node, face, *args, **kwargs)
            polydata.extend(pdata)
            colormap.extend(np.random.rand(1, 3).tolist() * len(pdata))
    else:
        face = volface(elem[:, :4])[0]
        polydata, colormap = plotasurf(node, face, *args, **kwargs)

    if "colormap" in locals() and len(colormap) > 0 and not "facecolors" in kwargs:
        kwargs["facecolors"] = colormap

    if "cmap" in kwargs and not "facecolors" in kwargs:
        node_values = node[:, 3] if node.shape[1] > 3 else node[:, 2]
        face_values = np.array([np.mean(node_values[f]) for f in elem[:, :4] - 1])
        norm = Normalize(vmin=face_values.min(), vmax=face_values.max())
        kwargs["facecolors"] = kwargs["cmap"](norm(face_values))

    if not "linewidth" in kwargs:
        kwargs["linewidth"] = 0.3

    patch = Poly3DCollection(polydata, edgecolors="k", **kwargs)
    ax.add_collection3d(patch)
    _autoscale_3d(ax, node)

    h["obj"].append(patch)

    # Restore RNG state
    np.random.set_state(rngstate)

    # Return handle if needed
    return h


# _________________________________________________________________________________________________________


def plotedges(node, edges, *args, **kwargs):
    """
    Plot a 3D polyline or closed loop (1D manifold).

    Parameters
    ----------
    node : ndarray (N, 3 or 4)
        Node coordinates. If a 4th column is present, it can represent color.
    edges : ndarray or list
        Either a 2-column edge list or a 1D list/array of node indices separated by NaN.
    *args : list
        Additional plotting options (passed to matplotlib).
    iso2mesh_randseed : int, optional
        Random seed for color generation (used for loops).

    Returns
    -------
    hh : list
        Handles to plotted elements.
    """
    hh = {"fig": [], "ax": [], "obj": []}
    edges = np.asarray(edges, order="F")  # Flatten in F order if needed

    if edges.size == 0:
        return hh

    rng_state = np.random.get_state()

    ax = plt.gca()

    hh["fig"].append(plt.gcf())
    hh["ax"].append(ax)

    if edges.ndim == 1 or edges.shape[1] == 1:
        # Loop: NaN-separated index list
        randseed = int("623F9A9E", 16) + COLOR_OFFSET
        if "iso2mesh_randseed" in kwargs:
            randseed = kwargs["iso2mesh_randseed"]
        np.random.seed(randseed)

        loops = edges.flatten(order="F")
        if not np.isnan(loops[-1]):
            loops = np.append(loops, np.nan)

        seg = np.where(np.isnan(loops))[0]
        seghead = 0

        for i in seg:
            segment = loops[seghead:i]
            segment = segment.astype(int) - 1  # 1-based to 0-based
            if segment.size > 1:
                (h,) = plt.plot(
                    node[segment, 0],
                    node[segment, 1],
                    node[segment, 2] if node.shape[1] >= 3 else None,
                    color=np.random.rand(
                        3,
                    ),
                    *args,
                    **kwargs,
                )
                hh["obj"].append(h)
            seghead = i + 1
    else:
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        # Edge list: each row connects 2 nodes
        edges = edges.astype(int) - 1  # 1-based to 0-based

        if node.shape[1] >= 3:
            segments = [[node[start], node[end]] for start, end in edges]
            h = Line3DCollection(segments, **kwargs)
            ax.add_collection3d(h)
            _autoscale_3d(ax, node)

        else:
            x = node[:, 0].flatten()
            y = node[:, 1].flatten()
            h = plt.plot(x[edges.T], y[edges.T], *args, **kwargs)

        hh["obj"].append(h)

    np.random.set_state(rng_state)
    return hh


# _________________________________________________________________________________________________________


def plotmesh(node, *args, **kwargs):
    """
    handles = plotmesh(node, face, elem, selector, ...)
    Plot surface and volumetric meshes in 3D.
    Converts 1-based MATLAB indices in `face` and `elem` to 0-based.
    Supports optional selector strings and stylistic options.
    """

    selector = None
    opt = []
    face = None
    elem = None
    node = np.array(node)

    # Parse inputs: detect selector strings, face/elem arrays, opts
    for i, a in enumerate(args):
        if isinstance(a, str):
            if any(c in a for c in "<>=&|") and any(c in a for c in "xyzXYZ"):
                selector = a
                opt = list(args[i + 1 :])
                break
            else:
                opt = list(args[i:])
                break
        else:
            if i == 0:
                if isinstance(a, list) or (
                    isinstance(a, np.ndarray) and a.ndim == 2 and a.shape[1] < 4
                ):
                    face = a
                elif isinstance(a, np.ndarray) and a.ndim == 2 and a.shape[1] in (4, 5):
                    uniq = np.unique(a[:, 3])
                    counts = np.bincount(a[:, 3].astype(int))
                    if len(uniq) == 1 or np.any(counts > 50):
                        face = a
                    else:
                        elem = a
                else:
                    elem = a
            elif i == 1:
                face = args[0]
                elem = a

    extraarg = {}
    if "hold" in kwargs:
        extraarg["hold"] = kwargs["hold"]

    ax = _createaxis(True, *args, **kwargs)

    handles = {"fig": [], "ax": [], "obj": []}
    handles["fig"].append(plt.gcf())
    handles["ax"].append(ax)

    for extraopt in ["hold", "parent", "subplot"]:
        if extraopt in kwargs:
            del kwargs[extraopt]

    # Plot points if no face/elem
    if face is None and elem is None:
        x, y, z = node[:, 0], node[:, 1], node[:, 2]
        idx = (
            np.where(eval(selector, {"x": x, "y": y, "z": z}))[0]
            if selector
            else slice(None)
        )
        if getattr(idx, "size", None) == 0:
            print("Warning: nothing to plot")
            return None
        (h,) = ax.plot(x[idx], y[idx], z[idx], *opt, **kwargs)
        handles["obj"].append(h)
        _autoscale_3d(ax, node)

    # Plot surface mesh
    if face is not None:
        if isinstance(face, list):
            handles = plotsurf(node, face, opt, *args, **kwargs)
        else:
            c0 = meshcentroid(node[:, :3], face[:, :3])
            x, y, z = c0[:, 0], c0[:, 1], c0[:, 2]
            idx = (
                np.where(eval(selector, {"x": x, "y": y, "z": z}))[0]
                if selector
                else slice(None)
            )
            if getattr(idx, "size", None) == 0:
                print("Warning: nothing to plot")
                return None
            handles = plotsurf(node, face[idx, :], opt, *args, **kwargs)

    # Plot tetrahedral mesh
    if elem is not None:
        c0 = meshcentroid(node[:, :3], elem[:, :4])
        x, y, z = c0[:, 0], c0[:, 1], c0[:, 2]
        idx = (
            np.where(eval(selector, {"x": x, "y": y, "z": z}))[0]
            if selector
            else slice(None)
        )
        if getattr(idx, "size", None) == 0:
            print("Warning: nothing to plot")
            return None
        handles = plottetra(node, elem[idx, :], opt, *args, **kwargs)

    if not "hold" in extraarg or not extraarg["hold"] or extraarg["hold"] == "off":
        plt.draw()
        plt.show(block=False)

    return handles


def _autoscale_3d(ax, points):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    boxas = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()])
    if boxas[0] > 0:
        ax.set_xlim([x.min(), x.max()])
    if boxas[1] > 0:
        ax.set_ylim([y.min(), y.max()])
    if boxas[2] > 0:
        ax.set_zlim([z.min(), z.max()])
    if np.all(boxas > 0):
        ax.set_box_aspect(boxas)


def _createaxis(*args, **kwargs):
    subplotid = kwargs.get("subplot", 111)
    docreate = True
    fig = None

    if "parent" in kwargs:
        hh = kwargs["parent"]
        if isinstance(hh, dict):
            fig = hh["fig"][0]
            ax = hh["ax"][-1]
        elif isinstance(hh, list):
            ax = hh[-1]
        if "subplot" in kwargs and fig:
            ax = fig.add_subplot(subplotid, projection="3d")
    elif not docreate and len(plt.get_fignums()) > 0 and len(plt.gcf().axes) > 0:
        if not fig:
            fig = plt.gcf()
        ax = fig.axes[-1]
    else:
        if docreate and not fig:
            fig = plt.figure()
        ax = fig.add_subplot(subplotid, projection="3d")

    if ax.name != "3d":
        fig = plt.figure()  # Create a new figure
        ax = fig.add_subplot(subplotid, projection="3d")

    return ax
