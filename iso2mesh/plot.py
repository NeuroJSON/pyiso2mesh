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
    "plotvolume",
    "plotbackend",
]

##====================================================================================
## dependent libraries
##====================================================================================

import os
import sys
import dis
import functools
import numpy as np

# matplotlib is the default backend; import it eagerly but tolerate its absence so
# that the module can still be used with an alternative backend (plotly/pyvista).
try:
    import matplotlib.pyplot as plt

    _MPL_AVAILABLE = True
except ImportError:
    plt = None
    _MPL_AVAILABLE = False

# Attempt to register 3D projection - handle version conflicts gracefully
_3D_AVAILABLE = False
try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    _3D_AVAILABLE = True
except ImportError:
    # Handle matplotlib version conflicts between system and pip installations
    try:
        # Alternative registration method for newer matplotlib
        import matplotlib.projections as proj

        if "3d" not in proj.get_projection_names():
            import warnings

            warnings.warn(
                "3D plotting may not be available due to matplotlib version conflict. "
                "Consider: pip uninstall matplotlib && pip install matplotlib"
            )
    except Exception:
        pass

from iso2mesh.trait import volface, meshcentroid

COLOR_OFFSET = 3

##====================================================================================
## handle dict that does not echo in Jupyter / IPython
##====================================================================================


class PlotHandle(dict):
    """
    A dict of plot handles (keys ``'fig'``, ``'ax'``, ``'obj'``) returned by the
    plotting functions. It behaves exactly like a normal dict -- indexing,
    ``print()`` and ``repr()`` all work -- but declares an empty rich display so
    that a bare ``plotmesh(...)`` as the last line of a Jupyter/Colab/IPython
    cell does not echo the (often huge) handle dict. Capture it (``h =
    plotmesh(...)``) to use the handles as before.

    This mechanism is environment-independent (unlike bytecode/``nargout``
    inspection) and works across notebook frontends and Python versions.
    """

    def _ipython_display_(self):
        # Defining this method makes IPython treat display as "handled"; by
        # doing nothing we suppress the automatic Out[...] echo entirely, while
        # repr()/print() and dict access keep working normally.
        pass


def _return_value_used():
    """
    Best-effort emulation of MATLAB's ``nargout`` for *non-notebook* callers:
    inspect the caller's bytecode and report whether the return value is used.

    Returns False when the result is discarded -- a bare ``plotmesh(...)``
    statement (``POP_TOP``) or auto-displayed as the last expression in a plain
    Python REPL (``PRINT_EXPR`` on <=3.11, ``CALL_INTRINSIC_1``/print on 3.12+).
    Returns True otherwise, or if the call site cannot be inspected. In notebooks
    the suppression is handled robustly by :class:`PlotHandle` instead, so a
    misfire here is harmless.
    """
    try:
        frame = sys._getframe(2)  # 0: this fn, 1: decorated wrapper, 2: caller
        lasti = frame.f_lasti
        nxt = next(
            (ins for ins in dis.get_instructions(frame.f_code) if ins.offset > lasti),
            None,
        )
        if nxt is None:
            return True
        if nxt.opname in ("POP_TOP", "PRINT_EXPR"):
            return False
        # Python 3.12 replaced PRINT_EXPR with CALL_INTRINSIC_1(INTRINSIC_PRINT)
        if nxt.opname == "CALL_INTRINSIC_1" and "PRINT" in str(nxt.argval):
            return False
        return True
    except Exception:
        return True


def _nargout_aware(func):
    """
    Decorator for the plotting functions. Wraps the returned handle dict in a
    :class:`PlotHandle` (so it never echoes in notebooks) and, for plain
    scripts/REPLs, returns ``None`` when the caller discards the value.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, dict) and not isinstance(result, PlotHandle):
            result = PlotHandle(result)
        return result if _return_value_used() else None

    return wrapper


##====================================================================================
## plotting backend selection (matplotlib | plotly | pyvista)
##====================================================================================

# Canonical backend names and their accepted aliases.
_BACKEND_ALIASES = {
    "matplotlib": "matplotlib",
    "mpl": "matplotlib",
    "plotly": "plotly",
    "pyvista": "pyvista",
    "pv": "pyvista",
    "vtk": "pyvista",
}

# The selected backend is stored in the ISO2MESH_PLOT_BACKEND environment
# variable, following the same convention as other iso2mesh runtime options
# (e.g. ISO2MESH_TEMP, ISO2MESH_SESSION, ISO2MESH_SURFBOOLEAN). It is read at
# the point of use so it can be set any time (via os.environ, the shell, or
# plotbackend()) and defaults to 'matplotlib' when unset.


def _canonical_backend(name):
    """Validate ``name`` and map any accepted alias to its canonical backend."""
    key = str(name).lower()
    if key not in _BACKEND_ALIASES:
        raise ValueError(
            f"unknown plotting backend '{name}'; "
            f"choose from {sorted(set(_BACKEND_ALIASES.values()))}"
        )
    return _BACKEND_ALIASES[key]


def plotbackend(name=None):
    """
    Get or set the global plotting backend used by ``plotmesh`` and friends.

    Called with no argument, returns the currently selected backend, read from
    the ``ISO2MESH_PLOT_BACKEND`` environment variable (default 'matplotlib'; an
    unrecognized value falls back to 'matplotlib').

    Called with a name, validates it and sets the global backend by writing the
    ``ISO2MESH_PLOT_BACKEND`` environment variable (setting that variable
    directly, e.g. in the shell or via ``os.environ``, has the same effect), then
    returns the resolved canonical name.

    Parameters:
        name: one of 'matplotlib' (default, alias 'mpl'), 'plotly', or
              'pyvista' (aliases 'pv', 'vtk'); or None to query the current one.

    Returns:
        The resolved canonical backend name.

    A per-call override is also possible by passing ``backend=`` to ``plotmesh``.
    """
    if name is None:
        return _BACKEND_ALIASES.get(
            os.environ.get("ISO2MESH_PLOT_BACKEND", "matplotlib").lower(), "matplotlib"
        )
    os.environ["ISO2MESH_PLOT_BACKEND"] = _canonical_backend(name)
    return os.environ["ISO2MESH_PLOT_BACKEND"]


def _resolve_backend(kwargs):
    """
    Pop and resolve the ``backend`` keyword (if any), otherwise fall back to the
    global ``ISO2MESH_PLOT_BACKEND`` setting. Returns the canonical backend name.
    """
    name = kwargs.pop("backend", None)
    return plotbackend() if name is None else _canonical_backend(name)


def _parse_plotmesh_args(node, args):
    """
    Parse the positional arguments of ``plotmesh`` into (node, face, elem,
    selector, opt). This logic is shared by every backend so that input
    handling stays identical regardless of the renderer.
    """
    selector = None
    opt = []
    face = None
    elem = None
    node = np.asarray(node)

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

    return node, face, elem, selector, opt


def _selector_idx(coords, selector):
    """
    Return the row indices of ``coords`` (centroids or points) that satisfy a
    MATLAB-style selector string such as 'x<10 & z>0'. Returns ``slice(None)``
    when no selector is given.
    """
    if not selector:
        return slice(None)
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    return np.where(eval(selector, {"x": x, "y": y, "z": z, "np": np}))[0]


def _opt_color(opt):
    """Extract a single-letter matplotlib-style color (e.g. from 'ro') if present."""
    table = {
        "r": "red",
        "g": "green",
        "b": "blue",
        "k": "black",
        "w": "white",
        "y": "yellow",
        "m": "magenta",
        "c": "cyan",
    }
    for a in opt or []:
        if isinstance(a, str):
            for ch in a:
                if ch in table:
                    return table[ch]
    return None


# A small qualitative palette reused by the plotly/pyvista backends to color
# tagged sub-surfaces and sub-domains.
_TAG_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

# _________________________________________________________________________________________________________


@_nargout_aware
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

    if (
        "cmap" in kwargs
        and not "facecolors" in kwargs
        and not isinstance(face, list)
        and face is not None
    ):
        f = np.asarray(face)[:, :3].astype(int) - 1
        node_values = node[:, 3] if node.shape[1] > 3 else node[:, 2]
        face_values = node_values[f].mean(axis=1)
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

    # vectorized: (Nf,3,3) vertex array and per-face mean values via fancy
    # indexing (the previous Python list comprehensions were O(Nf) interpreted
    # loops and dominated rendering time on large surfaces).
    f = np.asarray(face)[:, :3].astype(int) - 1
    poly3d = node[f, :3]
    node_values = node[:, 3] if node.shape[1] > 3 else node[:, 2]
    face_values = node_values[f].mean(axis=1)
    colmap = []
    if "cmap" in kwargs:
        norm = Normalize(vmin=face_values.min(), vmax=face_values.max())
        colmap = kwargs["cmap"](norm(face_values))
    return poly3d, colmap


# _________________________________________________________________________________________________________


@_nargout_aware
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
        e = elem[:, :4].astype(int) - 1
        node_values = node[:, 3] if node.shape[1] > 3 else node[:, 2]
        face_values = node_values[e].mean(axis=1)
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


@_nargout_aware
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


@_nargout_aware
def plotmesh(node, *args, **kwargs):
    """
    handles = plotmesh(node, face, elem, selector, ...)
    Plot surface and volumetric meshes in 3D.
    Converts 1-based MATLAB indices in `face` and `elem` to 0-based.
    Supports optional selector strings and stylistic options.

    The rendering backend can be selected globally via ``plotbackend()`` or the
    ``ISO2MESH_PLOT_BACKEND`` environment variable, or per call via the
    ``backend=`` keyword. Supported backends:

        'matplotlib' (default): static/interactive Matplotlib 3D axes.
        'plotly':               interactive, web/notebook-friendly (go.Mesh3d).
        'pyvista':              VTK/GPU-accelerated, best for large/volumetric
                                meshes with native tetrahedral support.

    The non-matplotlib backends require the corresponding package to be
    installed (``pip install plotly`` or ``pip install pyvista``).
    """

    backend = _resolve_backend(kwargs)
    if backend == "plotly":
        return _plotmesh_plotly(node, *args, **kwargs)
    if backend == "pyvista":
        return _plotmesh_pyvista(node, *args, **kwargs)

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
        if "alpha" not in kwargs and selector:
            faceselect, elemid = volface(elem[idx, :4])
            if elem.shape[1] > 4:
                faceselect = np.hstack(
                    (faceselect, elem[idx[elemid - 1], 4].reshape(-1, 1))
                )
            handles = plotsurf(node, faceselect, opt, *args, **kwargs)
        else:
            handles = plottetra(node, elem[idx, :], opt, *args, **kwargs)

    if not "hold" in extraarg or not extraarg["hold"] or extraarg["hold"] == "off":
        plt.draw()
        plt.show(block=False)

    return handles


# _________________________________________________________________________________________________________
#  Shared helpers for the plotly / pyvista backends
# _________________________________________________________________________________________________________


def _facet_polys(face):
    """
    Extract the original polygons from a cell-array style ``face`` (a list of
    facets, each either a node-index array or a ``[indices, [group_id]]`` pair).
    Returns a list of 1-based node-index arrays and a matching group-id array.
    """
    polys, gids = [], []
    for fc in face:
        if (
            isinstance(fc, (list, tuple))
            and len(fc) >= 2
            and isinstance(fc[0], (list, tuple, np.ndarray))
        ):
            polys.append(np.asarray(fc[0]).ravel())
            gids.append(int(np.asarray(fc[1]).ravel()[0]))
        else:
            polys.append(np.asarray(fc).ravel())
            gids.append(1)
    return polys, np.asarray(gids, dtype=int)


def _faces_from_list(node, face):
    """
    Fan-triangulate a cell-array style ``face`` into an (M,3) triangle array
    (1-based) and a matching per-triangle tag vector. The fan triangulation is
    only used by renderers (e.g. plotly) that cannot draw n-gons directly; the
    original polygon boundaries are recovered separately for the wireframe.
    """
    polys, gids = _facet_polys(face)
    tris, tags = [], []
    for idx, gid in zip(polys, gids):
        for t in range(1, len(idx) - 1):
            tris.append([idx[0], idx[t], idx[t + 1]])
            tags.append(gid)
    return np.asarray(tris, dtype=int), np.asarray(tags, dtype=int)


def _as_face_array(node, face):
    """Return ``face`` as an (M,>=3) array, fan-triangulating list facets."""
    if isinstance(face, list):
        tris, tags = _faces_from_list(node, face)
        return np.hstack((tris, tags.reshape(-1, 1)))
    return np.asarray(face)


def _tetra_surface(elem):
    """Extract the bounding triangular surface of a tet mesh, carrying tags."""
    if elem.shape[1] > 4:
        faces, elemid = volface(elem[:, :4])
        return np.hstack((faces, elem[elemid - 1, 4].reshape(-1, 1)))
    return volface(elem[:, :4])[0]


def _select_rows(node, conn, ncol, selector):
    """Filter face/elem rows by a centroid selector; returns rows or None if empty."""
    if not selector:
        return conn
    idx = _selector_idx(meshcentroid(node[:, :3], conn[:, :ncol]), selector)
    if getattr(idx, "size", None) == 0:
        print("Warning: nothing to plot")
        return None
    return conn[idx, :]


def _plot_options(kwargs, default_cmap):
    """Pop the options common to the plotly/pyvista backends."""
    hold = str(kwargs.pop("hold", "off")).lower() in ("on", "true", "1")
    return {
        "parent": kwargs.pop("parent", None),
        "cmap": kwargs.pop("cmap", default_cmap),
        "opacity": float(kwargs.pop("alpha", kwargs.pop("opacity", 1.0))),
        "show_edges": kwargs.pop("show_edges", True),
        "edgecolor": kwargs.pop("edgecolor", kwargs.pop("edgecolors", "black")),
        "jupyter_backend": kwargs.pop("jupyter_backend", None),
        # plotly: disable wheel-zoom by default to avoid accidental zooming while
        # scrolling; re-enable with scrollzoom=True. Bound the figure height so
        # the plot fits Colab's output frame. All are overridable per call.
        "scrollzoom": kwargs.pop("scrollzoom", False),
        "config": kwargs.pop("config", None),
        "height": kwargs.pop("height", 500),
        "width": kwargs.pop("width", None),
        "show": kwargs.pop("show", not hold),
    }


def _edge_lines(node, face):
    """
    Return nan-separated x/y/z polyline arrays tracing the unique triangle edges
    of ``face`` (1-based). Used to overlay a wireframe on plotly Mesh3d, which
    has no native edge rendering.
    """
    f = np.asarray(face)[:, :3].astype(int) - 1
    e = np.sort(np.vstack((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]])), axis=1)
    e = np.unique(e, axis=0)
    seg = np.full((e.shape[0] * 3, 3), np.nan)  # 3rd row of each triple = break
    seg[0::3] = node[e[:, 0], :3]
    seg[1::3] = node[e[:, 1], :3]
    return seg[:, 0], seg[:, 1], seg[:, 2]


def _polygon_edge_lines(node, polys):
    """
    Return nan-separated x/y/z polylines tracing the *closed boundary* of each
    polygon facet (1-based). Used for the plotly wireframe so PLC polygons show
    only their outline, not the internal fan-triangulation diagonals.
    """
    xs, ys, zs = [], [], []
    for p in polys:
        idx = np.asarray(p, dtype=int).ravel() - 1
        if idx.size < 2:
            continue
        loop = node[np.append(idx, idx[0]), :3]  # repeat first vertex to close
        xs.extend(loop[:, 0].tolist() + [np.nan])
        ys.extend(loop[:, 1].tolist() + [np.nan])
        zs.extend(loop[:, 2].tolist() + [np.nan])
    return np.array(xs), np.array(ys), np.array(zs)


def _render_mesh(node, args, kwargs, ops):
    """
    Backend-agnostic control flow shared by the plotly and pyvista renderers.
    ``ops`` supplies the canvas/points/surface/tetra/finalize primitives.
    """
    node, face, elem, selector, opt = _parse_plotmesh_args(node, args)
    node = np.asarray(node, dtype=float)
    kwargs.pop("subplot", None)
    cfg = _plot_options(kwargs, ops.default_cmap)
    canvas, handles = ops.canvas(cfg["parent"])

    if face is None and elem is None:
        idx = _selector_idx(node[:, :3], selector)
        pts = node[idx, :3]
        if pts.shape[0] == 0:
            print("Warning: nothing to plot")
            return None
        vals = node[idx, 3] if node.shape[1] > 3 else None
        ops.points(canvas, handles, pts, vals, _opt_color(opt), cfg)

    if face is not None:
        # Recover original polygons (for n-gon rendering / boundary-only edges).
        # A selector filters triangles, breaking the polygon mapping, so skip it.
        polys = gids = None
        if isinstance(face, list) and not selector:
            polys, gids = _facet_polys(face)
        face = _select_rows(node, _as_face_array(node, face), 3, selector)
        if face is None:
            return None
        ops.surface(canvas, handles, node, face, cfg, polys, gids)

    if elem is not None:
        elem = _select_rows(node, np.asarray(elem), 4, selector)
        if elem is None:
            return None
        ops.tetra(canvas, handles, node, elem, cfg)

    ops.finalize(canvas, cfg)
    return handles


# _________________________________________________________________________________________________________
#  Plotly backend
# _________________________________________________________________________________________________________


class _PlotlyOps:
    default_cmap = "Jet"

    def __init__(self, go):
        self.go = go

    def canvas(self, parent):
        if isinstance(parent, dict) and parent.get("fig"):
            fig = parent["fig"][0]
        elif isinstance(parent, self.go.Figure):
            fig = parent
        else:
            fig = self.go.Figure()
        return fig, {"fig": [fig], "ax": [None], "obj": []}

    def points(self, fig, h, pts, vals, color, cfg):
        tr = self.go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="markers",
            marker=dict(
                size=3,
                color=vals if vals is not None else (color or "red"),
                colorscale=cfg["cmap"],
            ),
            name="nodes",
        )
        fig.add_trace(tr)
        h["obj"].append(tr)

    def surface(self, fig, h, node, face, cfg, polys=None, gids=None):
        node_vals = node.shape[1] > 3
        common = dict(
            x=node[:, 0],
            y=node[:, 1],
            z=node[:, 2],
            opacity=cfg["opacity"],
            flatshading=True,
        )
        if face.shape[1] >= 4 and not node_vals:  # tagged sub-surfaces
            for n, t in enumerate(np.unique(face[:, 3])):
                f = face[face[:, 3] == t, :3].astype(int) - 1
                self._add(
                    fig,
                    h,
                    self.go.Mesh3d(
                        i=f[:, 0],
                        j=f[:, 1],
                        k=f[:, 2],
                        color=_TAG_PALETTE[n % len(_TAG_PALETTE)],
                        name=f"surf {int(t)}",
                        **common,
                    ),
                )
        else:
            f = face[:, :3].astype(int) - 1
            self._add(
                fig,
                h,
                self.go.Mesh3d(
                    i=f[:, 0],
                    j=f[:, 1],
                    k=f[:, 2],
                    intensity=node[:, 3] if node_vals else node[:, 2],
                    intensitymode="vertex",
                    colorscale=cfg["cmap"],
                    showscale=node_vals,
                    name="surf",
                    **common,
                ),
            )

        # Mesh3d has no native edges; overlay a wireframe as a line trace. For
        # PLC polygons draw only the facet boundaries (not the triangulation).
        if cfg["show_edges"]:
            if polys is not None:
                xe, ye, ze = _polygon_edge_lines(node, polys)
            else:
                xe, ye, ze = _edge_lines(node, face)
            self._add(
                fig,
                h,
                self.go.Scatter3d(
                    x=xe,
                    y=ye,
                    z=ze,
                    mode="lines",
                    line=dict(color=cfg["edgecolor"], width=1),
                    hoverinfo="skip",
                    showlegend=False,
                    name="edges",
                ),
            )

    def tetra(self, fig, h, node, elem, cfg):
        self.surface(fig, h, node, _tetra_surface(elem), cfg)

    def finalize(self, fig, cfg):
        layout = dict(
            scene=dict(
                xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="data"
            ),
            margin=dict(l=0, r=0, t=0, b=0),
        )
        if cfg.get("height"):
            layout["height"] = cfg["height"]
        if cfg.get("width"):
            layout["width"] = cfg["width"]
        fig.update_layout(**layout)
        if cfg["show"]:
            # scrollZoom off by default to avoid accidental zooming; pass
            # scrollzoom=True to enable wheel zoom. A user-supplied config still
            # takes precedence.
            config = dict(cfg.get("config") or {})
            config.setdefault("scrollZoom", bool(cfg["scrollzoom"]))
            fig.show(config=config)

    @staticmethod
    def _add(fig, h, trace):
        fig.add_trace(trace)
        h["obj"].append(trace)


def _plotmesh_plotly(node, *args, **kwargs):
    """Plotly implementation of plotmesh (see plotmesh for the public API)."""
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise ImportError(
            "the 'plotly' backend requires plotly; install with 'pip install plotly'"
        ) from e
    return _render_mesh(node, args, kwargs, _PlotlyOps(go))


# _________________________________________________________________________________________________________
#  PyVista backend
# _________________________________________________________________________________________________________


def _mesh_scalar(node, conn, tagcol):
    """Pick the coloring scalar for a mesh: node values, else element/face tags."""
    if node.shape[1] > 3:
        return "nodeval", node[:, 3]
    if conn.shape[1] > tagcol:
        return "tag", conn[:, tagcol]
    return None


def _vtk_cells(conn, n):
    """Build a VTK connectivity array [n, i0..i_{n-1}, ...] (0-based)."""
    c = conn[:, :n].astype(np.int64) - 1
    return np.hstack((np.full((c.shape[0], 1), n, dtype=np.int64), c)).ravel()


def _vtk_polygon_cells(polys):
    """
    Build a VTK connectivity array for arbitrary polygons (0-based): for each
    facet, [nverts, i0, i1, ...]. VTK renders n-gons natively, so PLC facets
    stay as single polygons (no triangulation artifact under show_edges).
    """
    cells = []
    for p in polys:
        idx = (np.asarray(p, dtype=np.int64).ravel() - 1).tolist()
        cells.append(len(idx))
        cells.extend(idx)
    return np.asarray(cells, dtype=np.int64)


class _PyVistaOps:
    default_cmap = "jet"

    def __init__(self, pv):
        self.pv = pv

    def canvas(self, parent):
        if isinstance(parent, dict) and parent.get("fig"):
            pl = parent["fig"][0]
        elif isinstance(parent, self.pv.Plotter):
            pl = parent
        else:
            pl = self.pv.Plotter()
        return pl, {"fig": [pl], "ax": [pl], "obj": []}

    def _add(self, pl, h, mesh, cfg, scalar=None, color="lightgray", **extra):
        opts = dict(
            opacity=cfg["opacity"],
            show_edges=cfg["show_edges"],
            edge_color=cfg["edgecolor"],
        )
        opts.update(extra)
        if scalar is not None:
            mesh[scalar[0]] = scalar[1]
            actor = pl.add_mesh(mesh, cmap=cfg["cmap"], **opts)
        else:
            actor = pl.add_mesh(mesh, color=color, **opts)
        h["obj"].append(actor)

    def points(self, pl, h, pts, vals, color, cfg):
        cloud = self.pv.PolyData(np.ascontiguousarray(pts, dtype=float))
        scalar = ("nodeval", vals) if vals is not None else None
        self._add(
            pl,
            h,
            cloud,
            cfg,
            scalar=scalar,
            color=color or "red",
            show_edges=False,
            render_points_as_spheres=True,
            point_size=8,
        )

    def surface(self, pl, h, node, face, cfg, polys=None, gids=None):
        pts = np.ascontiguousarray(node[:, :3], dtype=float)
        if polys is not None:
            # native polygon cells -> show_edges traces facet outlines only
            surf = self.pv.PolyData(pts, _vtk_polygon_cells(polys))
            if node.shape[1] > 3:
                scalar = ("nodeval", node[:, 3])
            elif gids is not None and np.unique(gids).size > 1:
                scalar = ("tag", gids.astype(float))
            else:
                scalar = None
        else:
            surf = self.pv.PolyData(pts, _vtk_cells(face, 3))
            scalar = _mesh_scalar(node, face, 3)
        self._add(pl, h, surf, cfg, scalar=scalar)

    def tetra(self, pl, h, node, elem, cfg):
        pts = np.ascontiguousarray(node[:, :3], dtype=float)
        celltypes = np.full(elem.shape[0], 10, dtype=np.uint8)  # VTK_TETRA == 10
        grid = self.pv.UnstructuredGrid(_vtk_cells(elem, 4), celltypes, pts)
        self._add(pl, h, grid, cfg, scalar=_mesh_scalar(node, elem, 4))

    def finalize(self, pl, cfg):
        if not cfg["show"]:
            return
        # In a Jupyter notebook pyvista renders a *static* image unless an
        # interactive backend is active; forward jupyter_backend so users can
        # request live rotation/zoom. Outside notebooks it opens an interactive
        # native window as usual. Google Colab cannot use the websocket-based
        # 'trame' server backend, so default to the self-contained client-side
        # 'html' backend there, which is rotatable without a server.
        jb = cfg.get("jupyter_backend")
        if jb is None and "google.colab" in sys.modules:
            jb = "html"
        try:
            pl.show(jupyter_backend=jb) if jb else pl.show()
        except Exception as e:
            import warnings

            warnings.warn(
                f"pyvista interactive backend '{jb}' failed ({e}); falling back "
                "to the default. For rotatable plots in Google Colab, run once:\n"
                "  !apt-get -qq install xvfb libgl1-mesa-glx\n"
                "  import pyvista as pv; pv.start_xvfb()\n"
                "  pv.set_jupyter_backend('html')\n"
                "or use backend='plotly', which is interactive out of the box."
            )
            pl.show()


def _plotmesh_pyvista(node, *args, **kwargs):
    """PyVista implementation of plotmesh (see plotmesh for the public API)."""
    try:
        import pyvista as pv
    except ImportError as e:
        raise ImportError(
            "the 'pyvista' backend requires pyvista; install with 'pip install pyvista'"
        ) from e
    return _render_mesh(node, args, kwargs, _PyVistaOps(pv))


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
    """
    Create or retrieve a 3D matplotlib axis.

    Parameters:
        *args: Positional arguments (unused but accepted for compatibility)
        **kwargs: Keyword arguments including:
            - subplot: subplot specification (default: 111)
            - parent: existing figure/axis handle dict or list

    Returns:
        ax: matplotlib 3D axis object
    """
    # Try to ensure 3D projection is available
    global _3D_AVAILABLE
    if not _3D_AVAILABLE:
        try:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            _3D_AVAILABLE = True
        except ImportError:
            pass

    subplotid = kwargs.get("subplot", 111)
    fig = None
    ax = None

    if "parent" in kwargs:
        hh = kwargs["parent"]
        if isinstance(hh, dict):
            fig = hh.get("fig", [None])[0]
            ax = hh.get("ax", [None])[-1]
        elif isinstance(hh, list) and len(hh) > 0:
            ax = hh[-1]

        # If we have a parent but need a new subplot
        if "subplot" in kwargs and fig is not None:
            ax = fig.add_subplot(subplotid, projection="3d")
            return ax

        # If we have a valid 3D axis, return it
        if ax is not None and hasattr(ax, "name") and ax.name == "3d":
            return ax

    # Check for existing figures and axes
    if len(plt.get_fignums()) > 0:
        fig = plt.gcf()
        if len(fig.axes) > 0:
            ax = fig.axes[-1]
            # Check if it's already a 3D axis
            if hasattr(ax, "name") and ax.name == "3d":
                return ax

    # Create new figure and 3D axis
    if fig is None:
        fig = plt.figure()

    ax = fig.add_subplot(subplotid, projection="3d")
    return ax


class plotvolume:
    """
    Interactive 3D volume slice viewer - Python port of MATLAB slice3i and mcxplotvol

    Display slices from a volume in 3-D with interactive controls.

    Vol is either a scalar or RGB volume, e.g. N x M x K or N x M x K x 3.
    This Python implementation uses matplotlib for 3D rendering with optimized
    single PatchCollection for better performance and no culling issues.

    Features:
    - Interactive slice dragging with accurate ray-casting
    - 3D rotation by clicking/holding the middle mouse button
    - Adjust colormap levels by dragging with the right mouse button
    - Support for 4-D data (up/down keys to change frames)
    - Subsampling for performance optimization
    - MATLAB-compatible interface

    Controls:
    - Left mouse: Click and drag to move slice positions
    - Middle mouse: Click and drag to rotate 3D view
    - Right mouse: Drag to adjust colormap levels
    - Sliders: Precise slice positioning
    - Up/Down keys: Navigate frames (4D volumes)

    Quality Settings:
    - subsample=1: Highest quality (slow)
    - subsample=2: High quality (recommended)
    - subsample=4: Fast rendering
    - subsample=8: Fastest rendering

    Example:

    import numpy as np
    vol = create_demo_volume()
    viewer = plotvolume(vol, subsample=2)
    viewer.setslice(0, 32)  # X-slice at position 32
    viewer.setslice(1, 32)  # Y-slice at position 32
    viewer.setslice(2, 14)  # Z-slice at position 14
    viewer.show()

    Original MATLAB implementation by:
    Author: Anders Brun, anders@cb.uu.se (2009)
    Modified by: Qianqian Fang, q.fang at neu.edu

    Python port features:
    - Ray-casting for accurate slice selection
    - Single PatchCollection for efficient rendering
    - Subsampling for performance optimization
    - Modern matplotlib integration
    """

    def __init__(
        self, vol, I2X=None, figsize=(12, 8), subsample=1, cmap="jet", **kwargs
    ):
        """
        Initialize the interactive volume slicer

        Parameters:
        vol: 3D or 4D numpy array (N x M x K or N x M x K x 3/frames)
        I2X: 4x4 transformation matrix from volume coords to world coords
        figsize: figure size tuple
        subsample: subsampling factor for performance (1=highest quality, 4=faster)
        """
        self.vol = vol
        self.subsample = subsample  # Store subsampling factor

        # Disable I2X transformation - use identity matrix
        self.I2X = np.eye(4, dtype=float)

        # Handle different volume types
        self.current_vol = vol
        self.frame = 0
        # accept colormap= as a back-compat alias for cmap= (and keep it out of
        # the Poly3DCollection params below)
        cmap = kwargs.pop("colormap", cmap)
        self.params = kwargs

        # Like mcxplotvol.m, any 4-D array is treated as a stack of frames
        # (the 4th dimension), navigable with the up/down arrow keys.
        if vol.ndim == 4:
            self.current_vol = vol[:, :, :, 0]
            self.is_multiframe = True
        else:
            self.is_multiframe = False

        # Initialize slice positions
        self.slice_positions = {
            0: self.current_vol.shape[0] // 2,  # i dimension (x-slice)
            1: self.current_vol.shape[1] // 2,  # j dimension (y-slice)
            2: self.current_vol.shape[2] // 2,  # k dimension (z-slice)
        }

        # GUI state for interaction
        self.gui_state = {
            "dragging": False,
            "rotating": False,
            "adjusting_colormap": False,
            "start_ray": None,
            "start_idx": None,
            "start_mouse_pos": None,
            "active_slicedim": None,
            "colormap_levels": 256,
            "rotation_enabled": False,
            "last_mouse_pos": None,
        }

        # Single patch collection for all slices
        self.patch_collection = None
        # may be a colormap name (str) or a matplotlib Colormap, like plotmesh
        self.colormap_name = cmap

        self.setup_figure(figsize)
        self.update_patches()
        self._update_title()

        # Display the viewer immediately (non-blocking) so a bare
        # plotvolume(vol) call pops up the interactive window like mcxplotvol,
        # while keeping the REPL/notebook responsive. Call .show() for a
        # blocking display (e.g. at the end of a script).
        try:
            plt.show(block=False)
        except Exception:
            pass

    def setup_figure(self, figsize):
        """Setup the main figure and 3D axis"""
        # Close any existing figures to prevent multiple windows
        plt.close("all")

        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection="3d")

        # Set initial view similar to MATLAB example
        self.ax.view_init(elev=45, azim=30)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        # Disable default mouse rotation
        self.ax.mouse_init(rotate_btn=None, zoom_btn=None)

        # Set equal aspect ratio
        self.set_equal_aspect()

        # Connect mouse and keyboard events
        self.fig.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_mouse_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

    def _update_title(self):
        """Show the current frame (of N) in the title for 4-D volumes."""
        if self.is_multiframe:
            self.ax.set_title(
                "use up/down keys to change frame "
                f"(frame {self.frame + 1} of {self.vol.shape[3]})"
            )

    def set_equal_aspect(self):
        """Set 1:1:1 aspect ratio with correct axis bounds (no transformation)"""
        # Get volume shape
        vol_shape = np.array(self.current_vol.shape)

        # Use volume indices directly (no transformation)
        padding = 2  # Small padding in voxel units

        x_min, x_max = -padding, vol_shape[0] + padding  # 0 to 64
        y_min, y_max = -padding, vol_shape[1] + padding  # 0 to 64
        z_min, z_max = -padding, vol_shape[2] + padding  # 0 to 27

        # Set the axis limits
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_zlim(z_min, z_max)

        # Set 1:1:1 aspect ratio - equal scaling for all axes
        self.ax.set_box_aspect([1, 1, 1])

        # Force matplotlib to respect the aspect ratio
        try:
            self.ax.set_aspect("equal")
        except:
            pass  # Some matplotlib versions don't support this for 3D

    def create_slice_polygons(self, slicedim, sliceidx):
        """
        Build quad vertices and per-quad colors for one slice (with subsampling).

        Fully vectorized: returns an (N, 4, 3) array of quad corners and an (N,)
        array of block-mean colors. The previous per-quad Python loops made
        interactive dragging slow on large volumes.
        """
        vol = self.current_vol
        sliceidx = int(np.clip(sliceidx, 0, vol.shape[slicedim] - 1))
        s = self.subsample

        if slicedim == 2:  # z-slice: in-plane axes (i=0, j=1)
            slice_data = vol[:, :, sliceidx]
            a_axis, b_axis = 0, 1
        elif slicedim == 1:  # y-slice: in-plane axes (i=0, k=2)
            slice_data = vol[:, sliceidx, :]
            a_axis, b_axis = 0, 2
        else:  # x-slice: in-plane axes (j=1, k=2)
            slice_data = vol[sliceidx, :, :]
            a_axis, b_axis = 1, 2

        na, nb = slice_data.shape
        ia = np.arange(0, na - s, s)
        jb = np.arange(0, nb - s, s)
        if ia.size == 0 or jb.size == 0:
            return np.empty((0, 4, 3)), np.empty((0,))

        nbi, nbj = ia.size, jb.size
        # block-mean color of each (s x s) tile (tiles are full-size by construction)
        colors = (
            slice_data[: nbi * s, : nbj * s]
            .reshape(nbi, s, nbj, s)
            .mean(axis=(1, 3))
            .ravel()
        )

        # quad corners in the two in-plane coordinates
        a0, b0 = np.meshgrid(ia, jb, indexing="ij")
        a0, b0 = a0.ravel(), b0.ravel()
        da = np.array([0, s, s, 0])
        db = np.array([0, 0, s, s])
        verts = np.empty((a0.size, 4, 3))
        verts[:, :, slicedim] = sliceidx
        verts[:, :, a_axis] = a0[:, None] + da[None, :]
        verts[:, :, b_axis] = b0[:, None] + db[None, :]
        return verts, colors

    def update_patches(self):
        """Update the single patch collection with all slice polygons"""
        # Remove existing patch collection
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from matplotlib.colors import Normalize

        if self.patch_collection is not None:
            self.patch_collection.remove()

        # Collect all polygons and colors from all active slices
        verts_list = []
        colors_list = []

        # Add polygons from each slice dimension
        for slicedim in [0, 1, 2]:
            verts, colors = self.create_slice_polygons(
                slicedim, self.slice_positions[slicedim]
            )
            if verts.shape[0]:
                verts_list.append(verts)
                colors_list.append(colors)

        if verts_list:
            all_polygons = np.concatenate(verts_list, axis=0)
            all_colors = np.concatenate(colors_list)
            # Normalize colors
            if all_colors.max() > 0:
                norm = Normalize(vmin=all_colors.min(), vmax=all_colors.max())
                normalized_colors = norm(all_colors)
            else:
                normalized_colors = all_colors

            params = dict(self.params)
            params.setdefault("alpha", 1.0)
            params.setdefault("edgecolors", "none")
            params.setdefault("linewidths", 0.0)
            params.setdefault("shade", False)
            params.setdefault("antialiased", False)
            params.setdefault("rasterized", True)

            # Discretize the colormap to the current level count (adjusted by
            # right-mouse drag), mirroring mcxplotvol's colormap(jet(level)).
            # self.colormap_name may be a name or a Colormap object (plotmesh
            # accepts both via cmap=).
            levels = int(self.gui_state.get("colormap_levels", 256))
            base = self.colormap_name
            cmap = (
                plt.get_cmap(base, levels)
                if isinstance(base, str)
                else base.resampled(levels)
            )
            facecolors = cmap(normalized_colors)

            # shade=True is matplotlib's closest analog to MATLAB camlight, but
            # its shading path divides by the edge colors and chokes on the
            # 'none' edge color used for a clean slice image. Give it a valid
            # (still invisible, linewidth 0) edge color so it can broadcast.
            if params.get("shade") and params.get("edgecolors") in (None, "none"):
                params["edgecolors"] = facecolors

            # Create single Poly3DCollection
            self.patch_collection = Poly3DCollection(
                all_polygons,
                facecolors=facecolors,
                **params,
            )

            self.ax.add_collection3d(self.patch_collection)

        self.fig.canvas.draw_idle()

    def _project_points(self, pts):
        """
        Forward-project Nx3 data-coordinate points to display pixels.

        Uses matplotlib's own 3D projection (reliable), unlike inverting
        ax.get_proj() by hand. Returns (Nx2 pixel array, depth array); smaller
        depth is closer to the viewer.
        """
        from mpl_toolkits.mplot3d import proj3d

        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        xs, ys, zs = proj3d.proj_transform(
            pts[:, 0], pts[:, 1], pts[:, 2], self.ax.get_proj()
        )
        disp = self.ax.transData.transform(
            np.column_stack([np.atleast_1d(xs), np.atleast_1d(ys)])
        )
        return np.atleast_2d(disp), np.atleast_1d(zs)

    @staticmethod
    def _quad_pick_depth(disp, zs, px, py):
        """
        If pixel (px, py) lies inside the projected slice rectangle, return the
        view depth there, else None. The rectangle projects to a parallelogram
        under matplotlib's orthographic 3D projection, so the click's affine
        (u, v) coordinates give an exact inside-test and a depth interpolation --
        which lets overlapping slices be ordered by depth *at the click point*
        (not by an averaged depth). Smaller depth is closer to the viewer.
        """
        c0, c1, c3 = disp[0], disp[1], disp[3]
        basis = np.array(
            [[c1[0] - c0[0], c3[0] - c0[0]], [c1[1] - c0[1], c3[1] - c0[1]]]
        )
        try:
            u, v = np.linalg.solve(
                basis, np.array([px - c0[0], py - c0[1]], dtype=float)
            )
        except np.linalg.LinAlgError:
            return None
        tol = 1e-9
        if -tol <= u <= 1 + tol and -tol <= v <= 1 + tol:
            return float(zs[0] + u * (zs[1] - zs[0]) + v * (zs[3] - zs[0]))
        return None

    def _slice_axis_screen(self, slicedim):
        """
        Screen-space displacement (in pixels) produced by advancing ``slicedim``
        by one voxel, evaluated at the volume center. Mirrors slice3i's
        projection of the click ray onto the slice-normal axis: dragging the
        cursor along this 2D vector moves the slice by the matching index step.
        """
        center = np.array(self.current_vol.shape, dtype=float) / 2.0
        normal = np.zeros(3)
        normal[slicedim] = 1.0
        disp, _ = self._project_points(np.vstack([center, center + normal]))
        return disp[1] - disp[0]

    def get_slice_rectangle_bounds(self, slicedim, sliceidx):
        """
        Get the rectangle boundary coordinates
        Returns 4 corner points of the rectangular slice in volume coordinates
        """
        vol_shape = self.current_vol.shape

        if slicedim == 0:  # X-slice (cutting perpendicular to X-axis, showing YZ plane)
            # Fixed X coordinate, varying Y and Z
            corners = np.array(
                [
                    [sliceidx, 0, 0],  # (X, Y_min, Z_min)
                    [sliceidx, vol_shape[1] - 1, 0],  # (X, Y_max, Z_min)
                    [sliceidx, vol_shape[1] - 1, vol_shape[2] - 1],  # (X, Y_max, Z_max)
                    [sliceidx, 0, vol_shape[2] - 1],  # (X, Y_min, Z_max)
                ]
            )

        elif (
            slicedim == 1
        ):  # Y-slice (cutting perpendicular to Y-axis, showing XZ plane)
            # Fixed Y coordinate, varying X and Z
            corners = np.array(
                [
                    [0, sliceidx, 0],  # (X_min, Y, Z_min)
                    [vol_shape[0] - 1, sliceidx, 0],  # (X_max, Y, Z_min)
                    [vol_shape[0] - 1, sliceidx, vol_shape[2] - 1],  # (X_max, Y, Z_max)
                    [0, sliceidx, vol_shape[2] - 1],  # (X_min, Y, Z_max)
                ]
            )

        else:  # slicedim == 2, Z-slice (cutting perpendicular to Z-axis, showing XY plane)
            # Fixed Z coordinate, varying X and Y
            corners = np.array(
                [
                    [0, 0, sliceidx],  # (X_min, Y_min, Z)
                    [vol_shape[0] - 1, 0, sliceidx],  # (X_max, Y_min, Z)
                    [vol_shape[0] - 1, vol_shape[1] - 1, sliceidx],  # (X_max, Y_max, Z)
                    [0, vol_shape[1] - 1, sliceidx],  # (X_min, Y_max, Z)
                ]
            )

        return corners

    def detect_clicked_slice(self, event):
        """
        Determine which slice was clicked by forward-projecting each slice's
        rectangle to screen pixels and testing which projected quad contains the
        cursor. When several overlap, the front-most (smallest depth) wins. This
        mirrors MATLAB slice3i, where each slice is its own clickable object.
        """
        if event.x is None or event.y is None:
            return self.detect_slice_from_view_angle()

        best_dim, best_depth = None, None
        for slicedim in (0, 1, 2):
            corners = self.get_slice_rectangle_bounds(
                slicedim, self.slice_positions[slicedim]
            ).astype(float)
            disp, zs = self._project_points(corners)
            depth = self._quad_pick_depth(disp, zs, event.x, event.y)
            if depth is not None and (best_depth is None or depth < best_depth):
                best_dim, best_depth = slicedim, depth

        if best_dim is not None:
            return best_dim

        # No slice under the cursor -> fall back to a view-angle heuristic
        return self.detect_slice_from_view_angle()

    def detect_slice_from_view_angle(self):
        """
        Fallback method: detect slice based on current view angles
        More reliable when 3D coordinates are not available
        """
        try:
            elev = self.ax.elev
            azim = self.ax.azim

            # Normalize angles
            azim = azim % 360
            elev = max(-90, min(90, elev))

            # Simple heuristic based on view angles
            # This determines which slice is most "facing" the camera

            if abs(elev) > 60:  # Looking from top or bottom
                return 2  # Z-slice
            elif 30 <= azim <= 150 or 210 <= azim <= 330:  # Looking from sides
                return 0  # X-slice
            else:  # Looking from front or back
                return 1  # Y-slice

        except (AttributeError, ValueError):
            # Ultimate fallback
            return 2  # Default to Z-slice

    def update_frame(self, frame):
        """Switch the displayed frame of a 4-D volume (keeps slice positions)."""
        if not self.is_multiframe:
            return
        self.frame = int(np.clip(frame, 0, self.vol.shape[3] - 1))
        self.current_vol = self.vol[:, :, :, self.frame]
        self._update_title()
        self.update_patches()

    def _begin_interactive(self):
        """
        Enter a coarse level-of-detail while dragging/rotating. matplotlib has no
        GPU acceleration for Poly3DCollection, so rendering every voxel quad is
        slow on large volumes; temporarily increasing the subsample keeps
        interaction responsive. Full resolution is restored in _end_interactive.
        """
        if self.gui_state.get("full_subsample") is not None:
            return
        self.gui_state["full_subsample"] = self.subsample
        # target ~40 quads along the largest dimension during interaction
        coarse = int(np.ceil(max(self.current_vol.shape) / 40.0))
        coarse = max(self.subsample, coarse)
        if coarse != self.subsample:
            self.subsample = coarse
            self.update_patches()

    def _end_interactive(self):
        """Restore full resolution after an interactive drag/rotation ends."""
        full = self.gui_state.get("full_subsample")
        self.gui_state["full_subsample"] = None
        if full is not None and self.subsample != full:
            self.subsample = full
            self.update_patches()

    def on_mouse_press(self, event):
        """Handle mouse press events"""
        if event.inaxes != self.ax:
            return

        if event.button == 2:  # Middle mouse - toggle rotation mode
            self.gui_state["rotation_enabled"] = True
            self.gui_state["rotating"] = True
            self.gui_state["last_mouse_pos"] = (
                event.x,
                event.y,
            )  # Use pixel coordinates
            self._begin_interactive()
            # Change cursor to indicate rotation mode
            try:
                self.fig.canvas.set_cursor(2)  # Hand cursor
            except:
                pass  # Ignore cursor errors
            return

        elif event.button == 3:  # Right mouse - colormap adjustment
            if not self.gui_state[
                "rotating"
            ]:  # Don't allow colormap adjustment during rotation
                self.gui_state["adjusting_colormap"] = True
                self.gui_state["start_mouse_pos"] = (event.x, event.y)
                # remember the level at drag start, like slice3i's level0
                self.gui_state["level0"] = self.gui_state["colormap_levels"]
            return

        elif event.button == 1:  # Left mouse - slice dragging
            if not self.gui_state[
                "rotating"
            ]:  # Don't allow slice dragging during rotation
                self.gui_state["dragging"] = True
                self.gui_state["start_mouse_pos"] = (event.x, event.y)

                # Detect the clicked slice by forward projection (robust), then
                # cache the screen direction of its normal for slice3i-style drag.
                # Detect BEFORE switching to coarse LOD so the click uses the
                # currently displayed slice geometry.
                detected_slice = self.detect_clicked_slice(event)
                self.gui_state["active_slicedim"] = detected_slice
                self.gui_state["start_idx"] = self.slice_positions[detected_slice]
                self.gui_state["drag_axis_screen"] = self._slice_axis_screen(
                    detected_slice
                )
                self._begin_interactive()

                # Change cursor to indicate dragging mode
                try:
                    self.fig.canvas.set_cursor(4)  # Move cursor
                except:
                    pass  # Ignore cursor errors

    def on_mouse_release(self, event):
        """Handle mouse release events"""
        if event.button == 2:  # Middle mouse release - disable rotation
            self.gui_state["rotating"] = False
            self.gui_state["rotation_enabled"] = False
            self.gui_state["last_mouse_pos"] = None
            self._end_interactive()
            try:
                self.fig.canvas.set_cursor(1)  # Default cursor
            except:
                pass

        elif event.button == 3:  # Right mouse release
            self.gui_state["adjusting_colormap"] = False
            self.gui_state["start_mouse_pos"] = None
            try:
                self.fig.canvas.set_cursor(1)  # Default cursor
            except:
                pass

        elif event.button == 1:  # Left mouse release
            self.gui_state["dragging"] = False
            self.gui_state["start_mouse_pos"] = None
            self.gui_state["start_idx"] = None
            self.gui_state["active_slicedim"] = None
            self._end_interactive()
            try:
                self.fig.canvas.set_cursor(1)  # Default cursor
            except:
                pass

    def on_mouse_move(self, event):
        """Handle mouse move events"""
        # Use pixel coordinates (valid even off-axis) so a drag/rotation keeps
        # working when the cursor briefly leaves the axes, like MATLAB's
        # window-level motion callback.
        if event.x is None or event.y is None:
            return

        if self.gui_state["rotating"] and self.gui_state["last_mouse_pos"]:
            # Manual 3D rotation (middle mouse drag)
            if event.x is not None and event.y is not None:
                last_x, last_y = self.gui_state["last_mouse_pos"]
                dx = event.x - last_x
                dy = event.y - last_y

                # Get current view angles
                elev = self.ax.elev
                azim = self.ax.azim

                # Update angles based on mouse movement (pixel coordinates).
                # Match MATLAB rotate3d / mcxplotvol: dragging right decreases
                # azimuth, dragging up decreases elevation. (event.y grows
                # upward, so dy > 0 means the cursor moved up.)
                azim_sensitivity = 0.5
                elev_sensitivity = 0.5

                new_azim = azim - dx * azim_sensitivity
                new_elev = elev - dy * elev_sensitivity

                # Clamp elevation to reasonable bounds
                new_elev = max(-90, min(90, new_elev))

                # Apply new view
                self.ax.view_init(elev=new_elev, azim=new_azim)

                # Update axis limits to keep slices properly visible after rotation
                self.set_equal_aspect()

                self.gui_state["last_mouse_pos"] = (event.x, event.y)
                self.fig.canvas.draw_idle()

        elif self.gui_state["adjusting_colormap"] and self.gui_state["start_mouse_pos"]:
            # Colormap level adjustment (right mouse drag), matching slice3i.m:
            #   currentlevel = round(level0 + delta_y_norm / 4 * 256)
            # i.e. a full-window vertical drag changes the count by ~64 colors,
            # relative to the level when the drag started (drag up = more levels).
            start_x, start_y = self.gui_state["start_mouse_pos"]
            fig_height = self.fig.get_size_inches()[1] * self.fig.dpi
            normalized_delta = (event.y - start_y) / fig_height

            level0 = self.gui_state.get("level0", self.gui_state["colormap_levels"])
            new_levels = int(round(level0 + normalized_delta / 4 * 256))
            new_levels = max(8, min(256, new_levels))

            if new_levels != self.gui_state["colormap_levels"]:
                self.gui_state["colormap_levels"] = new_levels
                self.update_patches()

        elif self.gui_state["dragging"] and self.gui_state["start_mouse_pos"]:
            # Slice dragging (left mouse drag) - slice3i-style: move the slice
            # along its normal proportional to the cursor displacement projected
            # onto the screen direction of that normal. This is correct for any
            # camera orientation/zoom (unlike a fixed pixel sensitivity).
            slicedim = self.gui_state.get("active_slicedim", 2)
            u = self.gui_state.get("drag_axis_screen")
            if u is None:
                return
            start_x, start_y = self.gui_state["start_mouse_pos"]
            delta = np.array([event.x - start_x, event.y - start_y], dtype=float)

            uu = float(np.dot(u, u))
            if uu < 1e-6:  # normal nearly parallel to view -> can't drag reliably
                return
            slice_delta = float(np.dot(delta, u)) / uu

            max_slices = self.current_vol.shape[slicedim]
            new_idx = self.gui_state["start_idx"] + slice_delta
            new_idx = int(round(max(0, min(max_slices - 1, new_idx))))

            if new_idx != self.slice_positions[slicedim]:
                self.slice_positions[slicedim] = new_idx
                self.update_patches()

    def on_key_press(self, event):
        """Up/Down arrows step through frames of a 4-D volume (like mcxplotvol)."""
        if self.is_multiframe:
            if event.key == "up":
                self.update_frame(self.frame + 1)
            elif event.key == "down":
                self.update_frame(self.frame - 1)

    def set_quality(self, subsample):
        """
        Set rendering quality by changing subsample factor

        Parameters:
        subsample: 1=highest quality (slow), 2=high quality, 4=fast, 8=fastest
        """
        self.subsample = max(1, int(subsample))
        self.update_patches()

    def setslice(self, slicedim, sliceidx):
        """
        Set a specific slice position (MATLAB-compatible interface)
        slicedim: 0, 1, or 2 for i, j, k dimensions
        sliceidx: slice index
        """
        self.slice_positions[slicedim] = int(sliceidx)
        self.update_patches()

    def show(self):
        """Display the interactive viewer"""
        # Don't call plt.show() here since the figure is already created and managed
        # Just make sure the figure is drawn
        self.fig.canvas.draw()
        plt.show()
