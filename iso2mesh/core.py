"""@package docstring
Iso2Mesh for Python - Mesh data queries and manipulations

Copyright (c) 2024-2025 Qianqian Fang <q.fang at neu.edu>
"""
__all__ = [
    "v2m",
    "v2s",
    "s2m",
    "s2v",
    "vol2mesh",
    "vol2surf",
    "surf2mesh",
    "surf2volz",
    "surf2vol",
    "binsurface",
    "cgalv2m",
    "cgals2m",
    "getintersecttri",
    "vol2restrictedtri",
    "fillsurf",
    "outersurf",
    "surfvolume",
    "insurface",
]

##====================================================================================
## dependent libraries
##====================================================================================

import numpy as np
import os
import re
import platform
import subprocess
from iso2mesh.trait import (
    surfinterior,
    surfseeds,
    surfedge,
    elemvolume,
    meshreorient,
    finddisconnsurf,
)
from iso2mesh.utils import *
from iso2mesh.io import saveoff, readoff, saveinr, readtetgen, savesurfpoly, readmedit
from iso2mesh.modify import (
    meshcheckrepair,
    sortmesh,
    meshresample,
    removeisolatedsurf,
    removeisolatednode,
)

##====================================================================================
## implementations
##====================================================================================


def v2m(img, isovalues, opt=None, maxvol=None, method=None):
    """
    Volumetric mesh generation from binary or gray-scale volumetric images.

    Parameters:
    img       : 3D numpy array, volumetric image data
    isovalues : scalar or list, isovalues to generate meshes
    opt       : options for mesh generation (default: None)
    maxvol    : maximum volume for elements (default: None)
    method    : method for surface extraction, default is 'cgalsurf'

    Returns:
    node      : generated mesh nodes
    elem      : elements of the mesh
    face      : surface triangles
    """
    if method is None:
        method = "cgalsurf"

    # Generate the mesh using vol2mesh (assumes vol2mesh exists in the Python environment)
    node, elem, face = vol2mesh(
        img,
        np.arange(img.shape[0]),
        np.arange(img.shape[1]),
        np.arange(img.shape[2]),
        opt,
        maxvol,
        1,
        method,
        isovalues,
    )

    return node, elem, face


def v2s(img, isovalues, opt=None, method=None):
    """
    Surface mesh generation from binary or grayscale volumetric images.

    Parameters:
    img       : 3D numpy array, volumetric image data
    isovalues : scalar or list, isovalues to generate meshes
    opt       : options for mesh generation (default: None)
    method    : method for surface extraction, default is 'cgalsurf'

    Returns:
    no        : generated mesh nodes
    el        : elements of the mesh
    regions   : mesh regions
    holes     : mesh holes
    """
    if method is None:
        method = "cgalsurf"

    if method == "cgalmesh":
        no, tet, el = v2m(np.uint8(img), isovalues, opt, 1000, method)
        regions = []
        fclist = np.unique(el[:, 3])

        for fc in fclist:
            pt = surfinterior(no[:, :3], el[el[:, 3] == fc, :3])[0]
            if pt.size > 0:
                regions.append(pt)

        el = np.unique(el[:, :3], axis=0)
        no, el = removeisolatednode(no[:, :3], el[:, :3])
        holes = []
        return no, el, regions, holes

    no, el, regions, holes = vol2surf(
        img,
        np.arange(img.shape[0]),
        np.arange(img.shape[1]),
        np.arange(img.shape[2]),
        opt,
        1,
        method,
        isovalues,
    )

    return no, el, regions, holes


def s2m(
    v, f, keepratio=None, maxvol=None, method="tetgen", regions=None, holes=None, *args
):
    """
    Volumetric mesh generation from a closed surface, shortcut for surf2mesh.

    Parameters:
    v        : vertices of the surface
    f        : faces of the surface
    keepratio: ratio of triangles to preserve or a structure of options (for 'cgalpoly')
    maxvol   : maximum volume of mesh elements
    method   : method to use ('tetgen' by default or 'cgalpoly')
    regions  : predefined mesh regions
    holes    : holes in the mesh

    Returns:
    node     : generated mesh nodes
    elem     : elements of the mesh
    face     : surface triangles
    """
    if method == "cgalpoly":
        node, elem, face = cgals2m(v, f, keepratio, maxvol)
        return node, elem, face

    if regions is None:
        regions = []
    if holes is None:
        holes = []

    if args:
        node, elem, face = surf2mesh(
            v, f, [], [], keepratio, maxvol, regions, holes, 0, method, *args
        )
    else:
        node, elem, face = surf2mesh(
            v, f, [], [], keepratio, maxvol, regions, holes, 0, method
        )

    return node, elem, face


def s2v(node, face, div=50, *args):
    """
    Convert a surface mesh to a volumetric binary image.

    Parameters:
    node   : array-like, the vertices of the triangular surface (Nx3 for x, y, z)
    face   : array-like, the triangle node indices (Mx3, each row is a triangle)
    div    : int, division number along the shortest edge of the mesh (resolution)
    *args  : additional arguments for the surf2vol function

    Returns:
    img    : volumetric binary image
    v2smap : 4x4 affine transformation matrix to map voxel coordinates back to the mesh space
    """
    p0 = np.min(node, axis=0)
    p1 = np.max(node, axis=0)

    if node.shape[0] == 0 or face.shape[0] == 0:
        raise ValueError("node and face cannot be empty")

    if div == 0:
        raise ValueError("div cannot be 0")

    dx = np.min(p1 - p0) / div

    if dx <= np.finfo(float).eps:
        raise ValueError("the input mesh is in a plane")

    xi = np.arange(p0[0] - dx, p1[0] + dx, dx)
    yi = np.arange(p0[1] - dx, p1[1] + dx, dx)
    zi = np.arange(p0[2] - dx, p1[2] + dx, dx)

    img, v2smap = surf2vol(node, face, xi, yi, zi, *args)

    return img, v2smap


def vol2mesh(img, ix, iy, iz, opt, maxvol, dofix, method="cgalsurf", isovalues=None):
    """
    Convert a binary or multi-valued volume to a tetrahedral mesh.

    Parameters:
    img       : 3D numpy array, volumetric image data
    ix, iy, iz: indices for subvolume selection in x, y, z directions
    opt       : options for mesh generation
    maxvol    : maximum volume for mesh elements
    dofix     : boolean, whether to validate and repair the mesh
    method    : method for mesh generation ('cgalsurf', 'simplify', 'cgalmesh', 'cgalpoly')
    isovalues : list of isovalues for the levelset (optional)

    Returns:
    node      : node coordinates of the mesh
    elem      : element list of the mesh (last column is region ID)
    face      : surface elements of the mesh (last column is boundary ID)
    regions   : interior points for closed surfaces
    """

    if method == "cgalmesh":
        vol = img[np.ix_(ix, iy, iz)]
        if len(np.unique(vol)) > 64 and dofix == 1:
            raise ValueError(
                "CGAL mesher does not support grayscale images. Use 'cgalsurf' for grayscale volumes."
            )
        node, elem, face = cgalv2m(vol, opt, maxvol)
        return node, elem, face

    if isovalues is not None:
        no, el, regions, holes = vol2surf(
            img, ix, iy, iz, opt, dofix, method, isovalues
        )
    else:
        no, el, regions, holes = vol2surf(img, ix, iy, iz, opt, dofix, method)

    if method == "cgalpoly":
        node, elem, face = cgals2m(no[:, :3], el[:, :3], opt, maxvol)
        return node, elem, face

    node, elem, face = surf2mesh(no, el, [], [], 1, maxvol, regions, holes)
    return node, elem, face


def vol2surf(img, ix, iy, iz, opt, dofix=0, method="cgalsurf", isovalues=None):
    """
    Convert a 3D volumetric image to surfaces.

    Parameters:
    img: volumetric binary image. If img is empty, vol2surf will return user-defined surfaces via opt.surf if it exists.
    ix, iy, iz: subvolume selection indices in x, y, z directions.
    opt: options dict containing function parameters.
    dofix: 1 to perform mesh validation and repair, 0 to skip repairing.
    method: meshing method ('simplify', 'cgalsurf', or 'cgalpoly'), defaults to 'cgalsurf'.
    isovalues: list of isovalues for level sets.

    Returns:
    no: node list on the resulting surface mesh, with 3 columns for x, y, z.
    el: list of triangular elements on the surface [n1, n2, n3, region_id].
    regions: list of interior points for all sub-regions.
    holes: list of interior points for all holes.
    """

    print("Extracting surfaces from a volume...")

    if isinstance(opt, (int, float)):
        opt = {"radbound": opt}

    el = np.array([])
    no = np.array([])
    holes = opt.get("holes", [])
    regions = opt.get("regions", [])

    if img is not None and len(img) > 0:
        img = img[np.ix_(ix, iy, iz)]
        dim = img.shape
        newdim = np.array(dim) + 2
        newimg = np.zeros(newdim, dtype=img.dtype)
        newimg[1:-1, 1:-1, 1:-1] = img

        if isovalues is None:
            maxlevel = newimg.max()
            isovalues = np.arange(1, maxlevel + 1)
        else:
            tmp = np.array([isovalues]).flatten()
            isovalues = np.unique(np.sort(tmp))
            maxlevel = len(isovalues)

        for i in range(maxlevel):
            if i < maxlevel - 1:
                levelmask = (newimg >= isovalues[i]) & (newimg < isovalues[i + 1])
            else:
                levelmask = newimg >= isovalues[i]

            levelno, levelel = binsurface(levelmask)

            if levelel.size > 0:
                if opt.get("autoregion", 0):
                    seeds = surfseeds(levelno, levelel)
                else:
                    seeds = surfinterior(levelno, levelel)[0]
                if len(seeds) > 0:
                    print(f"Region {i + 1} centroid: {seeds}")
                    regions = np.vstack((regions, seeds)) if len(regions) > 0 else seeds

        for i in range(maxlevel):
            print(f"Processing threshold level {i + 1}...")
            if method == "simplify":
                v0, f0 = binsurface(newimg >= isovalues[i])
                if dofix:
                    v0, f0 = meshcheckrepair(v0, f0)

                keepratio = (
                    opt.get("keepratio", 1)
                    if isinstance(opt, dict)
                    else opt[i].get("keepratio", 1)
                )
                print(f"Resampling surface mesh for level {i + 1}...")
                v0, f0 = meshresample(v0, f0, keepratio)
                f0 = removeisolatedsurf(v0, f0, 3)

                if dofix:
                    v0, f0 = meshcheckrepair(v0, f0)
            else:
                radbound = (
                    opt.get("radbound", 1)
                    if isinstance(opt, dict)
                    else opt[i].get("radbound", 1)
                )
                distbound = (
                    opt.get("distbound", radbound)
                    if isinstance(opt, dict)
                    else opt[i].get("distbound", radbound)
                )
                maxsurfnode = (
                    opt.get("maxnode", 40000)
                    if isinstance(opt, dict)
                    else opt[i].get("maxnode", 40000)
                )
                surfside = (
                    opt.get("side", "")
                    if isinstance(opt, dict)
                    else opt[i].get("side", "")
                )

                if surfside == "upper":
                    newimg[newimg <= isovalues[i] - 1e-9] = isovalues[i] - 1e-9
                elif surfside == "lower":
                    newimg[newimg >= isovalues[i] + 1e-9] = isovalues[i] + 1e-9

                perturb = 1e-4 * np.abs(isovalues).max()
                perturb = (
                    -perturb if np.all(newimg > isovalues[i] - perturb) else perturb
                )

                if regions.ndim == 1:
                    regions = regions[np.newaxis, :]

                v0, f0 = vol2restrictedtri(
                    newimg,
                    isovalues[i] - perturb,
                    regions[i],
                    np.sum(newdim**2) * 2,
                    30,
                    radbound,
                    distbound,
                    maxsurfnode,
                )

            if opt.get("maxsurf", 0) == 1:
                f0 = maxsurf(finddisconnsurf(f0))

            if "A" in opt and "B" in opt:
                v0 = (opt["A"] @ v0.T + opt["B"][:, None]).T

            if "hole" in opt:
                holes = np.vstack((holes, opt["hole"]))
            if "region" in opt:
                regions = np.vstack((regions, opt["region"]))

            e0 = np.hstack(
                (f0 + len(no), np.ones((f0.shape[0], 1), dtype=int) * (i + 1))
            )
            el = np.vstack((el, e0)) if len(el) > 0 else e0
            no = np.vstack((no, v0)) if len(no) > 0 else v0

    if "surf" in opt:
        for surf in opt["surf"]:
            surf["elem"][:, 3] = maxlevel + 1
            el = np.vstack((el, surf["elem"] + len(no)))
            no = np.vstack((no, surf["node"]))

    print("Surface mesh generation is complete")

    return no, el, regions, holes


# _________________________________________________________________________________________________________


def surf2mesh(
    v,
    f,
    p0,
    p1,
    keepratio,
    maxvol,
    regions=None,
    holes=None,
    dobbx=0,
    method="tetgen",
    cmdopt=None,
):
    """
    Create a quality volumetric mesh from isosurface patches.

    Parameters:
    v: isosurface node list, shape (nn,3). If v has 4 columns, the last column specifies mesh density near each node.
    f: isosurface face element list, shape (be,3). If f has 4 columns, it indicates the label of the face triangles.
    p0: coordinates of one corner of the bounding box, [x0, y0, z0].
    p1: coordinates of the other corner of the bounding box, [x1, y1, z1].
    keepratio: percentage of elements kept after simplification, between 0 and 1.
    maxvol: maximum volume of tetrahedra elements.
    regions: list of regions, specified by an internal point for each region.
    holes: list of holes, similar to regions.
    forcebox: 1 to add bounding box, 0 for automatic.
    method: meshing method (default is 'tetgen').
    cmdopt: additional options for the external mesh generator.

    Returns:
    node: node coordinates of the tetrahedral mesh.
    elem: element list of the tetrahedral mesh.
    face: mesh surface element list, with the last column denoting the boundary ID.
    """
    if keepratio > 1 or keepratio < 0:
        print(
            'The "keepratio" parameter must be between 0 and 1. No simplification will be performed.'
        )

    exesuff = getexeext()

    # Resample surface mesh if keepratio is less than 1
    if keepratio < 1 and not isinstance(f, list):
        print("Resampling surface mesh...")
        no, el = meshresample(v[:, :3], f[:, :3], keepratio)
        el = np.unique(np.sort(el, axis=1), axis=0)
    else:
        no = v
        el = f

    # Handle regions and holes arguments
    if regions is None:
        regions = np.array([])  # []
    if holes is None:
        holes = np.array([])

    # Warn if both maxvol and region-based volume constraints are specified
    if (
        isinstance(regions, np.ndarray)
        and regions.ndim > 1
        and regions.shape[1] >= 4
        and maxvol is not None
    ):
        print(
            "Warning: Both maxvol and region-based volume constraints are specified. maxvol will be ignored."
        )
        maxvol = None

    # Dump surface mesh to .poly file format
    if not isinstance(el, list) and no.size and el.size:
        saveoff(no[:, :3], el[:, :3], "post_vmesh.off")
    deletemeshfile(mwpath("post_vmesh.mtr"))
    savesurfpoly(
        no, el, holes, regions, p0, p1, mwpath("post_vmesh.poly"), forcebox=dobbx
    )

    moreopt = ""
    if len(no.shape) > 1 and no.shape[1] == 4:
        moreopt = moreopt + " -m "
    # Generate volumetric mesh from surface mesh
    deletemeshfile(mwpath("post_vmesh.1.*"))
    print("Creating volumetric mesh from surface mesh...")

    if cmdopt is None:
        try:
            cmdopt = eval("ISO2MESH_TETGENOPT")
        except:
            cmdopt = ""
    cmdstr = f'"{mcpath(method, exesuff)}" -A -q1.414a{maxvol} {moreopt} ' + mwpath(
        "post_vmesh.poly"
    )
    try:
        if not sys.platform.startswith("win"):
            import tetgen

            tgen = tetgen.TetGen(filename=mwpath("post_vmesh.poly"))
            mesh = tgen.tetrahedralize(flags=cmdopt)
            node = mesh["vertices"]
            elem = mesh["tetrahedra"]
            face = mesh["faces"]
    except:
        pass

    if "mesh" not in locals():
        if not cmdopt:
            status, cmdout = subprocess.getstatusoutput(cmdstr)
        else:
            cmdstr = f'"{mcpath(method, exesuff)}" {cmdopt} ' + mwpath(
                "post_vmesh.poly"
            )
            status, cmdout = subprocess.getstatusoutput(
                f'"{mcpath(method, exesuff)}" {cmdopt} ' + mwpath("post_vmesh.poly")
            )

        if status != 0:
            raise RuntimeError(f"Tetgen command failed: {cmdstr}\n{cmdout}")

        # Read generated mesh
        node, elem, face = readtetgen(mwpath("post_vmesh.1"))

    print("Volume mesh generation complete")
    return node, elem, face


def surf2volz(node, face, xi, yi, zi):
    """
    Convert a triangular surface to a shell of voxels in a 3D image along the z-axis.

    Parameters:
    node: node list of the triangular surface, with 3 columns for x/y/z
    face: triangle node indices, each row represents a triangle
    xi, yi, zi: x/y/z grid for the resulting volume

    Returns:
    img: a volumetric binary image at the position of ndgrid(xi, yi, zi)
    """

    ne = face.shape[0]
    img = np.zeros((len(xi), len(yi), len(zi)), dtype=np.uint8)
    dx0 = np.min(np.abs(np.diff(xi)))
    dx = dx0 / 2
    dy0 = np.min(np.abs(np.diff(yi)))
    dy = dy0 / 2
    dz0 = np.min(np.abs(np.diff(zi)))
    dl = np.sqrt(dx**2 + dy**2)
    minz = np.min(node[:, 2])
    maxz = np.max(node[:, 2])

    # Determine the z index range
    iz = np.histogram([minz, maxz], bins=zi)[0]
    hz = np.nonzero(iz)[0]
    iz = np.arange(hz[0], min(len(zi), hz[-1] + 1))

    for i in iz:
        plane = np.array([[0, 100, zi[i]], [100, 0, zi[i]], [0, 0, zi[i]]])
        bcutpos, bcutvalue, bcutedges = qmeshcut(face[:, :3], node, node[:, 0], plane)

        if bcutpos.size == 0:
            continue

        enum = bcutedges.shape[0]

        for j in range(enum):
            e0 = bcutpos[bcutedges[j, 0], :2]
            e1 = bcutpos[bcutedges[j, 1], :2]
            length = np.ceil(np.sum(np.abs(e1 - e0)) / (np.abs(dx) + np.abs(dy))) + 1
            dd = (e1 - e0) / length

            posx = np.floor(
                (e0[0] + np.arange(length + 1) * dd[0] - xi[0]) / dx0
            ).astype(int)
            posy = np.floor(
                (e0[1] + np.arange(length + 1) * dd[1] - yi[0]) / dy0
            ).astype(int)
            pos = np.vstack((posx, posy)).T

            pos = pos[(posx > 0) & (posx <= len(xi)) & (posy > 0) & (posy <= len(yi))]

            if len(pos) > 0:
                zz = np.floor((zi[i] - zi[0]) / dz0).astype(int)
                for k in range(pos.shape[0]):
                    img[pos[k, 0], pos[k, 1], zz] = 1

    return img


def surf2vol(node, face, xi, yi, zi, **kwargs):
    """
    Convert a triangular surface to a shell of voxels in a 3D image.

    Parameters:
    node: node list of the triangular surface, 3 columns for x/y/z
    face: triangle node indices, each row is a triangle
          If face contains a 4th column, it indicates the label of the face triangles.
          If face contains 5 columns, it stores a tetrahedral mesh with labels.
    xi, yi, zi: x/y/z grid for the resulting volume
    kwargs: optional parameters:
        'fill': if set to 1, the enclosed voxels are labeled as 1.
        'label': if set to 1, the enclosed voxels are labeled by the corresponding label of the face or element.
                 Setting 'label' to 1 also implies 'fill'.

    Returns:
    img: a volumetric binary image at the position of ndgrid(xi, yi, zi)
    v2smap (optional): a 4x4 matrix denoting the Affine transformation to map voxel coordinates back to the mesh space.
    """

    opt = kwargs
    label = opt.get("label", 0)
    elabel = 1

    if face.shape[1] >= 4:
        elabel = np.unique(face[:, -1])
        if face.shape[1] == 5:
            label = 1
            el = face
            face = np.empty((0, 4))
            for lbl in elabel:
                fc = volface(el[el[:, 4] == lbl, :4])
                fc = np.hstack((fc, np.full((fc.shape[0], 1), lbl)))
                face = np.vstack((face, fc))
    else:
        fc = face

    img = np.zeros((len(xi), len(yi), len(zi)), dtype=elabel.dtype)

    for lbl in elabel:
        if face.shape[1] == 4:
            fc = face[face[:, 3] == lbl, :3]

        im = surf2volz(node[:, :3], fc[:, :3], xi, yi, zi)
        im |= np.moveaxis(surf2volz(node[:, [2, 0, 1]], fc[:, :3], zi, xi, yi), 0, 2)
        im |= np.moveaxis(surf2volz(node[:, [1, 2, 0]], fc[:, :3], yi, zi, xi), 0, 1)

        if opt.get("fill", 0) or label:
            im = imfill(im, "holes")
            if label:
                im = im.astype(elabel.dtype) * lbl

        img = np.maximum(im.astype(img.dtype), img)

    v2smap = None
    if "v2smap" in kwargs:
        dlen = np.abs([xi[1] - xi[0], yi[1] - yi[0], zi[1] - zi[0]])
        offset = np.min(node, axis=0)
        v2smap = np.eye(4)
        v2smap[:3, :3] = np.diag(np.abs(dlen))
        v2smap[:3, 3] = offset

    return img, v2smap


def binsurface(img, nface=3):
    """
    node, elem = binsurface(img, nface)

    Fast isosurface extraction from 3D binary images.

    Parameters:
        img: 3D binary NumPy array
        nface:
            = 3 or omitted: triangular faces
            = 4: square (quad) faces
            = 0: return boundary mask image via `node`
            = 'iso': use marching cubes (`isosurface` equivalent)

    Returns:
        node: (N, 3) array of vertex coordinates
        elem: (M, 3) or (M, 4) array of face elements (1-based indices)
    """
    if isinstance(nface, str) and nface == "iso":
        from skimage.measure import marching_cubes

        verts, faces, _, _ = marching_cubes(img, level=0.5)
        node = verts[:, [1, 0, 2]] - 0.5  # reorder to match MATLAB
        elem = faces + 1  # 1-based indexing
        return node, elem

    dim = list(img.shape)
    if len(dim) < 3:
        dim += [1]
    newdim = [d + 1 for d in dim]

    # Compute differences in each direction
    d1 = np.diff(img, axis=0)
    d2 = np.diff(img, axis=1)
    d3 = np.diff(img, axis=2)

    ix, iy, iz = np.where((d1 == 1) | (d1 == -1))
    jx, jy, jz = np.where((d2 == 1) | (d2 == -1))
    kx, ky, kz = np.where((d3 == 1) | (d3 == -1))

    ix += 1
    jy += 1
    kz += 1

    # Adjust indices and wrap them to 3D
    id1 = np.ravel_multi_index((ix, iy, iz), newdim, order="F")
    id2 = np.ravel_multi_index((jx, jy, jz), newdim, order="F")
    id3 = np.ravel_multi_index((kx, ky, kz), newdim, order="F")

    if nface == 0:
        elem = np.concatenate([id1, id2, id3])
        node = np.zeros(newdim, dtype=np.int8)
        node.flat[elem] = 1
        node = node[1:-1, 1:-1, 1:-1] - 1
        return node, elem

    xy = newdim[0] * newdim[1]

    if nface == 3:  # triangles
        elem = np.vstack(
            [
                np.column_stack([id1, id1 + newdim[0], id1 + newdim[0] + xy]),
                np.column_stack([id1, id1 + newdim[0] + xy, id1 + xy]),
                np.column_stack([id2, id2 + 1, id2 + 1 + xy]),
                np.column_stack([id2, id2 + 1 + xy, id2 + xy]),
                np.column_stack([id3, id3 + 1, id3 + 1 + newdim[0]]),
                np.column_stack([id3, id3 + 1 + newdim[0], id3 + newdim[0]]),
            ]
        )
    else:  # quads
        elem = np.vstack(
            [
                np.column_stack([id1, id1 + newdim[0], id1 + newdim[0] + xy, id1 + xy]),
                np.column_stack([id2, id2 + 1, id2 + 1 + xy, id2 + xy]),
                np.column_stack([id3, id3 + 1, id3 + 1 + newdim[0], id3 + newdim[0]]),
            ]
        )

    # Compress the node indices
    maxid = np.max(elem) + 1
    nodemap = np.zeros(maxid, dtype=int)
    nodemap[elem.ravel(order="F")] = 1
    id = np.where(nodemap)[0]

    # Reindex elem to be compact and 1-based
    nodemap = np.zeros_like(nodemap)
    nodemap[id] = np.arange(1, len(id) + 1)  # 1-based
    elem = nodemap[elem]

    # Create node coordinates
    xi, yi, zi = np.unravel_index(id, newdim, order="F")
    node = np.column_stack([xi, yi, zi])

    if nface == 3:
        node, elem = meshcheckrepair(node, elem)

    return node, elem


def cgalv2m(vol, opt, maxvol):
    """
    Wrapper for CGAL 3D mesher (CGAL 3.5 or up) to convert a binary or multi-valued volume to tetrahedral mesh.

    Parameters:
    vol: a volumetric binary image.
    opt: parameters for the CGAL mesher. If opt is a structure:
        opt.radbound: maximum surface element size.
        opt.angbound: minimum angle of a surface triangle.
        opt.distbound: maximum distance between the center of the surface bounding circle and the element bounding sphere.
        opt.reratio: maximum radius-edge ratio.
        If opt is a scalar, it specifies radbound.
    maxvol: target maximum tetrahedral element volume.

    Returns:
    node: node coordinates of the tetrahedral mesh.
    elem: element list of the tetrahedral mesh, the last column is the region ID.
    face: mesh surface element list of the tetrahedral mesh, the last column denotes the boundary ID.
    """

    print("Creating surface and tetrahedral mesh from a multi-domain volume...")

    if not (np.issubdtype(vol.dtype, np.bool_) or vol.dtype == np.uint8):
        raise ValueError(
            "CGAL mesher can only handle uint8 volumes. Convert your image to uint8 first."
        )

    if not np.any(vol):
        raise ValueError("No labeled regions found in the input volume.")

    exesuff = getexeext()
    exesuff = fallbackexeext(exesuff, "cgalmesh")

    ang = 30
    ssize = 6
    approx = 0.5
    reratio = 3

    if not isinstance(opt, dict):
        ssize = opt

    if isinstance(opt, dict):
        ssize = opt.get("radbound", ssize)
        ang = opt.get("angbound", ang)
        approx = opt.get("distbound", approx)
        reratio = opt.get("reratio", reratio)

    saveinr(vol, mwpath("pre_cgalmesh.inr"))
    deletemeshfile(mwpath("post_cgalmesh.mesh"))

    randseed = int("623F9A9E", 16)

    cmd = f'"{mcpath("cgalmesh", exesuff)}" "{mwpath("pre_cgalmesh.inr")}" "{mwpath("post_cgalmesh.mesh")}" {ang} {ssize} {approx} {reratio} {maxvol} {randseed}'

    status, str_output = subprocess.getstatusoutput(cmd)

    if not os.path.exists(mwpath("post_cgalmesh.mesh")):
        raise RuntimeError(f"Output file was not found. Command failed: {cmd}")

    node, elem, face = readmedit(mwpath("post_cgalmesh.mesh"))

    if isinstance(opt, dict) and len(opt) == 1:
        if "A" in opt and "B" in opt:
            node[:, :3] = (
                opt["A"] @ node[:, :3].T
                + np.tile(opt["B"][:, None], (1, node.shape[0]))
            ).T

    print(
        f"Node number: {node.shape[0]}\nTriangles: {face.shape[0]}\nTetrahedra: {elem.shape[0]}\nRegions: {len(np.unique(elem[:, -1]))}"
    )
    print("Surface and volume meshes complete")

    if node.shape[0] > 0:
        node, elem, face, _ = sortmesh(
            node[0, :], node, elem, list(range(4)), face, list(range(3))
        )

    node += 0.5
    elem[:, :4] = meshreorient(node[:, :3], elem[:, :4])[0]

    return node, elem, face


def cgals2m(v, f, opt, maxvol, *args):
    """
    Convert a triangular surface to a tetrahedral mesh using CGAL mesher.

    Parameters:
    v : ndarray
        Node coordinate list of a surface mesh (nn x 3)
    f : ndarray
        Face element list of a surface mesh (be x 3)
    opt : dict or scalar
        Parameters for CGAL mesher. If it's a dict, it can include:
            - radbound: Maximum surface element size
            - angbound: Minimum angle of a surface triangle
            - distbound: Max distance between surface bounding circle center and element bounding sphere center
            - reratio: Maximum radius-edge ratio
        If it's a scalar, it only specifies radbound.
    maxvol : float
        Target maximum tetrahedral element volume.
    *args : Additional arguments

    Returns:
    node : ndarray
        Node coordinates of the tetrahedral mesh.
    elem : ndarray
        Element list of the tetrahedral mesh. The last column is the region id.
    face : ndarray
        Mesh surface element list of the tetrahedral mesh. The last column denotes the boundary ID.
    """

    print("Creating surface and tetrahedral mesh from a polyhedral surface ...")

    exesuff = fallbackexeext(getexeext(), "cgalpoly")

    ang = 30
    ssize = 6
    approx = 0.5
    reratio = 3
    flags = args_to_dict(*args)

    if not isinstance(opt, dict):
        ssize = opt

    if isinstance(opt, dict) and len(opt) == 1:
        ssize = opt.get("radbound", ssize)
        ang = opt.get("angbound", ang)
        approx = opt.get("distbound", approx)
        reratio = opt.get("reratio", reratio)

    if flags.get("DoRepair", 0) == 1:
        v, f = meshcheckrepair(v, f)

    saveoff(v, f, mwpath("pre_cgalpoly.off"))
    deletemeshfile(mwpath("post_cgalpoly.mesh"))

    randseed = os.getenv("ISO2MESH_SESSION", int("623F9A9E", 16))

    cmd = (
        f'"{mcpath("cgalpoly", exesuff)}" "{mwpath("pre_cgalpoly.off")}" "{mwpath("post_cgalpoly.mesh")}" '
        f"{ang:.16f} {ssize:.16f} {approx:.16f} {reratio:.16f} {maxvol:.16f} {randseed}"
    )

    status = subprocess.call(cmd, shell=True)

    if status != 0:
        raise RuntimeError("cgalpoly command failed")

    if not os.path.exists(mwpath("post_cgalpoly.mesh")):
        raise FileNotFoundError(
            f"Output file was not found, failure occurred when running command: \n{cmd}"
        )

    node, elem, face = readmedit(mwpath("post_cgalpoly.mesh"))

    print(f"node number:\t{node.shape[0]}")
    print(f"triangles:\t{face.shape[0]}")
    print(f"tetrahedra:\t{elem.shape[0]}")
    print(f"regions:\t{len(np.unique(elem[:, -1]))}")
    print("Surface and volume meshes complete")

    return node, elem, face


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
    tetgen_path = mcpath("tetgen", exesuff)

    command = f'"{tetgen_path}" -d "{os.path.join(tmppath, "post_vmesh.poly")}"'
    status, str_output = subprocess.getstatusoutput(command)

    eid = []
    if status == 0:
        ids = re.findall(r" #([0-9]+) ", str_output)
        eid = [int(id[0]) for id in ids]

    eid = np.unique(eid)
    return eid


def vol2restrictedtri(vol, thres, cent, brad, ang, radbound, distbound, maxnode):
    """
    Surface mesh extraction using CGAL mesher.

    Parameters:
    vol : ndarray
        3D volumetric image.
    thres : float
        Threshold for extraction.
    cent : tuple
        A 3D position (x, y, z) inside the resulting mesh.
    brad : float
        Maximum bounding sphere squared of the resulting mesh.
    ang : float
        Minimum angular constraint for triangular elements (degrees).
    radbound : float
        Maximum triangle Delaunay circle radius.
    distbound : float
        Maximum Delaunay sphere distances.
    maxnode : int
        Maximum number of surface nodes.

    Returns:
    node : ndarray
        List of 3D nodes (x, y, z) in the resulting surface.
    elem : ndarray
        Element list of the resulting mesh (3 columns of integers).
    """

    if radbound < 1:
        print(
            "You are meshing the surface with sub-pixel size. Check if opt.radbound is set correctly."
        )

    exesuff = getexeext()

    # Save the input volume in .inr format
    saveinr(vol, mwpath("pre_extract.inr"))

    # Delete previous output mesh file if exists
    deletemeshfile(mwpath("post_extract.off"))

    # Random seed
    randseed = os.getenv("ISO2MESH_SESSION", int("623F9A9E", 16))

    initnum = os.getenv("ISO2MESH_INITSIZE", 50)

    # Build the system command to run CGAL mesher
    cmd = (
        f'"{mcpath("cgalsurf", exesuff)}" "{mwpath("pre_extract.inr")}" '
        f"{thres:.16f} {cent[0]:.16f} {cent[1]:.16f} {cent[2]:.16f} {brad:.16f} {ang:.16f} {radbound:.16f} "
        f'{distbound:.16f} {maxnode} "{mwpath("post_extract.off")}" {randseed} {initnum}'
    )

    # Execute the system command
    status = subprocess.call(cmd, shell=True)
    if status != 0:
        raise RuntimeError(f"CGAL mesher failed with command: {cmd}")

    # Read the resulting mesh
    node, elem = readoff(mwpath("post_extract.off"))
    # Check and repair mesh if needed
    node, elem = meshcheckrepair(node, elem)

    # Assuming the origin [0, 0, 0] is located at the lower-bottom corner of the image
    node += 0.5

    return node, elem


def fillsurf(node, face):
    """
    Calculate the enclosed volume for a closed surface mesh.

    Args:
        node: Node coordinates (nn, 3).
        face: Surface triangle list (ne, 3).

    Returns:
        no: Node coordinates of the filled volume mesh.
        el: Element list (tetrahedral elements) of the filled volume mesh.
    """

    # Placeholder for calling an external function, typically using TetGen for surface to volume mesh conversion
    no, el, _ = surf2mesh(node, face, None, None, 1, 1, None, None, 0, "tetgen", "-YY")

    return no, el


def outersurf(node, face):
    """
    Extract the outer-most shell of a complex surface mesh.

    Parameters:
    node: Node coordinates
    face: Surface triangle list

    Returns:
    outface: The outer-most shell of the surface mesh
    """

    face = face[:, :3]  # Limit face to first 3 columns
    ed = surfedge(face)[0]  # Find surface edges

    # If surface is open, raise an error
    if ed.size != 0:
        raise ValueError(
            "Open surface detected, close it first. Consider meshcheckrepair() with meshfix option."
        )

    # Fill the surface and extract the volume's outer surface
    no, el = fillsurf(node, face)
    outface = volface(el)

    # Remove isolated nodes
    no, outface = removeisolatednode(no, outface)

    # Check matching of node coordinates
    maxfacenode = np.max(outface)
    I, J = ismember_rows(np.round(no[:maxfacenode, :] * 1e10), np.round(node * 1e10))

    # Map faces to the original node set
    outface = J[outface]

    # Remove faces with unmapped (zero-indexed) nodes
    ii, jj = np.where(outface == 0)
    outface = np.delete(outface, ii, axis=0)

    return outface


def surfvolume(node, face, option=None):
    """
    Calculate the enclosed volume for a closed surface.

    Parameters:
    node: Node coordinates
    face: Surface triangle list
    option: (Optional) additional option, currently unused

    Returns:
    vol: Total volume of the enclosed space
    """

    face = face[:, :3]  # Limit face to first 3 columns
    ed = surfedge(face)[0]  # Detect surface edges

    # If surface is open, raise an error
    if ed.size != 0:
        raise ValueError(
            "Open surface detected, you need to close it first. Consider meshcheckrepair() with the meshfix option."
        )

    # Fill the surface and calculate the volume of enclosed elements
    no, el = fillsurf(node, face)
    vol = elemvolume(no, el)

    # Sum the volume of all elements
    vol = np.sum(vol)

    return vol


def insurface(node, face, points):
    """
    Test if a set of 3D points is located inside a 3D triangular surface.

    Parameters:
    node: Node coordinates (Nx3 array)
    face: Surface triangle list (Mx3 array)
    points: A set of 3D points to test (Px3 array)

    Returns:
    tf: A binary vector of length equal to the number of points.
        1 indicates the point is inside the surface, and 0 indicates outside.
    """

    from scipy.spatial import Delaunay

    # Fill the surface and get nodes and elements
    no, el = fillsurf(node, face)

    # Check if points are inside the surface using Delaunay triangulation
    tri = Delaunay(no)
    tf = tri.find_simplex(points) >= 0

    # Set points inside the surface to 1, and outside to 0
    tf = tf.astype(int)

    return tf
