"""@package docstring
Iso2Mesh for Python - File I/O module

Copyright (c) 2024 Qianqian Fang <q.fang at neu.edu>
"""


__all__ = [
    "saveinr",
    "saveoff",
    "saveasc",
    "savestl",
    "savebinstl",
    "readmedit",
    "readtetgen",
    "savesurfpoly",
    "readoff",
    "savetetgennode",
    "savetetgenele",
    "readtetgen",
    "savegts",
    "readgts",
    "loadjmesh",
    "savejmesh",
    "readinr",
    "readmptiff",
    "readnirfast",
    "savenirfast",
    "readobjmesh",
    "readsmf",
    "saveabaqus",
    "savemphtxt",
    "savemsh",
    "savesmf",
    "savevrml",
]

##====================================================================================
## dependent libraries
##====================================================================================

import numpy as np
import struct
from datetime import datetime
import re
from iso2mesh.trait import (
    meshreorient,
    surfedge,
    extractloops,
    volface,
    surfplane,
    bbxflatsegment,
    internalpoint,
    uniqedges,
)

##====================================================================================
## implementations
##====================================================================================


def saveinr(vol, fname):
    """
    Save a 3D volume to INR format.

    Parameters:
    vol : ndarray
        Input, a binary volume.
    fname : str
        Output file name.
    """

    # Open file for writing in binary mode
    try:
        fid = open(fname, "wb")
    except PermissionError:
        raise PermissionError("You do not have permission to save mesh files.")

    # Determine the data type and bit length of the volume
    dtype = vol.dtype.name
    if vol.dtype == np.bool_ or dtype == "uint8":
        btype = "unsigned fixed"
        dtype = "uint8"
        bitlen = 8
    elif dtype == "uint16":
        btype = "unsigned fixed"
        bitlen = 16
    elif dtype == "float32":
        btype = "float"
        bitlen = 32
    elif dtype == "float64":
        btype = "float"
        bitlen = 64
    else:
        raise ValueError("Volume format not supported")

    # Prepare the INR header
    header = (
        f"#INRIMAGE-4#{{\nXDIM={vol.shape[0]}\nYDIM={vol.shape[1]}\nZDIM={vol.shape[2]}\n"
        f"VDIM=1\nTYPE={btype}\nPIXSIZE={bitlen} bits\nCPU=decm\nVX=1\nVY=1\nVZ=1\n"
    )
    # Ensure the header has the required 256 bytes length
    header = header + "\n" * (256 - len(header) - 4) + "##}\n"

    # Write the header and the volume data to the file
    fid.write(header.encode("ascii"))
    fid.write(np.transpose(vol, [2, 1, 0]).astype(dtype).tobytes())

    # Close the file
    fid.close()


# _________________________________________________________________________________________________________


def saveoff(v, f, fname):
    """
    saveoff(v, f, fname)

    save a surface mesh to Geomview Object File Format (OFF)

    author: Qianqian Fang, <q.fang at neu.edu>
    date: 2007/03/28

    input:
         v: input, surface node list, dimension (nn,3)
         f: input, surface face element list, dimension (be,3)
         fname: output file name
    """
    f = f - 1
    try:
        with open(fname, "wt") as fid:
            fid.write("OFF\n")
            fid.write(f"{len(v)}\t{len(f)}\t0\n")
            for vertex in v:
                fid.write(f"{vertex[0]:.16f}\t{vertex[1]:.16f}\t{vertex[2]:.16f}\n")
            face = np.hstack((f.shape[1] * np.ones([f.shape[0], 1]), f)).astype(int)
            np.savetxt(fid, face, fmt="%d", delimiter="\t")
    except IOError:
        raise PermissionError("You do not have permission to save mesh files.")


# _________________________________________________________________________________________________________


def saveasc(v, f, fname):
    """
    Save a surface mesh to FreeSurfer ASC mesh format.

    Parameters:
    v : ndarray
        Surface node list, dimension (nn, 3), where nn is the number of nodes.
    f : ndarray
        Surface face element list, dimension (be, 3), where be is the number of faces.
    fname : str
        Output file name.
    """

    try:
        with open(fname, "wt") as fid:
            fid.write(f"#!ascii raw data file {fname}\n")
            fid.write(f"{len(v)} {len(f)}\n")

            # Write vertices
            for vertex in v:
                fid.write(f"{vertex[0]:.16f} {vertex[1]:.16f} {vertex[2]:.16f} 0\n")

            # Write faces (subtract 1 to adjust from MATLAB 1-based indexing to Python 0-based)
            for face in f:
                fid.write(f"{face[0] - 1} {face[1] - 1} {face[2] - 1} 0\n")

    except PermissionError:
        raise PermissionError("You do not have permission to save mesh files.")


def savestl(node, elem, fname, solidname=""):
    """
    Save a tetrahedral mesh to an STL (Standard Tessellation Language) file.

    Parameters:
    node : ndarray
        Surface node list, dimension (N, 3).
    elem : ndarray
        Tetrahedral element list; if size is (N, 3), it's a surface mesh.
    fname : str
        Output file name.
    solidname : str, optional
        Name of the object in the STL file.
    """

    if len(node) == 0 or node.shape[1] < 3:
        raise ValueError("Invalid node input")

    if elem is not None and elem.shape[1] >= 5:
        elem = elem[:, :4]  # Discard extra columns if necessary

    with open(fname, "wt") as fid:
        fid.write(f"solid {solidname}\n")

        if elem is not None:
            if elem.shape[1] == 4:
                elem = volface(elem)  # Convert tetrahedra to surface triangles

            ev = surfplane(node, elem)  # Calculate the plane normals
            ev = (
                ev[:, :3] / np.linalg.norm(ev[:, :3], axis=1)[:, np.newaxis]
            )  # Normalize normals

            for i in range(elem.shape[0]):
                facet_normal = ev[i, :]
                vertices = node[elem[i, :3], :]
                fid.write(
                    f"facet normal {facet_normal[0]:e} {facet_normal[1]:e} {facet_normal[2]:e}\n"
                )
                fid.write("  outer loop\n")
                for vertex in vertices:
                    fid.write(f"    vertex {vertex[0]:e} {vertex[1]:e} {vertex[2]:e}\n")
                fid.write("  endloop\nendfacet\n")

        fid.write(f"endsolid {solidname}\n")


def savebinstl(node, elem, fname, solidname=""):
    """
    Save a tetrahedral mesh to a binary STL (Standard Tessellation Language) file.

    Parameters:
    node : ndarray
        Surface node list, dimension (N, 3).
    elem : ndarray
        Tetrahedral element list; if size(elem,2)==3, it is a surface.
    fname : str
        Output file name.
    solidname : str, optional
        An optional string for the name of the object.
    """

    if len(node) == 0 or node.shape[1] < 3:
        raise ValueError("Invalid node input")

    if elem is not None and elem.shape[1] >= 5:
        elem = elem[:, :4]  # Remove extra columns if needed

    # Open the file in binary write mode
    with open(fname, "wb") as fid:
        # Header structure containing metadata
        header = {
            "Ver": 1,
            "Creator": "iso2mesh",
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        if solidname:
            header["name"] = solidname

        headerstr = str(header).replace("\t", "").replace("\n", "").replace("\r", "")
        headerstr = headerstr[:80] if len(headerstr) > 80 else headerstr.ljust(80, "\0")
        fid.write(headerstr.encode("ascii"))

        if elem is not None:
            if elem.shape[1] == 4:
                elem = meshreorient(node, elem)
                elem = volface(elem)  # Convert tetrahedra to triangular faces

            # Compute surface normals
            ev = surfplane(node, elem)
            ev = ev[:, :3] / np.linalg.norm(ev[:, :3], axis=1, keepdims=True)

            # Write number of facets
            num_facets = len(elem)
            fid.write(struct.pack("<I", num_facets))

            # Write each facet
            for i in range(num_facets):
                # Normal vector
                fid.write(struct.pack("<3f", *ev[i, :]))
                # Vertices of the triangle
                for j in range(3):
                    fid.write(struct.pack("<3f", *node[elem[i, j], :]))
                # Attribute byte count (set to 0)
                fid.write(struct.pack("<H", 0))


def readmedit(filename):
    """
    Read a Medit mesh format file.

    Parameters:
    filename : str
        Name of the Medit data file.

    Returns:
    node : ndarray
        Node coordinates of the mesh.
    elem : ndarray
        List of elements of the mesh (tetrahedra).
    face : ndarray
        List of surface triangles of the mesh.
    """

    node = []
    elem = []
    face = []
    val = 0

    with open(filename, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            key = line.strip()
            if key == "End":
                break

            if key == "Vertices":
                val = int(fid.readline().strip())
                node_data = np.fromfile(fid, dtype=np.float32, count=4 * val, sep=" ")
                node = node_data.reshape((val, 4))

            elif key == "Triangles":
                val = int(fid.readline().strip())
                face_data = np.fromfile(fid, dtype=np.int32, count=4 * val, sep=" ")
                face = face_data.reshape((val, 4))

            elif key == "Tetrahedra":
                val = int(fid.readline().strip())
                elem_data = np.fromfile(fid, dtype=np.int32, count=5 * val, sep=" ")
                elem = elem_data.reshape((val, 5))

    return node, elem, face


# _________________________________________________________________________________________________________


def readtetgen(fstub):
    """
    [node, elem, face] = readtetgen(fstub)

    read tetgen output files

    input:
        fstub: file name stub

    output:
        node: node coordinates of the tetgen mesh
        elem: tetrahedra element list of the tetgen mesh
        face: surface triangles of the tetgen mesh

    -- this function is part of iso2mesh toolbox (http://iso2mesh.sf.net)


    read node file
    """
    try:
        node = np.loadtxt(f"{fstub}.node", skiprows=1)
        node = node[:, 1:4]
    except FileNotFoundError:
        raise FileNotFoundError("node file is missing!")

    # read element file
    try:
        elem = np.loadtxt(f"{fstub}.ele", skiprows=1, dtype=int)
        elem = elem[:, 1:]
        elem[:, :4] += 1
    except FileNotFoundError:
        raise FileNotFoundError("elem file is missing!")

    # read surface mesh file
    try:
        face = np.loadtxt(f"{fstub}.face", skiprows=1, dtype=int)
        face = face[:, 1:]
        face[:, :3] += 1
    except FileNotFoundError:
        raise FileNotFoundError("surface data file is missing!")

    elem[:, :4], evol, idx = meshreorient(node[:, :3], elem[:, :4])

    return node, elem, face


# _________________________________________________________________________________________________________


def savesurfpoly(v, f, holelist, regionlist, p0, p1, fname, forcebox=None):
    """
    Saves a set of surfaces into poly format for TetGen.

    Args:
        v (numpy array): Surface node list, shape (nn, 3) or (nn, 4)
        f (numpy array or list): Surface face elements, shape (be, 3)
        holelist (numpy array): List of holes, each hole as an internal point
        regionlist (numpy array): List of regions, similar to holelist
        p0 (numpy array): One end of the bounding box coordinates
        p1 (numpy array): Other end of the bounding box coordinates
        fname (str): Output file name
        forcebox (numpy array, optional): Specifies max-edge size at box corners

    This function is part of the iso2mesh toolbox.
    """
    dobbx = 0
    if forcebox != None:
        dobbx = any([forcebox])

    faceid = (
        f[:, 3]
        if not isinstance(f, list) and len(f.shape) > 1 and f.shape[1] == 4
        else None
    )
    f = (
        f[:, :3]
        if not isinstance(f, list) and len(f.shape) > 1 and f.shape[1] == 4
        else f
    )

    # Check and process node sizes if v has 4 columns
    nodesize = (
        v[:, 3]
        if not isinstance(v, list) and len(v.shape) > 1 and v.shape[1] == 4
        else None
    )
    v = (
        v[:, :3]
        if not isinstance(v, list) and len(v.shape) > 1 and v.shape[1] == 4
        else v
    )
    if p0 is None or len(p0) > 0 and v.size > 0:
        p0 = np.min(v, axis=0)
    if p1 is None or len(p0) > 0 and v.size > 0:
        p1 = np.max(v, axis=0)

    # Handle edges
    edges = surfedge(f)[0] if not isinstance(f, list) else []

    node = v
    bbxnum, loopvert, loopid, loopnum = 0, [], [], 1

    if len(edges) > 0:
        loops = extractloops(edges)
        if len(loops) < 3:
            raise ValueError("Degenerated loops detected")
        seg = [0] + list(np.where(np.isnan(loops))[0].tolist())
        segnum = len(seg) - 1
        newloops = []
        for i in range(segnum):
            if seg[i + 1] - (seg[i] + 1) == 0:
                continue
            oneloop = loops[seg[i] + 1 : seg[i + 1] - 1]
            if oneloop[0] == oneloop[-1]:
                oneloop = oneloop[:-1]
            newloops.extend([np.nan] + bbxflatsegment(node, oneloop))
        loops = newloops + [np.nan]

        seg = [0] + list(np.where(np.isnan(loops))[0].tolist())
        segnum = len(seg) - 1
        bbxnum = 6
        loopcount = np.zeros(bbxnum, dtype=np.int32)
        loopid = np.zeros(segnum, dtype=np.int32)
        for i in range(segnum):  # walk through the edge loops
            subloop = loops[seg[i] + 1 : seg[i + 1] - 1]
            if not subloop:
                continue
            loopvert.append(subloop)
            loopnum += 1
            boxfacet = np.where(np.sum(np.abs(np.diff(v[subloop, :])), axis=1) < 1e-8)[
                0
            ]  # find a flat loop
            if len(boxfacet) == 1:  # if the loop is flat along x/y/z dir
                bf = boxfacet[0]  # no degeneracy allowed
                if np.sum(np.abs(v[subloop[0], bf] - p0[bf])) < 1e-2:
                    loopcount[bf] += 1
                    v[subloop, bf] = p0[bf]
                    loopid[i] = bf
                elif np.sum(np.abs(v[subloop[0], bf] - p1[bf])) < 1e-2:
                    loopcount[bf + 3] += 1
                    v[subloop, bf] = p1[bf]
                    loopid[i] = bf + 3

    if dobbx and len(edges) == 0:
        bbxnum = 6
        loopcount = np.zeros(bbxnum, dtype=np.int32)

    if dobbx or len(edges) > 0:
        nn = v.shape[0]
        boxnode = np.array(
            [
                p0,
                [p1[0], p0[1], p0[2]],
                [p1[0], p1[1], p0[2]],
                [p0[0], p1[1], p0[2]],
                [p0[0], p0[1], p1[2]],
                [p1[0], p0[1], p1[2]],
                [p1[0], p1[1], p1[2]],
                [p0[0], p1[1], p1[2]],
            ]
        )
        boxelem = np.array(
            [
                [4, nn, nn + 3, nn + 7, nn + 4],  # x=xmin
                [4, nn, nn + 1, nn + 5, nn + 4],  # y=ymin
                [4, nn, nn + 1, nn + 2, nn + 3],  # z=zmin
                [4, nn + 1, nn + 2, nn + 6, nn + 5],  # x=xmax
                [4, nn + 2, nn + 3, nn + 7, nn + 6],  # y=ymax
                [4, nn + 4, nn + 5, nn + 6, nn + 7],  # z=zmax
            ]
        )

        node = np.vstack((v, boxnode)) if v.size > 0 else boxnode

    node = np.hstack((np.arange(node.shape[0])[:, np.newaxis], node))

    with open(fname, "wt") as fp:
        fp.write("#node list\n{} 3 0 0\n".format(len(node)))
        np.savetxt(fp, node, fmt="%d %.16f %.16f %.16f")

        if not isinstance(f, list):
            fp.write("#facet list\n{} 1\n".format(len(f) + bbxnum + len(loopvert)))
            elem = (
                np.hstack((3 * np.ones((len(f), 1)), f - 1))
                if f.size > 1
                else np.array([])
            ).astype("int")
            if elem.size > 0:
                if faceid is not None and len(faceid) == elem.shape[0]:
                    elemdata = np.hstack((faceid.reshape(-1, 1), elem[:, :4]))
                    np.savetxt(fp, elemdata, fmt="1 0 %d\n%d %d %d %d")
                else:
                    np.savetxt(fp, elem[:, :4], fmt="1 0\n%d %d %d %d")

            if loopvert:
                for i in range(len(loopvert)):  # walk through the edge loops
                    subloop = loopvert[i] - 1
                    fp.write("1 0 {}\n{}".format(i, len(subloop)))
                    fp.write("\t{}".format("\t".join(map(str, subloop))))
                    fp.write("\n")
        else:  # if the surface is recorded as a cell array
            totalplc = 0
            for i in range(len(f)):
                if len(f[i]) == 0:
                    continue
                if isinstance(f[i][0], list):
                    totalplc += len(f[i][0])
                else:
                    try:
                        dim = np.array(f[i]).shape
                        if len(dim) == 1:
                            totalplc += 1
                        else:
                            totalplc += dim[0]
                    except:
                        totalplc += 1
            fp.write("#facet list\n{} 1\n".format(totalplc + bbxnum))
            for i in range(len(f)):
                plcs = f[i]
                faceid = -1
                if (
                    isinstance(plcs, list)
                    and len(plcs) > 0
                    and isinstance(plcs[0], list)
                ):  # if each face is a cell, use plc{2} for face id
                    if len(plcs) > 1:
                        faceid = int(plcs[1][0])
                    plcs = plcs[0]
                elif isinstance(plcs, list):
                    if all(isinstance(el, list) for el in plcs):
                        plcs = [np.array(el) for el in plcs]
                    else:
                        plcs = [np.array(plcs)]
                for row in range(len(plcs)):
                    plc = np.array(plcs[row])
                    if np.any(
                        np.isnan(plc)
                    ):  # we use nan to separate outer contours and holes
                        holeid = np.where(np.isnan(plc))[0]
                        plc = np.array(plc, dtype=np.int32)
                        if faceid > 0:
                            fp.write(
                                "{} {} {}\n{}".format(
                                    len(holeid) + 1, len(holeid), faceid, holeid[0]
                                )
                            )
                        else:
                            fp.write(
                                "{} {}\n{}".format(
                                    len(holeid) + 1, len(holeid), holeid[0]
                                )
                            )
                        fp.write(
                            "\t{}".format("\t".join(map(str, plc[: holeid[0]] - 1)))
                        )
                        fp.write("\t1\n")
                        for j in range(len(holeid)):
                            if j == len(holeid) - 1:
                                fp.write(
                                    "{}\t{}".format(
                                        len(plc[holeid[j] + 1 :]),
                                        "\t".join(map(str, plc[holeid[j] + 1 :] - 1)),
                                    )
                                )
                            else:
                                fp.write(
                                    "{}\t{}".format(
                                        len(plc[holeid[j] + 1 : holeid[j + 1] - 1]),
                                        "\t".join(
                                            map(
                                                str,
                                                plc[holeid[j] + 1 : holeid[j + 1] - 1]
                                                - 1,
                                            )
                                        ),
                                    )
                                )
                            fp.write("\t1\n")
                        for j in range(len(holeid)):
                            if j == len(holeid) - 1:
                                fp.write(
                                    "{} {}\n".format(
                                        j + 1,
                                        "".join(
                                            "{:.16f} ".format(x)
                                            for x in np.mean(
                                                node[plc[holeid[j] + 1 :] - 1, 1:4],
                                                axis=0,
                                            )
                                        ),
                                    )
                                )
                            else:
                                fp.write(
                                    "{} {}\n".format(
                                        j + 1,
                                        "".join(
                                            "{:.16f} ".format(x)
                                            for x in np.mean(
                                                node[
                                                    plc[
                                                        holeid[j]
                                                        + 1 : holeid[j + 1]
                                                        - 1
                                                    ]
                                                    - 1,
                                                    1:4,
                                                ],
                                                axis=0,
                                            )
                                        ),
                                    )
                                )
                    else:
                        if faceid > 0:
                            fp.write("1 0 {}\n{}".format(faceid, len(plc)))
                        else:
                            fp.write("1 0\n{}".format(len(plc)))
                        fp.write("\t{}".format("\t".join(map(str, plc - 1))))
                        fp.write("\t1\n")

        if dobbx or isinstance(edges, np.ndarray) and edges.size > 0:
            for i in range(bbxnum):
                fp.write("{} {} 1\n".format(1 + loopcount[i], loopcount[i]))
                fp.write("{} {} {} {} {}\n".format(*boxelem[i, :]))
                if (
                    isinstance(edges, np.ndarray)
                    and edges.size > 0
                    and loopcount[i]
                    and np.any(loopid == i)
                ):
                    endid = np.where(loopid == i)[0]
                    for k in endid:
                        j = endid[k]
                        subloop = loops[seg[j] + 1 : seg[j + 1] - 1]
                        fp.write("{} ".format(len(subloop)))
                        fp.write("{} ".format(" ".join(map(str, subloop - 1))))
                        fp.write("\n")
                    for k in endid:
                        j = endid[k]
                        subloop = loops[seg[j] + 1 : seg[j + 1] - 1]
                        fp.write(
                            "{} {:.16f} {:.16f} {:.16f}\n".format(
                                k, internalpoint(v, subloop)
                            )
                        )

        holelist = np.array(holelist)
        regionlist = np.array(regionlist)

        if all(holelist.shape):
            fp.write("#hole list\n{}\n".format(holelist.shape[0]))
            for i in range(holelist.shape[0]):
                fp.write("{} {:.16f} {:.16f} {:.16f}\n".format(i + 1, *holelist[i, :]))
        else:
            fp.write("#hole list\n0\n")

        if regionlist.ndim == 1 and len(regionlist) > 0:
            regionlist = regionlist[:, np.newaxis].T

        if regionlist.shape[0]:
            fp.write("#region list\n{}\n".format(regionlist.shape[0]))
            if regionlist.shape[1] == 3:
                for i in range(regionlist.shape[0]):
                    fp.write(
                        "{} {:.16f} {:.16f} {:.16f} {}\n".format(
                            i + 1, *regionlist[i, :], i + 1
                        )
                    )
            elif regionlist.shape[1] == 4:
                for i in range(regionlist.shape[0]):
                    fp.write(
                        "{} {:.16f} {:.16f} {:.16f} {} {:.16f}\n".format(
                            i + 1, *regionlist[i, :3], i + 1, regionlist[i, 3]
                        )
                    )

        if nodesize:
            if len(nodesize) + len(forcebox) == node.shape[0]:
                nodesize = np.concatenate((nodesize, forcebox))
            with open(fname.replace(".poly", ".mtr"), "wt") as fid:
                fid.write("{} 1\n".format(len(nodesize)))
                np.savetxt(fid, nodesize, fmt="%.16f")


# _________________________________________________________________________________________________________


def readoff(fname):
    """
    Read Geomview Object File Format (OFF)

    Parameters:
        fname: name of the OFF data file

    Returns:
        node: node coordinates of the mesh
        elem: list of elements of the mesh
    """
    node = []
    elem = []

    with open(fname, "rb") as fid:
        while True:
            line = fid.readline().decode("utf-8").strip()
            dim = re.search("[0-9.]+ [0-9.]+ [0-9.]+", line)
            if dim:
                dim = np.fromstring(dim.group(), sep=" ", dtype=int)
                break

        line = nonemptyline(fid)

        nodalcount = 3
        if line:
            val = np.fromstring(line, sep=" ", count=-1, dtype=float)
            nodalcount = len(val)
        else:
            return node, elem

        node = np.fromfile(
            fid, dtype=float, sep=" ", count=(nodalcount * (dim[0] - 1))
        ).reshape(-1, nodalcount)
        node = np.vstack((val, node))

        line = nonemptyline(fid)
        facetcount = 4
        if line:
            val = np.fromstring(line, sep=" ", count=-1, dtype=float)
            facetcount = len(val)
        else:
            return node, elem
        elem = np.fromfile(
            fid, dtype=float, sep=" ", count=(facetcount * (dim[1] - 1))
        ).reshape(-1, facetcount)
        elem = np.vstack((val, elem))

    elem = elem[:, 1:]

    if elem.shape[1] <= 3:
        elem[:, :3] = np.round(elem[:, :3])
    else:
        elem[:, :4] = np.round(elem[:, :4])

    elem = elem.astype(int) + 1

    return node, elem


def nonemptyline(fid):
    str_ = ""
    if fid == 0:
        raise ValueError("invalid file")

    while (not re.search(r"\S", str_) or re.search(r"^#", str_)) and not fid.closed:
        str_ = fid.readline().decode("utf-8").strip()
        if not isinstance(str_, str):
            str_ = ""
            return str_

    return str_


def savetetgennode(node, fname):
    """
    savetetgennode(node, fname)

    Save a mesh node list to TetGen .node format

    Parameters:
        node : ndarray
            Node coordinates, shape (N, 3) or (N, >3).
            Columns beyond the 3rd are treated as markers or attributes.
        fname : str
            Output filename for the .node file

    This function writes TetGen-compatible node files from the given mesh data.

    Author:
        Qianqian Fang <q.fang at neu.edu>
    """

    nnode, ncol = node.shape
    hasprop = max(ncol - 4, 0)  # Number of attributes
    hasmarker = int(ncol >= 4)

    # First line header: <# of points> <dimension> <# of attributes> <# of boundary markers>
    header = f"{nnode} 3 {hasprop} {hasmarker}\n"

    # Index vector (0-based)
    idx = np.arange(nnode).reshape(-1, 1)

    # Split columns
    coords = node[:, :3]
    attributes = node[:, 3 : 3 + hasprop] if hasprop > 0 else np.empty((nnode, 0))
    markers = (
        node[:, 3 + hasprop : 3 + hasprop + 1] if hasmarker else np.empty((nnode, 0))
    )

    # Concatenate all columns
    full_data = np.hstack([idx, coords, attributes, markers])

    # Define format string
    fmt = ["%d", "%e", "%e", "%e"] + ["%e"] * hasprop + (["%d"] if hasmarker else [])
    fmt_str = " ".join(fmt)

    # Write to file
    try:
        with open(fname, "w") as f:
            f.write(header)
            np.savetxt(f, full_data, fmt=fmt_str)
    except IOError:
        raise IOError(f"Cannot write to file {fname}")


def savetetgenele(elem, fname):
    """
    savetetgenele(elem, fname)

    Save a mesh tetrahedral element list to TetGen .ele format

    Parameters:
        elem : ndarray
            Element connectivity array, shape (N, 4) or (N, >4).
            Columns beyond the 4th are treated as attributes or markers.
        fname : str
            Output filename for the .ele file

    This function writes TetGen-compatible element (.ele) files from mesh data.

    Author:
        Qianqian Fang <q.fang at neu.edu>
    """

    nelem, ncol = elem.shape
    hasprop = max(ncol - 5, 0)  # number of attributes
    hasmarker = int(ncol >= 5)

    # First line header: <# of tetrahedra> <nodes per element> <# of attributes>
    header = f"{nelem} 4 {hasprop + hasmarker}\n"

    # TetGen uses 0-based indexing; adjust indices
    elem = elem.copy()
    elem[:, :4] -= 1

    # Create index column
    idx = np.arange(nelem).reshape(-1, 1)

    # Extract node indices, attributes, and marker if present
    nodes = elem[:, :4]
    attributes = elem[:, 4 : 4 + hasprop] if hasprop > 0 else np.empty((nelem, 0))
    markers = elem[:, 4 + hasprop : 5 + hasprop] if hasmarker else np.empty((nelem, 0))

    # Combine all columns
    full_data = np.hstack([idx, nodes, attributes, markers])

    # Define format string
    fmt = (
        ["%d", "%d", "%d", "%d", "%d"]
        + ["%e"] * hasprop
        + (["%d"] if hasmarker else [])
    )
    fmt_str = " ".join(fmt)

    # Write to file
    try:
        with open(fname, "w") as f:
            f.write(header)
            np.savetxt(f, full_data, fmt=fmt_str)
    except IOError:
        raise IOError(f"Cannot write to file {fname}")


def savegts(v, f, fname, edges=None):
    """
    savegts(v, f, fname, edges=None)

    Save a surface mesh to GNU Triangulated Surface Format (GTS)

    Parameters:
        v : ndarray
            Surface node list, shape (N, 3)
        f : ndarray
            Surface face list, shape (M, 3)
        fname : str
            Output file name
        edges : ndarray (optional)
            Precomputed edge list. If None, will be computed automatically.

    Returns:
        nedge : int
            Number of unique edges in the mesh

    Author:
        Qianqian Fang, <q.fang at neu.edu>
    """
    v = v[:, :3]
    f = f[:, :3]

    if edges is None:
        edges, _, edgemap = uniqedges(f)
    else:
        # Assume edgemap is correct if edges are provided
        raise NotImplementedError(
            "Precomputed edges not supported in this simplified version"
        )

    nedge = edges.shape[0]

    with open(fname, "w") as fid:
        fid.write(f"{v.shape[0]} {nedge} {f.shape[0]}\n")
        np.savetxt(fid, v, fmt="%.16f %.16f %.16f")
        np.savetxt(fid, edges, fmt="%d %d")
        np.savetxt(fid, edgemap, fmt="%d %d %d")

    return nedge


def readgts(fname):
    """
    readgts(fname)

    Read a GNU Triangulated Surface (.gts) file

    Parameters:
        fname : str
            Name of the GTS file

    Returns:
        node : ndarray
            Node coordinates (N, 3)
        elem : ndarray
            Face list (M, 3)
        edges : ndarray
            Edge list (E, 2)
        edgemap : ndarray
            Mapping of faces to edges (M, 3)

    Author:
        Qianqian Fang, <q.fang at neu.edu>
    """
    with open(fname, "r") as fid:
        header = fid.readline().strip().split()
        nv, ne, nf = map(int, header)

        node = np.loadtxt(fid, max_rows=nv).reshape((nv, 3))
        edges = np.loadtxt(fid, max_rows=ne, dtype=int).reshape((ne, 2))
        edgemap = np.loadtxt(fid, max_rows=nf, dtype=int).reshape((nf, 3))

    # Reconstruct element connectivity from edge map
    elem = np.zeros((nf, 3), dtype=int)
    edgetable = edges.T
    try:
        for i in range(nf):
            edge_indices = edgemap[i] - 1  # convert to 0-based indexing
            verts = np.concatenate(
                [
                    edgetable[:, edge_indices[0]],
                    edgetable[:, edge_indices[1]],
                    edgetable[:, edge_indices[2]],
                ]
            )
            elem[i, :3] = np.unique(verts)[:3]
    except Exception as e:
        raise ValueError(f"Invalid GTS face at index {i}") from e

    return node, elem, edges, edgemap


# _________________________________________________________________________________________________________


def loadjmesh(filename, **kwargs):
    """
    Load a JMesh format file (.jmsh or .bmsh).

    Parameters:
        filename : str
            Input file name. Use .bmsh for binary, .jmsh for text format.
        **kwargs : dict
            Additional options passed to loadjson/loadbj.

    Returns:
        jmsh : dict
            Mesh structure containing JMesh data.

    Raises:
        ValueError: If file suffix is not .jmsh or .bmsh
        ImportError: If jdata package is not installed

    Example:
        >>> newmesh = loadjmesh('box.jmsh')
    """
    try:
        from jdata import load as jdload
    except ImportError:
        raise ImportError("You must install jdata package: pip install jdata")

    if filename.endswith(".jmsh"):
        return jdload(filename, **kwargs)
    elif filename.endswith(".bmsh"):
        return jdload(filename, **kwargs)
    else:
        raise ValueError(
            "File suffix must be .jmsh for text JMesh or .bmsh for binary JMesh"
        )


# _________________________________________________________________________________________________________


def savejmesh(node, face=None, elem=None, fname=None, **kwargs):
    """
    Save a mesh to JMesh format (.jmsh or .bmsh).

    Parameters:
        node : ndarray
            Node list, shape (nn, 3) or more columns.
        face : ndarray, optional
            Surface face element list, shape (nf, 3) or more.
        elem : ndarray, optional
            Tetrahedral element list, shape (ne, 4) or more.
        fname : str
            Output file name (.jmsh for text, .bmsh for binary).
        **kwargs : dict
            Options including:
            - Dimension: 2 or 3 (default: auto from node shape)
            - Author: string
            - MeshTitle: string
            - MeshTag: value
            - Comment: string
            - Flexible: 0 (dimension-specific) or 1 (generic containers)
            - Header: 1 (include metadata) or 0

    Example:
        >>> savejmesh(node, face, elem, 'mesh.jmsh', Dimension=3)
    """
    try:
        from jdata import save as jdsave
    except ImportError:
        raise ImportError("You must install jdata package: pip install jdata")

    # Handle variable argument forms
    if fname is None:
        if face is None:
            raise ValueError("Must provide at least node and filename")
        fname = face
        face = None
        elem = None
    elif elem is None and isinstance(face, str):
        fname = face
        face = None

    from datetime import datetime

    meshdim = kwargs.get("Dimension", node.shape[1] if node.ndim > 1 else 3)
    mesh = {}

    # Add metadata header
    if kwargs.get("Header", 1) == 1:
        metadata = {
            "JMeshVersion": 0.5,
            "Dimension": meshdim,
            "CreationTime": datetime.now().isoformat(),
            "Comment": "Created by pyiso2mesh",
            "AnnotationFormat": "https://github.com/NeuroJSON/jmesh/blob/master/JMesh_specification.md",
        }
        if "Author" in kwargs:
            metadata["Author"] = kwargs["Author"]
        if "MeshTitle" in kwargs:
            metadata["MeshTitle"] = kwargs["MeshTitle"]
        if "MeshTag" in kwargs:
            metadata["MeshTag"] = kwargs["MeshTag"]
        if "Comment" in kwargs:
            metadata["Comment"] = kwargs["Comment"]
        mesh["_DataInfo_"] = metadata

    # Build mesh structure
    if kwargs.get("Flexible", 0) == 1:
        mesh["MeshNode"] = node
        if face is not None:
            mesh["MeshSurf"] = face
        if elem is not None:
            mesh["MeshElem"] = elem
    elif meshdim == 3:
        if node.shape[1] < 3:
            raise ValueError("Expecting 3 or more columns in node")
        mesh["MeshVertex3"] = node[:, :3]
        if node.shape[1] > 3:
            mesh["MeshVertex3"] = {
                "Data": node[:, :3],
                "Properties": {"Value": node[:, 3:]},
            }
        if face is not None:
            if face.shape[1] < 3:
                raise ValueError("Expecting 3 or more columns in face")
            mesh["MeshTri3"] = face[:, :3]
            if face.shape[1] > 3:
                mesh["MeshTri3"] = {
                    "Data": face[:, :3],
                    "Properties": {"Value": face[:, 3:]},
                }
        if elem is not None:
            if elem.shape[1] < 4:
                raise ValueError("Expecting 4 or more columns in elem")
            mesh["MeshTet4"] = elem[:, :4]
            if elem.shape[1] > 4:
                mesh["MeshTet4"] = {
                    "Data": elem[:, :4],
                    "Properties": {"Value": elem[:, 4:]},
                }
    elif meshdim == 2:
        if node.shape[1] < 2:
            raise ValueError("Expecting 2 or more columns in node")
        mesh["MeshVertex2"] = node[:, :2]
        if node.shape[1] > 2:
            mesh["MeshVertex2"] = {
                "Data": node[:, :2],
                "Properties": {"Value": node[:, 2:]},
            }
        if face is not None:
            if face.shape[1] < 3:
                raise ValueError("Expecting 3 or more columns in face")
            mesh["MeshTri3"] = face[:, :3]
            if face.shape[1] > 3:
                mesh["MeshTri3"] = {
                    "Data": face[:, :3],
                    "Properties": {"Value": face[:, 3:]},
                }
        if elem is not None:
            import warnings

            warnings.warn("elem is redundant in a 2D mesh, skipping")
    else:
        raise ValueError("Specified Dimension is not supported")

    jdsave(mesh, fname, **kwargs)


# _________________________________________________________________________________________________________


def readinr(fname):
    """
    Load a volume from an INR file.

    Parameters:
        fname : str
            Input file name.

    Returns:
        dat : ndarray
            Volume data read from the INR file.

    Raises:
        ValueError: If header format is invalid or unsupported.
    """
    with open(fname, "rb") as fid:
        header = fid.read(256).decode("ascii")

        if not header.startswith("#INRIMAGE-4"):
            raise ValueError("INRIMAGE header was not found")

        def extract(pattern, name):
            m = re.search(pattern, header)
            if m:
                return int(m.group(1))
            raise ValueError(f"No {name} found")

        nx = extract(r"XDIM\s*=\s*(\d+)", "XDIM")
        ny = extract(r"YDIM\s*=\s*(\d+)", "YDIM")
        nz = extract(r"ZDIM\s*=\s*(\d+)", "ZDIM")

        m = re.search(r"VDIM\s*=\s*(\d+)", header)
        nv = int(m.group(1)) if m else 1

        m = re.search(r"TYPE=([a-z ]+)", header)
        if not m:
            raise ValueError("No TYPE found")
        dtype_str = m.group(1).strip()

        m = re.search(r"PIXSIZE=(\d+)", header)
        if not m:
            raise ValueError("No PIXSIZE found")
        pixel = int(m.group(1))

        # Determine numpy dtype
        if dtype_str == "unsigned fixed" and pixel == 8:
            dtype = np.uint8
        elif dtype_str == "unsigned fixed" and pixel == 16:
            dtype = np.uint16
        elif dtype_str == "float" and pixel == 32:
            dtype = np.float32
        elif dtype_str == "float" and pixel == 64:
            dtype = np.float64
        else:
            raise ValueError("Volume format not supported")

        dat = np.frombuffer(fid.read(), dtype=dtype)

        if nv == 1:
            dat = dat.reshape((nx, ny, nz), order="F")
        else:
            dat = dat.reshape((nx, ny, nz, nv), order="F")

    return dat


# _________________________________________________________________________________________________________


def readmptiff(fname):
    """
    Load a volume from a multi-page TIFF file.

    Parameters:
        fname : str
            Input file name.

    Returns:
        dat : ndarray
            3D volume data (height, width, nslices).
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PIL/Pillow is required: pip install Pillow")

    img = Image.open(fname)
    slices = []
    try:
        while True:
            slices.append(np.array(img))
            img.seek(img.tell() + 1)
    except EOFError:
        pass

    if len(slices) == 0:
        raise ValueError("No data found in the TIFF")

    return np.stack(slices, axis=-1)


# _________________________________________________________________________________________________________


def readnirfast(filestub):
    """
    Load mesh files in NIRFAST format.

    Parameters:
        filestub : str
            File name stub (without extension).

    Returns:
        nirfastmesh : dict with keys:
            - nodes: node coordinates (N, 3)
            - elements: element list (M, 3 or 4)
            - bndvtx: boundary flags for each node
            - dimension: mesh dimension (2 or 3)
            - region: node labels (optional)
            - excoef: extinction coefficients (optional)
            - excoefheader: extinction coeff field names (optional)
            - type: header from .param file (optional)
            - prop: optical properties (optional)
    """
    import os

    nirfastmesh = {}

    # Read node file
    nodefile = f"{filestub}.node"
    if not os.path.exists(nodefile):
        raise FileNotFoundError(f"{nodefile} could not be found")
    data = np.loadtxt(nodefile)
    nirfastmesh["bndvtx"] = data[:, 0]
    nirfastmesh["nodes"] = data[:, 1:]

    # Read element file
    elemfile = f"{filestub}.elem"
    if not os.path.exists(elemfile):
        raise FileNotFoundError(f"{elemfile} could not be found")
    nirfastmesh["elements"] = np.loadtxt(elemfile, dtype=int)
    nirfastmesh["dimension"] = nirfastmesh["elements"].shape[1] - 1

    # Read region file (optional)
    regionfile = f"{filestub}.region"
    if os.path.exists(regionfile):
        nirfastmesh["region"] = np.loadtxt(regionfile)

    # Read extinction coefficient file (optional)
    excoeffile = f"{filestub}.excoef"
    if os.path.exists(excoeffile):
        with open(excoeffile, "r") as fid:
            textheader = []
            for line in fid:
                vals = line.strip().split()
                try:
                    data = [float(v) for v in vals]
                    if len(data) > 1:
                        # Read remaining data
                        remaining = np.loadtxt(fid)
                        if remaining.ndim == 1:
                            remaining = remaining.reshape(1, -1)
                        nirfastmesh["excoef"] = np.vstack([data, remaining])
                        nirfastmesh["excoefheader"] = textheader
                        break
                except ValueError:
                    textheader.append(line.strip())

    # Read param file (optional)
    paramfile = f"{filestub}.param"
    if os.path.exists(paramfile):
        with open(paramfile, "r") as fid:
            for i, line in enumerate(fid):
                if i == 0:
                    nirfastmesh["type"] = line.strip()
                vals = line.strip().split()
                try:
                    data = [float(v) for v in vals]
                    if len(data) > 1:
                        remaining = np.loadtxt(fid)
                        if remaining.ndim == 1:
                            remaining = remaining.reshape(1, -1)
                        nirfastmesh["prop"] = np.vstack([data, remaining])
                        break
                except ValueError:
                    pass

    return nirfastmesh


# _________________________________________________________________________________________________________


def savenirfast(v, f=None, filestub=None, nodeseg=None, nodeprop=None, proptype="stnd"):
    """
    Save a mesh to NIRFAST format.

    Parameters:
        v : ndarray or dict
            Node coordinates (N, 3+) or a NIRFAST mesh dict.
        f : ndarray or str
            Element list (M, 3 or 4), or filestub if v is a dict.
        filestub : str
            Output file stub.
        nodeseg : ndarray, optional
            Node segmentation labels.
        nodeprop : ndarray, optional
            Node properties (mua, musp, n).
        proptype : str or list, optional
            Property type header (default 'stnd').
    """
    # Handle dict input
    if isinstance(v, dict):
        filestub = f
        node = v["nodes"]
        f = v["elements"]
        proptype = v.get("type", "stnd")
        nodeseg = v.get("region")
        if "mua" in v:
            nodeprop = np.column_stack([v["mua"], v["mus"], v["ri"]])
        isboundary = v.get("bndvtx")
    else:
        node = v
        isboundary = None

    if node.shape[1] > 3:
        extra = node[:, 3:]
        node = node[:, :3]
        if nodeprop is not None:
            nodeprop = np.hstack([extra, nodeprop])
        else:
            nodeprop = extra

    if nodeseg is None:
        nodeseg = np.zeros(node.shape[0], dtype=int)

    if f.shape[1] > 4:
        f = f[:, :4]

    # Compute boundary nodes if not provided
    if isboundary is None:
        face = surfedge(f)[0] if f.shape[1] == 4 else f
        isboundary = np.isin(np.arange(node.shape[0]), face.flatten() - 1).astype(int)

    # Write node file
    with open(f"{filestub}.node", "w") as fid:
        for i in range(node.shape[0]):
            fid.write(
                f"{int(isboundary[i])}\t{node[i,0]:.16f}\t{node[i,1]:.16f}\t{node[i,2]:.16f}\n"
            )

    # Write elem file
    with open(f"{filestub}.elem", "w") as fid:
        fmt = "\t".join(["%6d"] * f.shape[1]) + "\n"
        for row in f:
            fid.write(fmt % tuple(row))

    # Write region file
    if nodeseg is not None:
        with open(f"{filestub}.region", "w") as fid:
            for val in nodeseg:
                fid.write(f"{int(val)}\n")

    # Write param file
    if nodeprop is not None:
        with open(f"{filestub}.param", "w") as fid:
            if isinstance(proptype, list):
                proptype = "\n".join(proptype)
            fid.write(f"{proptype}\n")
            for row in nodeprop:
                fid.write("\t".join(f"{v:.16f}" for v in row) + "\n")


# _________________________________________________________________________________________________________


def readobjmesh(fname):
    """
    Read a Wavefront OBJ mesh file.

    Parameters:
        fname : str
            Name of the .obj file.

    Returns:
        node : ndarray
            Node coordinates (N, 3).
        face : ndarray
            Face list (M, 3), 1-based indices.
    """
    with open(fname, "r") as f:
        content = f.read()

    # Extract vertices
    verts = re.findall(r"v\s+([-\d.e]+)\s+([-\d.e]+)\s+([-\d.e]+)", content)
    node = np.array([[float(x), float(y), float(z)] for x, y, z in verts])

    # Extract faces (handle v, v/vt, v/vt/vn, v//vn formats)
    faces = re.findall(
        r"f\s+(\d+)(?:/+\d*)*\s+(\d+)(?:/+\d*)*\s+(\d+)(?:/+\d*)*", content
    )
    face = np.array([[int(a), int(b), int(c)] for a, b, c in faces])

    return node, face


# _________________________________________________________________________________________________________


def readsmf(fname):
    """
    Read a Simple Model Format (SMF) file.

    Parameters:
        fname : str
            Name of the SMF file.

    Returns:
        node : ndarray
            Node coordinates (N, 3).
        elem : ndarray
            Element list (M, 3), 1-based indices.
    """
    nodes, elems = [], []
    with open(fname, "r") as fid:
        for line in fid:
            line = line.strip()
            if not line:
                continue
            if line.startswith("v "):
                vals = [float(x) for x in line[2:].split()]
                if len(vals) == 3:
                    nodes.append(vals)
            elif line.startswith("f "):
                vals = [int(x) for x in line[2:].split()]
                if len(vals) == 3:
                    elems.append(vals)

    return np.array(nodes), np.array(elems)


# _________________________________________________________________________________________________________


def saveabaqus(node, face=None, elem=None, fname=None, heading=None):
    """
    Save a mesh to ABAQUS input format.

    Parameters:
        node : ndarray
            Node coordinates (N, 3).
        face : ndarray, optional
            Surface triangles (M, 3 or 4 with labels).
        elem : ndarray, optional
            Tetrahedral elements (K, 4 or 5 with labels).
        fname : str
            Output file name.
        heading : str, optional
            Descriptive header string.
    """
    # Handle variable arguments
    if fname is None:
        if elem is None:
            fname = face
            face, elem = None, None
        else:
            fname = elem
            elem = None

    with open(fname, "w") as fid:
        fid.write("*HEADING\n")
        if heading:
            fid.write(f"**{heading}\n")
        fid.write("*PREPRINT,MODEL=NO,HISTORY=NO,ECHO=NO\n")

        # Write nodes
        if node is not None and len(node) > 0:
            fid.write("*NODE, NSET=MeshNode\n")
            for i, n in enumerate(node[:, :3], 1):
                fid.write(f"{i},\t{n[0]:e},\t{n[1]:e},\t{n[2]:e}\n")

        # Write tetrahedral elements
        count = 0
        if elem is not None and len(elem) > 0:
            if elem.shape[1] == 4:
                elem = np.hstack([elem, np.zeros((len(elem), 1), dtype=int)])
            labels = np.unique(elem[:, 4])
            elsetall = []
            for lab in labels:
                idx = np.where(elem[:, 4] == lab)[0]
                fid.write(f"*ELEMENT, ELSET=MeshTetra{int(lab)}, TYPE=C3D4\n")
                for j, i in enumerate(idx, count + 1):
                    fid.write(
                        f"{j},\t{elem[i,0]},\t{elem[i,1]},\t{elem[i,2]},\t{elem[i,3]}\n"
                    )
                count += len(idx)
                elsetall.append(f"MeshTetra{int(lab)}")
            if elsetall:
                fid.write(f"*ELSET, ELSET=MeshTetraAll\n{','.join(elsetall)}\n")

        # Write surface triangles
        if face is not None and len(face) > 0:
            if face.shape[1] == 3:
                face = np.hstack([face, np.zeros((len(face), 1), dtype=int)])
            labels = np.unique(face[:, 3])
            elsetall = []
            for lab in labels:
                idx = np.where(face[:, 3] == lab)[0]
                fid.write(f"*ELEMENT, ELSET=MeshTri{int(lab)}, TYPE=S3R\n")
                for j, i in enumerate(idx, count + 1):
                    fid.write(f"{j},\t{face[i,0]},\t{face[i,1]},\t{face[i,2]}\n")
                count += len(idx)
                elsetall.append(f"MeshTri{int(lab)}")
            if elsetall:
                fid.write(f"*ELSET, ELSET=MeshTriAll\n{','.join(elsetall)}\n")


# _________________________________________________________________________________________________________


def savemphtxt(node, face, elem, filename):
    """
    Save a tetrahedral mesh to COMSOL .mphtxt format.

    Parameters:
        node : ndarray
            Node coordinates (N, 3).
        face : ndarray
            Surface triangles (M, 3 or 4 with labels).
        elem : ndarray
            Tetrahedral elements (K, 4 or 5 with labels).
        filename : str
            Output file name.
    """
    elem = elem.copy()
    face = face.copy()

    # Reorient elements
    elem[:, :4] = meshreorient(node[:, :3], elem[:, :4])[0]

    # Handle labels
    if face.shape[1] < 4:
        face = np.hstack([face, np.ones((len(face), 1), dtype=int)])
    elif face[:, 3].min() == 0:
        face[:, 3] += 1

    if elem.shape[1] < 5:
        elem = np.hstack([elem, np.ones((len(elem), 1), dtype=int)])
    elif elem[:, 4].min() == 0:
        elem[:, 4] += 1

    with open(filename, "w") as fp:
        fp.write("# Created by pyiso2mesh\n")
        fp.write("0 1\n1\n5 mesh1\n1\n3 obj\n\n")
        fp.write(f"0 0 1\n4 Mesh\n2\n3\n{len(node)}\n1\n")

        # Write nodes
        for n in node[:, :3]:
            fp.write(f"{n[0]:.16f} {n[1]:.16f} {n[2]:.16f}\n")

        # Write triangles
        fp.write("\n2\n\n3 tri\n")
        fp.write(f"\n3\n{len(face)}\n\n")
        for f in face[:, :3]:
            fp.write(f"{f[0]} {f[1]} {f[2]}\n")
        fp.write(f"\n1\n0\n{len(face)}\n")
        for f in face:
            fp.write(f"{int(f[3])}\n")
        fp.write(f"\n{len(face)}\n")

        # Write tetrahedra
        fp.write(f"\n\n3 tet\n4\n\n{len(elem)}\n")
        for e in elem[:, :4]:
            fp.write(f"{e[0]} {e[1]} {e[2]} {e[3]}\n")
        fp.write(f"\n4\n0\n{len(elem)}\n{len(elem)}\n")
        for e in elem:
            fp.write(f"{int(e[4])}\n")
        fp.write("\n0\n")


# _________________________________________________________________________________________________________


def savemsh(node, elem, fname, rname=None):
    """
    Save a tetrahedral mesh to GMSH .msh format (version 2.2).

    Parameters:
        node : ndarray
            Node coordinates (N, 3).
        elem : ndarray
            Tetrahedral elements (M, 4 or 5 with region labels).
        fname : str
            Output file name.
        rname : list of str, optional
            Names for each region.
    """
    if rname is None:
        rname = []

    elem = elem.copy()
    if elem.shape[1] < 5:
        elem = np.hstack([elem, np.ones((len(elem), 1), dtype=int)])

    # Reorient elements
    elem[:, :4] = meshreorient(node, elem[:, :4])[0]

    # Handle negative/zero labels
    reg = np.unique(elem[:, 4])
    reg[reg <= 0] = reg.max() + 1 - reg[reg <= 0]

    with open(fname, "w") as fid:
        # Header
        fid.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")

        # Physical names
        nreg = int(reg.max())
        if nreg > 0:
            fid.write("$PhysicalNames\n")
            fid.write(f"{nreg}\n")
            for r in range(1, nreg + 1):
                name = rname[r - 1] if r <= len(rname) else f"region_{r}"
                fid.write(f'3 {r} "{name}"\n')
            fid.write("$EndPhysicalNames\n")

        # Nodes
        fid.write("$Nodes\n")
        fid.write(f"{len(node)}\n")
        for i, n in enumerate(node, 1):
            fid.write(f"{i} {n[0]:.10f} {n[1]:.10f} {n[2]:.10f}\n")
        fid.write("$EndNodes\n")

        # Elements
        fid.write("$Elements\n")
        fid.write(f"{len(elem)}\n")
        for i, e in enumerate(elem, 1):
            # Format: id type num_tags tag1 tag2 tag3 n1 n2 n3 n4
            region = int(e[4])
            fid.write(
                f"{i} 4 3 {region} {region} 0 {int(e[0])} {int(e[1])} {int(e[2])} {int(e[3])}\n"
            )
        fid.write("$EndElements\n")


# _________________________________________________________________________________________________________


def savesmf(v, f, fname):
    """
    Save a surface mesh to Simple Model Format (SMF).

    Parameters:
        v : ndarray
            Node coordinates (N, 3).
        f : ndarray
            Face list (M, 3), 1-based indices.
        fname : str
            Output file name.
    """
    with open(fname, "w") as fid:
        for vertex in v:
            fid.write(f"v {vertex[0]:.16f} {vertex[1]:.16f} {vertex[2]:.16f}\n")
        for face in f:
            fid.write(f"f {int(face[0])} {int(face[1])} {int(face[2])}\n")


# _________________________________________________________________________________________________________


def savevrml(node, face=None, elem=None, fname=None):
    """
    Save a mesh to VRML 1.0 format.

    Parameters:
        node : ndarray
            Node coordinates (N, 3).
        face : ndarray, optional
            Surface triangles (M, 3), 1-based indices.
        elem : ndarray, optional
            Not used directly; if provided without face, extracts surface.
        fname : str
            Output file name.
    """
    # Handle variable arguments
    if fname is None:
        if elem is None:
            fname = face
            face = None
        else:
            fname = elem
            elem = None

    with open(fname, "w") as fid:
        fid.write("#VRML V1.0 ascii\n#Generated by pyiso2mesh\n")
        fid.write(f"Separator {{\nSwitch {{\n\tDEF {fname}\n\tSeparator {{\n")

        if node is not None and len(node) > 0:
            fid.write("\t\tCoordinate3 {\n\t\t\tpoint [\n")
            for n in node[:, :3]:
                fid.write(f"{n[0]:.16f} {n[1]:.16f} {n[2]:.16f},\n")
            fid.write("\t\t\t]\n\t\t}\n")

        if face is not None and len(face) > 0:
            fid.write("\t\tIndexedFaceSet {\n\t\t\tcoordIndex [\n")
            for f in face[:, :3]:
                fid.write(f"{int(f[0]-1)} {int(f[1]-1)} {int(f[2]-1)} -1\n")
            fid.write("\t\t\t]\n\t\t}\n")

        fid.write("\t} # Separator\n}\n}\n")
