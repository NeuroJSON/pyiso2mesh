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
]

##====================================================================================
## dependent libraries
##====================================================================================

import numpy as np
import struct
from datetime import datetime
import re
from iso2mesh.trait import meshreorient, surfedge, extractloops

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
        with open(f"{fstub}.node", "rb") as fp:
            dim = [int(x) for x in next(fp).split()]
            if len(dim) < 4:
                raise ValueError("wrong node file")
            node = np.array([])
            for ii in range(dim[0]):
                row = [float(x) for x in next(fp).split()]
                node = np.append(node, row)
            node = node.reshape(dim[0], 4)
            idx = node[:, 1]
            node = node[:, 1:4]
    except FileNotFoundError:
        raise FileNotFoundError("node file is missing!")

    # read element file
    try:
        with open(f"{fstub}.ele", "rb") as fp:
            dim = [int(x) for x in next(fp).split()]
            if len(dim) < 3:
                raise ValueError("wrong elem file")
            elem = np.array([])
            for ii in range(dim[0]):
                row = [int(x) for x in next(fp).split()]
                elem = np.append(elem, row)
            elem = elem.reshape(dim[0], dim[1] + dim[2] + 1)
            elem = elem[:, 1:].astype(int)
            elem[:, :4] += 1
            # elem[:, :dim[1]] += (1 - idx[0])
    except FileNotFoundError:
        raise FileNotFoundError("elem file is missing!")

    # read surface mesh file
    try:
        with open(f"{fstub}.face", "rb") as fp:
            dim = [int(x) for x in next(fp).split()]
            if len(dim) < 2:
                raise ValueError("wrong surface file")
            face = np.array([])
            for ii in range(dim[0]):
                row = [int(x) for x in next(fp).split()]
                face = np.append(face, row)
            face = face.reshape(dim[0], 5)
            face = np.hstack((face[:, 1:-1], face[:, -1:])).astype(int)
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
                    for i in range(len(faceid)):
                        fp.write("1 0 {} \n{} {} {} {}\n".format(faceid[i], *elem[i]))
                else:
                    for i in range(elem.shape[0]):
                        fp.write("1 0\n{} {} {} {}\n".format(*elem[i]))

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
