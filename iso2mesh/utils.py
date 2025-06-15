"""@package docstring
Iso2Mesh for Python - Primitive shape meshing functions

Copyright (c) 2024 Edward Xu <xu.ed at neu.edu>
              2024-2025 Qianqian Fang <q.fang at neu.edu>
"""

__all__ = [
    "getexeext",
    "fallbackexeext",
    "mwpath",
    "mcpath",
    "deletemeshfile",
    "rotatevec3d",
    "rotmat2vec",
    "varargin2struct",
    "jsonopt",
]
##====================================================================================
## dependent libraries
##====================================================================================

import sys
import numpy as np
import os
import shutil
import platform
import re

ISO2MESH_BIN_VER = "1.9.8"

##====================================================================================
## implementations
##====================================================================================


def getexeext():
    osarch = platform.machine()
    ext = ".exe"
    if sys.platform == "linux":
        ext = ".mexa64"
    elif sys.platform.startswith("win"):
        ext = ".exe"
    elif sys.platform == "darwin" and osarch == "arm64":
        ext = ".mexmaca64"
    elif sys.platform == "darwin":
        ext = ".mexmaci64"
    else:
        print("Unable to find extension type")

    return ext


# _________________________________________________________________________________________________________


def fallbackexeext(exesuffix, exename):
    """
    Get the fallback external tool extension names for the current platform.

    Parameters:
        exesuffix: the output executable suffix from getexeext
        exename: the executable name

    Returns:
        exesuff: file extension for iso2mesh tool binaries
    """
    exesuff = exesuffix
    if exesuff == ".mexa64" and not os.path.isfile(
        mcpath(exename, exesuff)
    ):  # fall back to i386 linux
        exesuff = ".mexglx"
    if exesuff == ".mexmaci64" and not os.path.isfile(
        mcpath(exename, exesuff)
    ):  # fall back to i386 mac
        exesuff = ".mexmaci"
    if exesuff == ".mexmaci" and not os.path.isfile(
        mcpath(exename, exesuff)
    ):  # fall back to ppc mac
        exesuff = ".mexmac"
    if not os.path.isfile(mcpath(exename, exesuff)) and not os.path.isfile(
        os.path.join(mcpath(exename))
    ):  # fall back to OS native package
        exesuff = ""

    if not os.path.isfile(mcpath(exename, exesuff)) and not os.path.isfile(
        mcpath(exename)
    ):
        if shutil.which(exename):
            return exesuff
        raise FileNotFoundError(
            f"The following executable:\n\t{mcpath(exename)}{getexeext()}\n"
            "is missing. Please download it from "
            "https://github.com/fangq/iso2mesh/tree/master/bin/ "
            "and save it to the above path, then rerun the script.\n"
        )

    return exesuff


# _________________________________________________________________________________________________________


def mwpath(fname=""):
    """
    Get the full temporary file path by prepending the working directory
    and current session name.

    Parameters:
    fname : str, optional
        Input file name string (default is empty string).

    Returns:
    tempname : str
        Full file name located in the working directory.
    """

    # Retrieve the ISO2MESH_TEMP and ISO2MESH_SESSION environment variables
    p = os.getenv("ISO2MESH_TEMP")
    session = os.getenv("ISO2MESH_SESSION", "")

    # Get the current user's name for Linux/Unix/Mac/Windows
    username = os.getenv("USER") or os.getenv("UserName", "")
    if username:
        username = f"pyiso2mesh-{username}"

    tempname = ""

    if not p:
        tdir = os.path.abspath(
            os.path.join(os.sep, "tmp")
        )  # Use default temp directory
        if username:
            tdir = os.path.join(tdir, username)
            if not os.path.exists(tdir):
                os.makedirs(tdir)

        tempname = os.path.join(tdir, session, fname)
    else:
        tempname = os.path.join(p, session, fname)

    return tempname


# _________________________________________________________________________________________________________


def mcpath(fname, ext=None):
    """
    Get full executable path by prepending a command directory path.

    Parameters:
    fname : str
        Input file name string.
    ext : str, optional
        File extension.

    Returns:
    str
        Full file name located in the bin directory.
    """
    from pathlib import Path

    binname = ""

    # the bin folder under iso2mesh is searched first
    # tempname = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bin', fname)
    tempname = os.path.join(os.path.expanduser("~"), "iso2mesh-tools")
    binfolder = Path(os.path.join(tempname, "iso2mesh-" + ISO2MESH_BIN_VER, "bin"))

    if os.path.isdir(tempname):
        binname = os.path.join(tempname, "iso2mesh-" + ISO2MESH_BIN_VER, "bin", fname)

        if ext:
            if os.path.isfile(binname + ext):
                binname = binname + ext
            else:
                binname = fname + ext

    else:
        import urllib.request
        import zipfile

        print("Iso2mesh meshing utilities do not exist locally, downloading now ...")
        os.makedirs(tempname)
        binurl = f"https://github.com/fangq/iso2mesh/archive/refs/tags/v{ISO2MESH_BIN_VER}.zip"
        filehandle, _ = urllib.request.urlretrieve(binurl)

        with zipfile.ZipFile(filehandle, "r") as zip_ref:
            for file in zip_ref.namelist():
                if file.startswith(f"iso2mesh-{ISO2MESH_BIN_VER}/bin/"):
                    zip_ref.extract(file, tempname)
                    extractfile = os.path.join(tempname, file)
                    print("Extracting " + extractfile)
                    if os.path.isfile(extractfile):
                        print("Setting permission " + extractfile)
                        os.chmod(extractfile, 0o755)
        if ext:
            binname = os.path.join(
                tempname, "iso2mesh-" + ISO2MESH_BIN_VER, "bin", fname + ext
            )
        else:
            binname = os.path.join(
                tempname, "iso2mesh-" + ISO2MESH_BIN_VER, "bin", fname
            )

    # on 64bit windows machine, try 'exename_x86-64.exe' first
    if (
        os.name == "nt"
        and "64" in os.environ["PROCESSOR_ARCHITECTURE"]
        and not re.search(r"_x86-64$", fname)
    ):
        w64bin = re.sub(r"(\.[eE][xX][eE])$", "_x86-64.exe", binname, count=1)
        if os.path.isfile(w64bin):
            binname = w64bin

    # if no such executable exist in iso2mesh/bin, find it in PATH env variable
    if "extractfile" not in locals() and ext and not os.path.isfile(binname):
        binname = fname

    return binname


# _________________________________________________________________________________________________________


def deletemeshfile(fname):
    """
    delete a given work mesh file under the working directory

    author: Qianqian Fang, <q.fang at neu.edu>

    input:
        fname: specified file name (without path)

    output:
        flag: not used
    """

    try:
        if os.path.exists(fname):
            os.remove(fname)
    except Exception as e:
        raise PermissionError(
            "You do not have permission to delete temporary files. If you are working in a multi-user "
            "environment, such as Unix/Linux and there are other users using iso2mesh, "
            "you may need to define ISO2MESH_SESSION='yourstring' to make your output "
            "files different from others; if you do not have permission to "
            f"{os.getcwd()} as the temporary directory, you have to define "
            "ISO2MESH_TEMP='/path/you/have/write/permission' in Python base workspace."
        ) from e


# _________________________________________________________________________________________________________


def rotatevec3d(pt, v1, u1=None, p0=None):
    """
    Rotate 3D points from one Cartesian coordinate system to another.

    Parameters:
    pt : numpy.ndarray
        3D points defined in a standard Cartesian system where a unitary
        z-vector is (0,0,1), 3 columns for x, y and z.
    v1 : numpy.ndarray
        The unitary z-vector for the target coordinate system.
    u1 : numpy.ndarray, optional
        The unitary z-vector for the source coordinate system, if ignored,
        u1=(0,0,1).
    p0 : numpy.ndarray, optional
        Offset of the new coordinate system, if ignored, p0=(0,0,0).

    Returns:
    newpt : numpy.ndarray
        The transformed 3D points.
    """

    if u1 is None:
        u1 = np.array([0, 0, 1])
    if p0 is None:
        p0 = np.array([0, 0, 0])

    u1 = u1 / np.linalg.norm(u1)
    v1 = v1 / np.linalg.norm(v1)

    R, s = rotmat2vec(u1.flatten(), v1.flatten())
    newpt = (R @ pt.T * s).T

    if p0 is not None:
        p0 = p0.flatten()
        newpt += np.tile(p0, (newpt.shape[0], 1))

    return newpt


# _________________________________________________________________________________________________________


def rotmat2vec(u, v):
    """
    [R,s]=rotmat2vec(u,v)

    the rotation matrix from vector u to v, satisfying R*u*s=v

    input:
      u: a 3D vector in the source coordinate system;
      v: a 3D vector in the target coordinate system;

    output:
      R: a rotation matrix to transform normalized u to normalized v
      s: a scaling factor, so that R*u*s=v
    """
    s = np.linalg.norm(v) / np.linalg.norm(u)
    u1 = u / np.linalg.norm(u)
    v1 = v / np.linalg.norm(v)

    k = np.cross(u1, v1)
    if not np.any(k):  # u and v are parallel
        R = np.eye(3)
        return R, s

    # Rodrigues's formula:
    costheta = np.dot(u1, v1)
    R = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = costheta * np.eye(3) + R + np.outer(k, k) * (1 - costheta) / np.sum(k**2)

    return R, s


# _________________________________________________________________________________________________________


def varargin2struct(*args):
    opt = {}
    length = len(args)
    if length == 0:
        return opt

    i = 0
    while i < length:
        if isinstance(args[i], dict):
            opt = {**opt, **args[i]}  # Merging dictionaries
        elif isinstance(args[i], str) and i < length - 1:
            opt[args[i].lower()] = args[i + 1]
            i += 1
        else:
            raise ValueError(
                "input must be in the form of ...,'name',value,... pairs or structs"
            )
        i += 1

    return opt


# _________________________________________________________________________________________________________


def jsonopt(key, default, *args):
    val = default
    if len(args) <= 0:
        return val
    key0 = key.lower()
    opt = args[0]
    if isinstance(opt, dict):
        if key0 in opt:
            val = opt[key0]
        elif key in opt:
            val = opt[key]
    return val
