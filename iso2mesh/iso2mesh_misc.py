"""@package docstring
Iso2Mesh for Python - Primitive shape meshing functions

Copyright (c) 2024 Edward Xu <xu.ed at neu.edu>
              2024-2025 Qianqian Fang <q.fang at neu.edu>
"""

__all__ = [
    "getexeext",
    "rotatevec3d",
    "fallbackexeext",
    "varargin2struct",
    "jsonopt",
]
##====================================================================================
## dependent libraries
##====================================================================================

import sys
import numpy as np
import os
import iso2mesh as im
import shutil

##====================================================================================
## implementations
##====================================================================================


def getexeext():
    ext = ".exe"
    if sys.platform == "linux":
        ext = ".mexa64"
    elif sys.platform.startwith("win"):
        ext = ".exe"
    elif sys.platform == "darwin":
        ext = ".mexmaci64"
    elif sys.platform == "darwin" and sys.maxsize > 2**32:
        ext = ".mexmaca64"
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
        im.mcpath(exename, exesuff)
    ):  # fall back to i386 linux
        exesuff = ".mexglx"
    if exesuff == ".mexmaci64" and not os.path.isfile(
        im.mcpath(exename, exesuff)
    ):  # fall back to i386 mac
        exesuff = ".mexmaci"
    if exesuff == ".mexmaci" and not os.path.isfile(
        im.mcpath(exename, exesuff)
    ):  # fall back to ppc mac
        exesuff = ".mexmac"
    if not os.path.isfile(im.mcpath(exename, exesuff)) and not os.path.isfile(
        os.path.join(im.mcpath(exename))
    ):  # fall back to OS native package
        exesuff = ""

    if not os.path.isfile(im.mcpath(exename, exesuff)) and not os.path.isfile(
        im.mcpath(exename)
    ):
        if shutil.which(exename):
            return exesuff
        raise FileNotFoundError(
            f"The following executable:\n\t{im.mcpath(exename)}{getexeext()}\n"
            "is missing. Please download it from "
            "https://github.com/fangq/iso2mesh/tree/master/bin/ "
            "and save it to the above path, then rerun the script.\n"
        )

    return exesuff


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
