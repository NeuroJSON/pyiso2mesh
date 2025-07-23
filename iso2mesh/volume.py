"""@package docstring
Iso2Mesh for Python - Mesh data queries and manipulations

Copyright (c) 2024-2025 Qianqian Fang <q.fang at neu.edu>
"""
__all__ = [
    "volgrow",
    "volshrink",
    "volopen",
    "volclose",
    "fillholes3d",
    "thickenbinvol",
    "thinbinvol",
]

##====================================================================================
## dependent libraries
##====================================================================================

import numpy as np
from scipy import ndimage
from typing import Optional


##====================================================================================
## implementations
##====================================================================================


def validatemask(mask, ndim=3):
    # Create default mask if not provided or empty
    if mask is None or mask.size == 0:
        if ndim == 3:
            # Create 3D cross-shaped mask for 3D volumes
            mask = ndimage.generate_binary_structure(3, 1)
        else:
            # Create 2D cross-shaped mask for 2D images
            mask = ndimage.generate_binary_structure(2, 1)

    # Rotate mask by 180 degrees (equivalent to rot90(mask, 2) in MATLAB)
    if mask.ndim == 3:
        # For 3D arrays, rotate around all axes
        mask = np.rot90(mask, 2, axes=(0, 1))
        mask = np.rot90(mask, 2, axes=(0, 2))
    else:
        # For 2D arrays, simple 180 degree rotation
        mask = np.rot90(mask, 2)

    return mask


def volgrow(
    vol: np.ndarray, layer: int = 1, mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Thickening a binary image or volume by a given pixel width
    This is similar to bwmorph(vol,'thicken',3) except
    this does it in both 2d and 3d

    Author: Qianqian Fang, <q.fang at neu.edu>
    Python version adapted from original MATLAB code

    Parameters:
    -----------
    vol : ndarray
        A volumetric binary image
    layer : int, optional
        Number of iterations for the thickening (default: 1)
    mask : ndarray, optional
        A 2D or 3D neighborhood mask (default: None, will create appropriate mask)

    Returns:
    --------
    newvol : ndarray
        The volume image after the thickening

    Notes:
    ------
    This function is part of iso2mesh toolbox (http://iso2mesh.sf.net)
    """

    mask = validatemask(mask, vol.ndim)

    # Convert vol to appropriate type for processing
    newvol = vol.astype(np.float32)

    # Perform iterative dilation using scipy's binary_dilation
    # which is more appropriate for binary morphological operations
    mask_bool = mask > 0

    # Use scipy's binary_dilation for proper binary morphological operation
    newvol = ndimage.binary_dilation(newvol > 0, structure=mask_bool, iterations=layer)

    # Convert back to double precision (equivalent to MATLAB's double())
    newvol = newvol.astype(np.float64)

    return newvol


def volshrink(
    vol: np.ndarray, layer: int = 1, mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Alternative implementation using scipy's binary_erosion for proper morphological thinning

    This version uses scipy's binary_erosion which is more mathematically appropriate
    for binary morphological thinning operations.

    Parameters:
    -----------
    vol : ndarray
        A volumetric binary image
    layer : int, optional
        Number of iterations for the thinning (default: 1)
    mask : ndarray, optional
        A 2D or 3D neighborhood mask (default: None, will create appropriate mask)

    Returns:
    --------
    newvol : ndarray
        The volume image after the thinning operations
    """

    mask = validatemask(mask, vol.ndim)

    # Convert input to binary
    newvol = vol != 0

    # Perform iterative binary erosion (morphological thinning)
    newvol = ndimage.binary_erosion(newvol, structure=mask, iterations=layer)

    # Convert back to double precision
    newvol = newvol.astype(np.float64)

    return newvol


def volclose(
    vol: np.ndarray, layer: int = 1, mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Alternative implementation using scipy's binary_closing for proper morphological closing

    This version uses scipy's optimized binary_closing operation which is more
    mathematically appropriate and efficient for morphological closing.

    Parameters:
    -----------
    vol : ndarray
        A volumetric binary image
    layer : int, optional
        Number of iterations for the closing (default: 1)
    mask : ndarray, optional
        A 2D or 3D neighborhood mask (default: None, will create appropriate mask)

    Returns:
    --------
    newvol : ndarray
        The volume image after closing
    """

    # Validate input
    if vol is None:
        raise ValueError("must provide a volume")

    mask = validatemask(mask, vol.ndim)

    # Convert input to binary
    newvol = vol != 0

    # Perform iterative binary closing
    newvol = ndimage.binary_closing(newvol, structure=mask, iterations=layer)

    # Convert back to double precision
    newvol = newvol.astype(np.float64)

    return newvol


def volopen(
    vol: np.ndarray, layer: int = 1, mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Alternative implementation using scipy's binary_closing for proper morphological closing

    This version uses scipy's optimized binary_closing operation which is more
    mathematically appropriate and efficient for morphological closing.

    Parameters:
    -----------
    vol : ndarray
        A volumetric binary image
    layer : int, optional
        Number of iterations for the closing (default: 1)
    mask : ndarray, optional
        A 2D or 3D neighborhood mask (default: None, will create appropriate mask)

    Returns:
    --------
    newvol : ndarray
        The volume image after closing
    """

    # Validate input
    if vol is None:
        raise ValueError("must provide a volume")

    mask = validatemask(mask, vol.ndim)

    # Convert input to binary
    newvol = vol != 0

    # Perform iterative binary closing
    newvol = ndimage.binary_opening(newvol, structure=mask, iterations=layer)

    # Convert back to double precision
    newvol = newvol.astype(np.float64)

    return newvol


def fillholes3d(
    img: np.ndarray, maxgap=None, mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Close a 3D image with the specified gap size and then fill the holes

    Author: Qianqian Fang, <q.fang at neu.edu>
    Python version adapted from original MATLAB code

    Parameters:
    -----------
    img : ndarray
        A 2D or 3D binary image
    maxgap : int, ndarray, list, tuple, or None
        If is a scalar, specify maximum gap size for image closing
        If a pair of coordinates, specify the seed position for floodfill
        If None, no initial closing operation is performed
    mask : ndarray, optional
        Neighborhood structure element for floodfilling (default: None)

    Returns:
    --------
    resimg : ndarray
        The image free of holes

    Notes:
    ------
    This function is part of iso2mesh toolbox (http://iso2mesh.sf.net)

    The function works in two phases:
    1. If maxgap is a scalar > 0, apply morphological closing to bridge small gaps
    2. Fill holes using either scipy's binary_fill_holes or custom flood-fill algorithm

    When maxgap is coordinates, it specifies seed points for flood-filling specific regions.
    """

    # Convert to binary and fill holes
    binary_img = img > 0
    if maxgap:
        binary_img = volclose(binary_img, maxgap, mask)

    filled = ndimage.binary_fill_holes(binary_img)

    # Convert back to float64
    resimg = filled.astype(np.float64)

    return resimg


##====================================================================================
## aliases
##====================================================================================

thickenbinvol = volgrow
thinbinvol = volshrink
