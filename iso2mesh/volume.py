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
    "maskdist",
    "ndgaussian",
    "ndimfilter",
]

##====================================================================================
## dependent libraries
##====================================================================================

from typing import Optional
import numpy as np
from scipy import ndimage


##====================================================================================
## implementations
##====================================================================================


def validatemask(mask, ndim=3):
    """
    Create a 2D or 3D kernel based on the input data dimension
    Input:
        mask: an imdilate and imerode structuring matrix, compute if None
        ndim: 2 or 3

    Returns:
        validated mask
    """
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
    newvol = ndimage.binary_erosion(
        newvol, structure=mask, iterations=layer, border_value=1
    )

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


def maskdist(vol):
    """
    Return the distance in each voxel towards the nearest label boundaries.

    Parameters:
        vol : ndarray
            A 2D or 3D array with label values.

    Returns:
        dist : ndarray
            An array storing the distance (in voxel units) towards the nearest
            boundary between two distinct non-zero voxels. Zero voxels and space
            outside the array are treated as a unique label. For minimum distance
            measured from voxel center, use (dist - 0.5).

    Raises:
        ValueError: If vol is empty or has more than 256 unique values.

    Example:
        >>> a = np.ones((60, 60, 60))
        >>> a[:, :, :10] = 2
        >>> a[:, :, 10:20] = 3
        >>> im = maskdist(a)
        >>> plt.imshow(im[:, 30, :])
    """
    if vol.size == 0:
        raise ValueError("Input vol cannot be empty")

    vals = np.unique(vol)
    if len(vals) > 256:
        raise ValueError(
            "Input appears to be a gray-scale image; convert to binary or labels first"
        )

    # Pad volume with a new unique label
    pad_val = vals.max() + 1
    if vol.ndim == 2:
        newvol = np.full((vol.shape[0] + 2, vol.shape[1] + 2), pad_val, dtype=vol.dtype)
        newvol[1:-1, 1:-1] = vol
    elif vol.ndim == 3:
        newvol = np.full(
            (vol.shape[0] + 2, vol.shape[1] + 2, vol.shape[2] + 2),
            pad_val,
            dtype=vol.dtype,
        )
        newvol[1:-1, 1:-1, 1:-1] = vol
    else:
        raise ValueError("vol must be 2D or 3D")

    # Include padding label, exclude zero (treat zero as padding label)
    vals = list(vals)
    vals.append(pad_val)
    vals = [v for v in vals if v != 0]

    # Replace zeros with padding label
    newvol[newvol == 0] = pad_val

    # Compute minimum distance to any label boundary
    dist = np.full(newvol.shape, np.inf)

    for val in vals:
        mask = newvol == val
        vdist = ndimage.distance_transform_edt(~mask)
        vdist[vdist == 0] = np.inf
        dist = np.minimum(dist, vdist)

    # Remove padding
    if vol.ndim == 2:
        dist = dist[1:-1, 1:-1]
    else:
        dist = dist[1:-1, 1:-1, 1:-1]

    return dist


def ndgaussian(r=1, sigma=1, ndim=3):
    """
    Create an N-dimensional Gaussian kernel.

    Parameters:
        r : int
            Kernel half-width. Output size is (2*r+1) in each dimension.
        sigma : float
            Standard deviation. If inf, returns a box filter.
        ndim : int
            Number of dimensions.

    Returns:
        kernel : ndarray
            Normalized Gaussian kernel.
    """
    size = 2 * r + 1

    if np.isinf(sigma):
        # Box filter
        kernel = np.ones([size] * ndim)
        return kernel / kernel.sum()

    # Create coordinate grids
    coords = [np.arange(-r, r + 1) for _ in range(ndim)]
    grids = np.meshgrid(*coords, indexing="ij")

    # Compute squared distance from center
    dist_sq = sum(g**2 for g in grids)

    # Gaussian
    kernel = np.exp(-dist_sq / (2 * sigma**2))
    return kernel / kernel.sum()


def ndimfilter(im, kernel="box", *args):
    """
    Filter an ND array using convolution.

    Parameters:
        im : ndarray
            Input ND array.
        kernel : ndarray or str
            Filter kernel array, or one of:
            - 'box': box filter (requires r)
            - 'gaussian': Gaussian filter (requires r, sigma)
        *args : additional arguments
            r : int - kernel half-width (output is 2*r+1 in each dimension)
            sigma : float - standard deviation for Gaussian (default: 1)

    Returns:
        img : ndarray
            Filtered ND array.

    Example:
        >>> filtered = ndimfilter(volume, 'gaussian', 2, 1.5)
        >>> filtered = ndimfilter(volume, 'box', 3)
    """
    if isinstance(kernel, str):
        if kernel == "box":
            if len(args) < 1:
                r = 1
            else:
                r = args[0]
            kernel = ndgaussian(r, np.inf, im.ndim)
        elif kernel == "gaussian":
            r = args[0] if len(args) >= 1 else 1
            sigma = args[1] if len(args) >= 2 else 1
            kernel = ndgaussian(r, sigma, im.ndim)
        else:
            raise ValueError(f"Filter type '{kernel}' is not supported")

    return ndimage.convolve(im, kernel, mode="reflect")


##====================================================================================
## aliases
##====================================================================================

thickenbinvol = volgrow
thinbinvol = volshrink
