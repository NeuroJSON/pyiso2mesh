"""@package docstring
Iso2Mesh for Python - Mesh data queries and manipulations

Copyright (c) 2024-2025 Qianqian Fang <q.fang at neu.edu>
"""
__all__ = [
    "brain2mesh",
    "brain1020",
    "intriangulation",
    "label2tpm",
    "tpm2label",
]

##====================================================================================
## dependent libraries
##====================================================================================

import numpy as np
import sys
from iso2mesh.core import v2s, s2m
from iso2mesh.modify import (
    removeisolatednode,
    removedupelem,
    surfboolean,
    meshcheckrepair,
    sms,
    mergemesh,
    slicesurf3,
    slicesurf,
)
from iso2mesh.trait import (
    meshcentroid,
    elemvolume,
    volface,
    layersurf,
    faceneighbors,
    ray2surf,
)
from iso2mesh.geometry import meshabox
from iso2mesh.volume import volgrow, fillholes3d
from iso2mesh.line import polylineinterp, polylinelen, polylinesimplify, closestnode
from iso2mesh.plot import plotmesh

from typing import Tuple, Dict, Union, Optional, List
import matplotlib.pyplot as plt
import warnings
from collections import OrderedDict

##====================================================================================
## implementations
##====================================================================================


def brain2mesh(seg, **cfg):
    """
    Brain2mesh: a one-liner for human brain 3D mesh generation

    Author: Qianqian Fang <q.fang at neu.edu>
    Other contributors: see AUTHORS.txt for details
    Version: 0.8 (Python port)
    URL: http://mcx.space/brain2mesh
    License: GPL version 3
    Reference:
      Anh Phong Tran, Shijie Yan and Qianqian Fang, "Improving model-based
      fNIRS analysis using mesh-based anatomical and light-transport models,"
      Neurophotonics, 7(1), 015008, URL: http://dx.doi.org/10.1117/1.NPh.7.1.015008

    Parameters:
    -----------
    seg : dict or ndarray
        Pre-segmented brain volume (supporting both probalistic tissue
        segmentation and labeled volume). Two formats are accepted:
        1. a dict with subfields (wm,gm,csf,skull,scalp)
           e.g.: seg['wm'], seg['gm'], seg['csf'] represents the white-matter,
           gray-matter and csf segmentations, respectively,
               or
        2. a 4D array for with tissues sorted in outer-to-inner order
           the 4th dimension of the array can 3-6, with the following assumptions
           size(seg,3) == 6 assumes 0-Scalp, 1-Skull, 2-CSF, 3-GM, 4-WM, 5-air pockets
           size(seg,3) == 5 assumes 0-Scalp, 1-Skull, 2-CSF, 3-GM, 4-WM
           size(seg,3) == 4 assumes 0-Scalp, 1-CSF, 2-GM, 3-WM
           size(seg,3) == 3 assumes 0-CSF, 1-GM, 2-WM

    cfg : dict, optional
        Configuration options for the resulting tetrahedral mesh
        cfg['radbound'] : dict with keys {wm,gm,csf,skull,scalp}
           Radius of the Delaunay sphere used in sampling the surfaces.
           Default values are 1.7, 1.7, 2, 2.5 and 3, respectively (reference values for 1x1x1mm^3)
           Scale proportionally for denser volumes. Lower values correspond to denser, higher
           fidelity surface extraction, but also results in denser meshes.
        cfg['maxnode'] : int, default=100000
           Maximum number of nodes extracted for a given surface.
        cfg['maxvol'] : float, default=100
           Maximum volumetric size of elements
        cfg['smooth'] : int, default=0
           Number of iterations to smooth each tissue surface
        cfg['ratio'] : float, default=1.414
           Radius-edge ratio. Lower values increase quality but results in denser meshes
        cfg['dorelabel'] : bool, default=False
           Removes layered meshing assumptions. Currently only works if all five tissue types are present.
        cfg['doairseg'] : bool, default=True
           Within skull layer, additional segmentations can be found.
        cfg['dotruncate'] : str or bool, default='-z' or True
           Truncate mesh in specified direction or disable with 0/False.
        cfg['marginsize'] : int, default=4
           Number of voxels below CSF mesh to truncate when dotruncate is set.
        cfg['imfill'] : str, default='fillholes3d'
           Function name for 3D image hole-filling function

    Returns:
    --------
    brain_n : ndarray
        Node coordinates of the tetrahedral mesh
    brain_el : ndarray
        Element list of the tetrahedral mesh / the last column denotes the boundary ID
    brain_f : ndarray
        Mesh surface element list of the tetrahedral mesh

    Tissue ID for the outputs are as follow:
    0-Air/background, 1-Scalp, 2-Skull, 3-CSF, 4-GM, 5-WM, 6-air pockets

    Reference:
    If you use Brain2Mesh in your publication, the authors of this toolbox
    appreciate if you can cite our Neurophotonics paper listed above.
    """

    # Handling the inputs
    if seg is None:
        print(__doc__)
        return

    # Default density and adaptiveness parameters
    density = {"wm": 2, "gm": 2, "csf": 5, "skull": 4, "scalp": 8}
    adaptiveness = {"wm": 1, "gm": 1, "csf": 1, "skull": 1, "scalp": 1}

    # Parse configuration
    radbound = cfg.get("radbound", density)
    distbound = cfg.get("distbound", adaptiveness)
    qratio = cfg.get("ratio", 1.414)
    maxvol = cfg.get("maxvol", 100)
    maxnode = cfg.get("maxnode", 100000)
    dotruncate = cfg.get("dotruncate", 1)
    dorelabel = cfg.get("dorelabel", 0)
    doairseg = cfg.get("doairseg", 1)
    threshold = cfg.get("threshold", 0.5)
    smooth = cfg.get("smooth", 0)
    surfonly = cfg.get("surfonly", 0)
    marginsize = cfg.get("marginsize", 4)

    segname = list(density.keys())

    # Convert seg to dict format if needed
    if isinstance(seg, dict):
        tpm = seg
    elif isinstance(seg, np.ndarray) and seg.ndim == 4:
        tpm = {}
        for i in range(seg.shape[3]):
            if i < len(segname):
                tpm[segname[i]] = seg[:, :, :, i]
    else:
        raise ValueError("This seg input is currently not supported")

    # Normalizing segmentation inputs to 0-1
    def normalizer(x):
        return np.array(x, dtype=np.float64) / float(np.max(x))

    for key in tpm:
        tpm[key] = normalizer(tpm[key])

    opt = []
    for i in range(len(tpm)):
        opt_i = {"maxnode": maxnode}
        if segname[i] in radbound:
            opt_i["radbound"] = radbound[segname[i]]
        opt_i["distbound"] = distbound[segname[i]]
        opt.append(opt_i)

    cube3 = np.ones((3, 3, 3), dtype=bool)

    # Pre-processing steps to create separations between the tissues in the volume space
    dim = tpm["wm"].shape
    tpm["wm"] = fillholes3d(tpm["wm"] > 0)
    p_wm = tpm["wm"].copy()
    p_pial = p_wm + tpm["gm"]
    p_pial = np.maximum(p_pial, volgrow(p_wm, 1, cube3))
    p_pial = fillholes3d(p_pial > 0)
    expandedGM = p_pial - tpm["wm"] - tpm["gm"]
    expandedGM = volgrow(expandedGM, 1, cube3)

    if "csf" in tpm:
        p_csf = p_pial + tpm["csf"]
        p_csf[p_csf > 1] = 1
        p_csf = np.maximum(p_csf, volgrow(p_pial, 1, cube3))
        expandedCSF = p_csf - tpm["wm"] - tpm["gm"] - tpm["csf"] - expandedGM
        expandedCSF = volgrow(expandedCSF, 1, cube3)

    if "skull" in tpm and "scalp" in tpm and "csf" in tpm:
        p_bone = p_csf + tpm["skull"]
        p_bone[p_bone > 1] = 1
        p_bone = np.maximum(p_bone, volgrow(p_csf, 1, cube3))
        p_skin = p_bone + tpm["scalp"]
        p_skin[p_skin > 1] = 1
        p_skin = np.maximum(p_skin, volgrow(p_bone, 1, cube3))
        expandedSkull = (
            p_bone
            - tpm["wm"]
            - tpm["gm"]
            - tpm["csf"]
            - tpm["skull"]
            - expandedCSF
            - expandedGM
        )
        expandedSkull = volgrow(expandedSkull, 1, cube3)
    elif "scalp" in tpm and "skull" not in tpm:
        p_skin = p_csf + tpm["scalp"]
        p_skin[p_skin > 1] = 1
        p_skin = np.maximum(p_skin, volgrow(p_csf, 1, cube3))
    elif "skull" in tpm and "scalp" not in tpm:
        p_bone = p_csf + tpm["skull"]
        p_bone[p_bone > 1] = 1
        p_bone = np.maximum(p_bone, volgrow(p_csf, 1, cube3))

    # Grayscale/Binary extractions of the surface meshes for the different tissues
    thresh = 0.5
    if not isinstance(threshold, dict):
        thresh = threshold

    wm_thresh = threshold.get("wm", thresh) if isinstance(threshold, dict) else thresh
    gm_thresh = threshold.get("gm", thresh) if isinstance(threshold, dict) else thresh

    wm_n, wm_f, _, _ = v2s(p_wm, wm_thresh, opt[0], "cgalsurf")
    pial_n, pial_f, _, _ = v2s(p_pial, gm_thresh, opt[1], "cgalsurf")
    wm_n, wm_f = meshcheckrepair(wm_n, wm_f[:, :3], "isolated")
    pial_n, pial_f = meshcheckrepair(pial_n, pial_f[:, :3], "isolated")

    if "csf" in tpm:
        csf_thresh = (
            threshold.get("csf", thresh) if isinstance(threshold, dict) else thresh
        )
        csf_n, csf_f, _, _ = v2s(p_csf, csf_thresh, opt[2], "cgalsurf")
        csf_n, csf_f = meshcheckrepair(csf_n, csf_f[:, :3], "isolated")

    if "skull" in tpm:
        optskull = {"radbound": radbound["skull"], "maxnode": maxnode}
        skull_thresh = (
            threshold.get("skull", thresh) if isinstance(threshold, dict) else thresh
        )
        bone_n, bone_f, _, _ = v2s(p_bone, skull_thresh, optskull, "cgalsurf")

        bone_node, el_bone, _ = s2m(
            bone_n, bone_f, 1.0, maxvol, "tetgen1.5", None, None, "-A"
        )
        unique_labels = np.unique(el_bone[:, 4])  # 5th column is index 4 (0-based)
        vol_bone = []
        for i, label in enumerate(unique_labels):
            vol_bone.append(
                np.sum(elemvolume(bone_node, el_bone[el_bone[:, 4] == label, :4]))
            )

        maxval = np.max(vol_bone)
        I = np.argmax(vol_bone)
        max_label = unique_labels[I]

        if len(unique_labels) > 1:
            no_air2 = bone_node
            el_air2 = el_bone[el_bone[:, 4] != max_label, :]
            no_air2, el_air2, _ = removeisolatednode(no_air2, el_air2)
            f_air2 = volface(el_air2[:, :4])[0]

        bone_n2 = bone_node
        bone_f2 = volface(el_bone[:, :4])[0]
        bone_f2 = removedupelem(bone_f2)
        bone_n2, bone_f2, _ = removeisolatednode(bone_n2, bone_f2)
        if doairseg == 0:
            bone_n = bone_n2
            bone_f = bone_f2

    if "scalp" in tpm:
        optscalp = {"radbound": radbound["scalp"], "maxnode": maxnode}
        scalp_thresh = (
            threshold.get("scalp", thresh) if isinstance(threshold, dict) else thresh
        )
        skin_n, skin_f, _, _ = v2s(p_skin, scalp_thresh, optscalp, "cgalsurf")

    # Smoothing step
    if isinstance(smooth, dict) or smooth > 0:
        scount = 0
        if not isinstance(smooth, dict):
            scount = smooth

        wm_smooth = smooth.get("wm", scount) if isinstance(smooth, dict) else scount
        if wm_smooth > 0:
            wm_n = sms(wm_n, wm_f[:, :3], wm_smooth, 0.5, "lowpass")
            wm_n, wm_f = meshcheckrepair(wm_n, wm_f[:, :3], "meshfix")

        gm_smooth = smooth.get("gm", scount) if isinstance(smooth, dict) else scount
        if gm_smooth > 0:
            pial_n = sms(pial_n, pial_f[:, :3], gm_smooth, 0.5, "lowpass")
            pial_n, pial_f = meshcheckrepair(pial_n, pial_f[:, :3], "meshfix")

        if "csf" in tpm:
            csf_smooth = (
                smooth.get("csf", scount) if isinstance(smooth, dict) else scount
            )
            if csf_smooth > 0:
                csf_n = sms(csf_n, csf_f[:, :3], csf_smooth, 0.5, "lowpass")
                csf_n, csf_f = meshcheckrepair(csf_n, csf_f[:, :3], "meshfix")

        if "skull" in tpm:
            skull_smooth = (
                smooth.get("skull", scount) if isinstance(smooth, dict) else scount
            )
            if skull_smooth > 0:
                bone_n = sms(bone_n, bone_f[:, :3], skull_smooth, 0.5, "lowpass")
                bone_n, bone_f = meshcheckrepair(bone_n, bone_f[:, :3], "meshfix")

        if "scalp" in tpm:
            scalp_smooth = (
                smooth.get("scalp", scount) if isinstance(smooth, dict) else scount
            )
            if scalp_smooth > 0:
                skin_n = sms(skin_n, skin_f[:, :3], scalp_smooth, 0.5, "lowpass")
                skin_n, skin_f = meshcheckrepair(skin_n, skin_f[:, :3], "meshfix")

    # Surface only mode
    if surfonly == 1:
        wm_f = np.column_stack([wm_f, np.ones(wm_f.shape[0])])
        pial_f = np.column_stack([pial_f, np.full(pial_f.shape[0], 2)])
        brain_n, brain_el = mergemesh(wm_n, wm_f, pial_n, pial_f)

        if "csf" in tpm:
            csf_f = np.column_stack([csf_f, np.full(csf_f.shape[0], 3)])
            brain_n, brain_el = mergemesh(brain_n, brain_el, csf_n, csf_f)
            if "skull" in tpm:
                bone_f = np.column_stack([bone_f, np.full(bone_f.shape[0], 4)])
                brain_n, brain_el = mergemesh(brain_n, brain_el, bone_n, bone_f)
                if "scalp" in tpm:
                    skin_f = np.column_stack([skin_f, np.full(skin_f.shape[0], 5)])
                    brain_n, brain_el = mergemesh(brain_n, brain_el, skin_n, skin_f)

        labels, ia, ib = np.unique(
            brain_el[:, 3], return_index=True, return_inverse=True
        )
        labels = np.arange(5, 5 - len(labels), -1)
        brain_el[:, 3] = labels[ib]
        brain_f = np.array([])
        return brain_n, brain_el, brain_f

    # Main loop for the meshing pipeline
    for loop in range(2):
        # If the first pass fails, a second pass is called using the decoupled function
        # to eliminate intersections between surface meshes
        if (loop == 1) and ("label_elem" in locals()):
            continue
        if (loop == 1) and ("label_elem" not in locals()):
            if "bone_n" in locals() and "skin_n" in locals():
                bone_n, bone_f = surfboolean(
                    bone_n[:, :3],
                    bone_f[:, :3],
                    "decouple",
                    skin_n[:, :3],
                    skin_f[:, :3],
                )
            if "bone_n" in locals() and "csf_n" in locals():
                csf_n, csf_f = surfboolean(
                    csf_n[:, :3], csf_f[:, :3], "decouple", bone_n[:, :3], bone_f[:, :3]
                )
            if "pial_n" in locals() and "csf_n" in locals():
                pial_n, pial_f = surfboolean(
                    pial_n[:, :3], pial_f[:, :3], "decouple", csf_n[:, :3], csf_f[:, :3]
                )
            if "pial_n" in locals() and "wm_n" in locals():
                wm_n, wm_f = surfboolean(
                    wm_n[:, :3], wm_f[:, :3], "decouple", pial_n[:, :3], pial_f[:, :3]
                )

        if "wm" in tpm and "gm" in tpm:
            surf_n, surf_f = surfboolean(
                wm_n[:, :3], wm_f[:, :3], "resolve", pial_n, pial_f
            )
        if "csf" in tpm:
            surf_n, surf_f = surfboolean(surf_n, surf_f, "resolve", csf_n, csf_f)
        if "skull" in tpm:
            surf_n, surf_f = surfboolean(surf_n, surf_f, "resolve", bone_n, bone_f)
        if "scalp" in tpm:
            surf_n, surf_f = surfboolean(surf_n, surf_f, "resolve", skin_n, skin_f)

        final_surf_n = surf_n
        final_surf_f = surf_f

        if surfonly == 2:
            brain_n = final_surf_n
            brain_el = final_surf_f
            brain_f = np.array([])
            return brain_n, brain_el, brain_f

        # If the whole head option is deactivated, the cut is made at the base of the brain using a box cutting
        if dotruncate == 1 or isinstance(dotruncate, str):
            dim_max = np.max(surf_n, axis=0)
            if "csf" in tpm:
                dim2 = np.min(csf_n, axis=0)
            else:
                dim2 = np.min(surf_n, axis=0)

            if dotruncate == 1 or dotruncate == "-z":
                nbox, fbox, ebox = meshabox(
                    [-1, -1, dim2[2] + marginsize],
                    [dim_max[0] + 1, dim_max[1] + 1, dim_max[2] + 1],
                    500,
                )
            elif dotruncate == "-y":
                nbox, fbox, ebox = meshabox(
                    [-1, dim2[1] + marginsize, -1],
                    [dim_max[0] + 1, dim_max[1] + 1, dim_max[2] + 1],
                    500,
                )
            elif dotruncate == "-x":
                nbox, fbox, ebox = meshabox(
                    [dim2[0] + marginsize, -1, -1],
                    [dim_max[0] + 1, dim_max[1] + 1, dim_max[2] + 1],
                    500,
                )
            elif dotruncate == "+z":
                nbox, fbox, ebox = meshabox(
                    [-1, -1, -1],
                    [dim_max[0] + 1, dim_max[1] + 1, dim2[2] - marginsize],
                    500,
                )
            elif dotruncate == "+y":
                nbox, fbox, ebox = meshabox(
                    [-1, -1, -1],
                    [dim_max[0] + 1, dim2[1] - marginsize, dim_max[2] + 1],
                    500,
                )
            elif dotruncate == "+x":
                nbox, fbox, ebox = meshabox(
                    [-1, -1, -1],
                    [dim2[0] - marginsize, dim_max[1] + 1, dim_max[2] + 1],
                    500,
                )

            fbox = volface(ebox)[0]
            nbox, fbox, _ = removeisolatednode(nbox, fbox)
            final_surf_n, final_surf_f = surfboolean(
                nbox, fbox, "first", surf_n, surf_f
            )

        if surfonly == 3:
            brain_n = final_surf_n
            brain_el = final_surf_f
            brain_f = np.array([])
            return brain_n, brain_el, brain_f

        # Generates a coarse tetrahedral mesh of the combined tissues
        try:
            final_n, final_e, _ = s2m(
                final_surf_n, final_surf_f, 1.0, maxvol, "tetgen1.5", None, None, "-A"
            )
        except:
            print(
                "volumetric mesh generation failed, returning the intermediate surface model only"
            )
            brain_n = final_surf_n
            brain_f = final_surf_f
            brain_el = np.array([])
            return brain_n, brain_el, brain_f

        # Removes the elements that are part of the box, but not the brain/head
        if dotruncate == 1 or isinstance(dotruncate, str):
            maxval_idx = np.argmax(final_n, axis=0)
            k = np.where(final_e[:, :4] == maxval_idx[2] + 1)[0]
            if len(k) > 0:
                k = k[0]
                exclude_label = final_e[k % len(final_e), 4]
                final_e = final_e[final_e[:, 4] != exclude_label, :]
                final_n, final_e, _ = removeisolatednode(final_n, final_e)

        # Here the labels created through the coarse mesh generated through Tetgen are saved
        # with the centroid of one of the elements for intriangulation testing later
        label, label_elem = np.unique(final_e[:, 4], return_index=True)
        label_centroid = meshcentroid(final_n, final_e[label_elem, :4])

        if "scalp" in tpm:
            no_skin, el_skin, _ = s2m(
                skin_n, skin_f, 1.0, maxvol, "tetgen1.5", None, None, "-A"
            )
            unique_skin_labels = np.unique(el_skin[:, 4])
            vol_skin = []
            for i, skin_label in enumerate(unique_skin_labels):
                vol_skin.append(
                    np.sum(
                        elemvolume(no_skin, el_skin[el_skin[:, 4] == skin_label, :4])
                    )
                )

            maxval = np.max(vol_skin)
            I = np.argmax(vol_skin)
            max_skin_label = unique_skin_labels[I]

            if len(unique_skin_labels) > 1:
                no_air = no_skin
                el_air = el_skin[el_skin[:, 4] != max_skin_label, :]
                no_air, el_air, _ = removeisolatednode(no_air, el_air)
                f_air = volface(el_air[:, :4])[0]
                f_air = removedupelem(f_air)

            el_skin = el_skin[el_skin[:, 4] == max_skin_label, :]
            no_skin, el_skin, _ = removeisolatednode(no_skin, el_skin)
            f_skin = volface(el_skin[:, :4])[0]
            f_skin = removedupelem(f_skin)

        # When the label_elem does not exist, it often indicates a failure at the generation of a coarse
        # tetrahedral mesh. The alternative meshing pathway using decoupling is then called to make a
        # second attempt at creating the combined tetrahedral mesh.
        if "label_elem" not in locals() and loop == 0:
            print(
                "Initial meshing procedure failed. The option parameter might need to be adjusted."
            )
            print("Activating alternative meshing pathway...")
            continue

        # The labels are given to each of the tissues
        # WM(1) - GM(2) - CSF(3) - Bone(4) - Scalp(5) - Air(6)
        newlabel = np.zeros(len(label_elem))
        if "bone_n" in locals() and "no_air2" in locals():
            # Need to subtract 1 when using as indices since faces/elements are 1-indexed
            newlabel = intriangulation(no_air2, f_air2[:, :3], label_centroid)
        if "no_skin" in locals() and "no_air" in locals():
            newlabel = newlabel | intriangulation(no_air, f_air[:, :3], label_centroid)

        newlabel = newlabel.astype(float)
        idx = np.where(newlabel == 0)[0]

        newtag = np.zeros(len(idx))
        newtag = intriangulation(wm_n, wm_f[:, :3], label_centroid[idx, :]) * 6
        newtag = np.maximum(
            newtag,
            intriangulation(pial_n, pial_f[:, :3], label_centroid[idx, :]) * 5,
        )
        if "csf_n" in locals():
            newtag = np.maximum(
                newtag,
                intriangulation(csf_n, csf_f[:, :3], label_centroid[idx, :]) * 4,
            )
        if "bone_n2" in locals():
            newtag = np.maximum(
                newtag,
                intriangulation(bone_n2, bone_f2[:, :3], label_centroid[idx, :]) * 3,
            )
        if "no_skin" in locals():
            newtag = np.maximum(
                newtag,
                intriangulation(no_skin, f_skin[:, :3], label_centroid[idx, :]) * 2,
            )

        newlabel[idx] = newtag
        newlabel = 7 - newlabel
        final_e[:, 4] = newlabel[final_e[:, 4] - 1]  # Subtract 1 for 0-based indexing

        # This step consolidates adjacent labels of the same tissue
        new_label = np.unique(final_e[:, 4])
        face = np.empty((0, 3), dtype=int)
        for i in new_label:
            face = np.vstack([face, volface(final_e[final_e[:, 4] == i, :4])[0]])
        face = np.sort(face, axis=1)
        face = np.unique(face, axis=0)
        node, face, _ = removeisolatednode(final_n, face)

        # The final mesh is generated here with the desired properties
        cmdopt = f"-A -pq{qratio}a{maxvol}"
        brain_n, brain_el, _ = s2m(
            node, face, 1.0, maxvol, "tetgen1.5", None, None, cmdopt
        )

        label2, label_brain_el = np.unique(brain_el[:, 4], return_index=True)
        label_centroid2 = meshcentroid(brain_n, brain_el[label_brain_el, :4])

        # The labeling process is repeated for the final mesh
        # WM(1) - GM(2) - CSF(3) - Bone(4) - Scalp(5) - Air(6)
        newlabel = np.zeros(len(label_brain_el))
        if "bone_n" in locals() and "no_air2" in locals():
            newlabel = intriangulation(no_air2, f_air2[:, :3], label_centroid2)
        if "no_skin" in locals() and "no_air" in locals():
            newlabel = newlabel | intriangulation(no_air, f_air[:, :3], label_centroid2)

        newlabel = newlabel.astype(float)
        idx = np.where(newlabel == 0)[0]

        newtag = np.zeros(len(idx))
        newtag = intriangulation(wm_n, wm_f[:, :3], label_centroid2[idx, :]) * 6
        newtag = np.maximum(
            newtag,
            intriangulation(pial_n, pial_f[:, :3], label_centroid2[idx, :]) * 5,
        )
        if "csf_n" in locals():
            newtag = np.maximum(
                newtag,
                intriangulation(csf_n, csf_f[:, :3], label_centroid2[idx, :]) * 4,
            )
        if "bone_n2" in locals():
            newtag = np.maximum(
                newtag,
                intriangulation(bone_n2, bone_f2[:, :3], label_centroid2[idx, :]) * 3,
            )
        if "no_skin" in locals():
            newtag = np.maximum(
                newtag,
                intriangulation(no_skin, f_skin[:, :3], label_centroid2[idx, :]) * 2,
            )

        newlabel[idx] = newtag
        newlabel = 7 - newlabel
        brain_el[:, 4] = newlabel[brain_el[:, 4] - 1]  # Subtract 1 for 0-based indexing

        break  # Exit the loop after successful completion

    # Relabeling step to remove layered assumptions
    if dorelabel == 1 and ("skull" in tpm and "scalp" in tpm):
        centroid = meshcentroid(brain_n[:, :3], brain_el[:, :4])
        centroid = np.ceil(centroid).astype(int)
        tag = np.zeros(len(brain_el))
        facenb = faceneighbors(brain_el[:, :4])

        for i in range(len(brain_el)):
            cx, cy, cz = centroid[i]
            if (expandedGM[cx, cy, cz] > 0.5) and (brain_el[i, 4] == 2):
                if tpm["scalp"][cx, cy, cz] > 0.5:
                    brain_el[i, 4] = 5
                elif tpm["skull"][cx, cy, cz] > 0.5:
                    brain_el[i, 4] = 4
                else:
                    brain_el[i, 4] = 3
                tag[i] = 1
                for j in range(4):
                    if facenb[i, j] > 0:
                        tag[facenb[i, j] - 1] = 1  # Subtract 1 for 0-based indexing
            elif (expandedCSF[cx, cy, cz] > 0.5) and (brain_el[i, 4] == 3):
                if tpm["scalp"][cx, cy, cz] > 0.5:
                    brain_el[i, 4] = 5
                else:
                    brain_el[i, 4] = 4
                tag[i] = 1
                for j in range(4):
                    if facenb[i, j] > 0:
                        tag[facenb[i, j] - 1] = 1  # Subtract 1 for 0-based indexing
            elif (expandedSkull[cx, cy, cz] > 0.5) and (brain_el[i, 4] == 4):
                brain_el[i, 4] = 5
                tag[i] = 1
                for j in range(4):
                    if facenb[i, j] > 0:
                        tag[facenb[i, j] - 1] = 1  # Subtract 1 for 0-based indexing

        labels = np.zeros((len(brain_el), 4))
        labels2 = np.zeros((len(brain_el), 6))
        for i in range(len(brain_el)):
            for j in range(4):
                if facenb[i, j] > 0:
                    neighbor_idx = facenb[i, j] - 1  # Subtract 1 for 0-based indexing
                    labels[i, j] = brain_el[neighbor_idx, 4]
                    labels2[i, int(labels[i, j])] += 1
                else:
                    labels[i, j] = 0

        max_labels = np.argmax(labels2, axis=1)
        max_counts = np.max(labels2, axis=1)

        for i in range(len(brain_el)):
            if tag[i] == 1:
                if (max_counts[i] > 2) and (brain_el[i, 4] != max_labels[i]):
                    brain_el[i, 4] = max_labels[i]

    brain_el[:, 4] = 6 - brain_el[:, 4]

    # Generate brain_f if requested
    brain_f = np.array([])
    #    if len(sys._getframe().f_back.f_code.co_names) > 2:  # Check if brain_f is requested
    #        brain_f = layersurf(brain_el)

    return brain_n, brain_el, brain_f


def intriangulation(
    vertices: np.ndarray, faces: np.ndarray, testp: np.ndarray, heavytest: int = 0
) -> np.ndarray:
    """
    Test points in 3d whether inside or outside a (closed) triangulation
    Usage: in = intriangulation(vertices, faces, testp, heavytest)

    Arguments (input):
    vertices   - points in 3d as matrix with three columns
    faces      - description of triangles as matrix with three columns.
                Each row contains three indices into the matrix of vertices
                which gives the three cornerpoints of the triangle.
    testp      - points in 3d as matrix with three columns
    heavytest  - int n >= 0. Perform n additional randomized rotation tests.

    IMPORTANT: the set of vertices and faces has to form a watertight surface!

    Arguments (output):
    in - a vector of length size(testp,0), containing 0 and 1.
         in[nr] =  0: testp[nr,:] is outside the triangulation
         in[nr] =  1: testp[nr,:] is inside the triangulation
         in[nr] = -1: unable to decide for testp[nr,:]

    Thanks to Adam A for providing the FEX submission voxelise. The
    algorithms of voxelise form the algorithmic kernel of intriangulation.

    Thanks to Sven to discussions about speed and avoiding problems in
    special cases.

    Author: Johannes Korsawe, heavily based on voxelise from Adam A.
    E-mail: johannes.korsawe@volkswagen.de
    Python conversion: Preserves exact algorithm from MATLAB version
    Release: 1.3
    Release date: 25/09/2013
    """

    # Check number of inputs
    if vertices is None or faces is None or testp is None:
        print("??? Error using ==> intriangulation\nThree input matrices are needed.\n")
        return np.array([])

    if heavytest is None:
        heavytest = 0

    # Check size of inputs
    if vertices.shape[1] != 3 or faces.shape[1] != 3 or testp.shape[1] != 3:
        print(
            "??? Error using ==> intriangulation\nAll input matrices must have three columns.\n"
        )
        return np.array([])

    # Convert faces from 1-based to 0-based indexing when used as indices
    # Note: we keep faces array unchanged, but subtract 1 when indexing
    ipmax = np.max(faces)
    zerofound = np.any(faces == 0)
    if ipmax > vertices.shape[0] or zerofound:
        print(
            "??? Error using ==> intriangulation\nThe triangulation data is defect. use trisurf(faces,vertices[:,0],vertices[:,1],vertices[:,2]) for test of deficiency.\n"
        )
        return np.array([])

    # Loop for heavytest
    inreturn = np.zeros((testp.shape[0], 1))
    VER = vertices.copy()
    TESTP = testp.copy()

    for n in range(1, heavytest + 2):  # MATLAB: for n = 1:heavytest + 1

        # Randomize
        if n > 1:
            v = np.random.rand(1, 3)
            D = rotmatrix(v / np.linalg.norm(v), np.random.rand() * 180 / np.pi)
            vertices = VER @ D
            testp = TESTP @ D
        else:
            vertices = VER.copy()

        # Preprocessing data
        meshXYZ = np.zeros((faces.shape[0], 3, 3))
        for loop in range(3):
            # Subtract 1 when using faces as indices (convert from 1-based to 0-based)
            meshXYZ[:, :, loop] = vertices[faces[:, loop] - 1, :]

        # Basic idea (ingenious from FeX-submission voxelise):
        # If point is inside, it will cross the triangulation an uneven number of times in each direction (x, -x, y, -y, z, -z).

        # The function VOXELISEinternal is about 98% identical to its version inside voxelise.m.
        # This includes the elaborate comments. Thanks to Adam A!

        # z-direction:
        # initialization of results and correction list
        in_result, cl = VOXELISEinternal(testp[:, 0], testp[:, 1], testp[:, 2], meshXYZ)

        # x-direction:
        # has only to be done for those points, that were not determinable in the first step --> cl
        if len(cl) > 0:
            in2, cl2 = VOXELISEinternal(
                testp[cl, 1], testp[cl, 2], testp[cl, 0], meshXYZ[:, [1, 2, 0], :]
            )
            # Use results of x-direction that determined "inside"
            in_result[cl[in2 == 1]] = 1
            # remaining indices with unclear result
            cl = cl[cl2]

        # y-direction:
        # has only to be done for those points, that were not determinable in the first and second step --> cl
        if len(cl) > 0:
            in3, cl3 = VOXELISEinternal(
                testp[cl, 2], testp[cl, 0], testp[cl, 1], meshXYZ[:, [2, 0, 1], :]
            )

            # Use results of y-direction that determined "inside"
            in_result[cl[in3 == 1]] = 1
            # remaining indices with unclear result
            cl = cl[cl3]

        # Mark those indices, where all three tests have failed
        in_result[cl] = -1

        if n == 1:
            inreturn = in_result.copy()  # Starting guess
        else:
            # if ALWAYS inside, use as inside!
            # I = find(inreturn ~= in);
            # inreturn(I(in(I)==0)) = 0;

            # if AT LEAST ONCE inside, use as inside!
            I = np.where(inreturn.flatten() != in_result.flatten())[0]
            inreturn.flatten()[I[in_result.flatten()[I] == 1]] = 1

    return inreturn.flatten().astype(int)


def VOXELISEinternal(
    testx: np.ndarray, testy: np.ndarray, testz: np.ndarray, meshXYZ: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Internal voxelization function - exact port from MATLAB voxelise

    This function is about 98% identical to its version inside voxelise.m.
    This includes the elaborate comments. Thanks to Adam A!
    """

    # Prepare logical array to hold the logical data:
    OUTPUT = np.zeros((testx.shape[0], 1))

    # Identify the min and max x,y coordinates of the mesh:
    meshZmin = np.min(meshXYZ[:, 2, :])
    meshZmax = np.max(meshXYZ[:, 2, :])

    # Identify the min and max x,y,z coordinates of each facet:
    meshXYZmin = np.min(meshXYZ, axis=2)
    meshXYZmax = np.max(meshXYZ, axis=2)

    # ======================================================
    # TURN OFF DIVIDE-BY-ZERO WARNINGS
    # ======================================================
    # This prevents the Y1predicted, Y2predicted, Y3predicted and YRpredicted
    # calculations creating divide-by-zero warnings. Suppressing these warnings
    # doesn't affect the code, because only the sign of the result is important.
    # That is, 'Inf' and '-Inf' results are ok.
    # The warning will be returned to its original state at the end of the code.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        # ======================================================
        # START COMPUTATION
        # ======================================================

        correctionLIST = (
            []
        )  # Prepare to record all rays that fail the voxelisation. This array is built on-the-fly, but since
        # it ought to be relatively small should not incur too much of a speed penalty.

        # Loop through each testpoint.
        # The testpoint-array will be tested by passing rays in the z-direction through
        # each x,y coordinate of the testpoints, and finding the locations where the rays cross the mesh.
        facetCROSSLIST = np.zeros(1000, dtype=int)  # uses countindex: nf
        nm = meshXYZmin.shape[0]

        for loop in range(len(OUTPUT)):

            nf = 0
            # - 1a - Find which mesh facets could possibly be crossed by the ray:
            # possibleCROSSLISTy = find( meshXYZmin(:,2)<=testy(loop) & meshXYZmax(:,2)>=testy(loop) );

            # - 1b - Find which mesh facets could possibly be crossed by the ray:
            # possibleCROSSLIST = possibleCROSSLISTy( meshXYZmin(possibleCROSSLISTy,1)<=testx(loop) & meshXYZmax(possibleCROSSLISTy,1)>=testx(loop) );

            # Do - 1a - and - 1b - faster
            possibleCROSSLISTy = np.where(
                (testy[loop] - meshXYZmin[:, 1]) * (meshXYZmax[:, 1] - testy[loop]) > 0
            )[0]
            possibleCROSSLISTx = (testx[loop] - meshXYZmin[possibleCROSSLISTy, 0]) * (
                meshXYZmax[possibleCROSSLISTy, 0] - testx[loop]
            ) > 0
            possibleCROSSLIST = possibleCROSSLISTy[possibleCROSSLISTx]

            if (
                len(possibleCROSSLIST) != 0
            ):  # Only continue the analysis if some nearby facets were actually identified

                # - 2 - For each facet, check if the ray really does cross the facet rather than just passing it close-by:

                # GENERAL METHOD:
                # 1. Take each edge of the facet in turn.
                # 2. Find the position of the opposing vertex to that edge.
                # 3. Find the position of the ray relative to that edge.
                # 4. Check if ray is on the same side of the edge as the opposing vertex.
                # 5. If this is true for all three edges, then the ray definitely passes through the facet.
                #
                # NOTES:
                # 1. If the ray crosses exactly on an edge, this is counted as crossing the facet.
                # 2. If a ray crosses exactly on a vertex, this is also taken into account.

                for loopCHECKFACET in possibleCROSSLIST:

                    # Check if ray crosses the facet. This method is much (>>10 times) faster than using the built-in function 'inpolygon'.
                    # Taking each edge of the facet in turn, check if the ray is on the same side as the opposing vertex. If so, let testVn=1

                    Y1predicted = meshXYZ[loopCHECKFACET, 1, 1] - (
                        (meshXYZ[loopCHECKFACET, 1, 1] - meshXYZ[loopCHECKFACET, 1, 2])
                        * (
                            meshXYZ[loopCHECKFACET, 0, 1]
                            - meshXYZ[loopCHECKFACET, 0, 0]
                        )
                        / (
                            meshXYZ[loopCHECKFACET, 0, 1]
                            - meshXYZ[loopCHECKFACET, 0, 2]
                        )
                    )
                    YRpredicted = meshXYZ[loopCHECKFACET, 1, 1] - (
                        (meshXYZ[loopCHECKFACET, 1, 1] - meshXYZ[loopCHECKFACET, 1, 2])
                        * (meshXYZ[loopCHECKFACET, 0, 1] - testx[loop])
                        / (
                            meshXYZ[loopCHECKFACET, 0, 1]
                            - meshXYZ[loopCHECKFACET, 0, 2]
                        )
                    )

                    if (
                        (
                            Y1predicted > meshXYZ[loopCHECKFACET, 1, 0]
                            and YRpredicted > testy[loop]
                        )
                        or (
                            Y1predicted < meshXYZ[loopCHECKFACET, 1, 0]
                            and YRpredicted < testy[loop]
                        )
                        or (
                            meshXYZ[loopCHECKFACET, 1, 1]
                            - meshXYZ[loopCHECKFACET, 1, 2]
                        )
                        * (meshXYZ[loopCHECKFACET, 0, 1] - testx[loop])
                        == 0
                    ):
                        # testV1 = 1;   # The ray is on the same side of the 2-3 edge as the 1st vertex.
                        pass
                    else:
                        # testV1 = 0;   # The ray is on the opposite side of the 2-3 edge to the 1st vertex.
                        # As the check is for ALL three checks to be true, we can continue here, if only one check fails
                        continue

                    Y2predicted = meshXYZ[loopCHECKFACET, 1, 2] - (
                        (meshXYZ[loopCHECKFACET, 1, 2] - meshXYZ[loopCHECKFACET, 1, 0])
                        * (
                            meshXYZ[loopCHECKFACET, 0, 2]
                            - meshXYZ[loopCHECKFACET, 0, 1]
                        )
                        / (
                            meshXYZ[loopCHECKFACET, 0, 2]
                            - meshXYZ[loopCHECKFACET, 0, 0]
                        )
                    )
                    YRpredicted = meshXYZ[loopCHECKFACET, 1, 2] - (
                        (meshXYZ[loopCHECKFACET, 1, 2] - meshXYZ[loopCHECKFACET, 1, 0])
                        * (meshXYZ[loopCHECKFACET, 0, 2] - testx[loop])
                        / (
                            meshXYZ[loopCHECKFACET, 0, 2]
                            - meshXYZ[loopCHECKFACET, 0, 0]
                        )
                    )

                    if (
                        (
                            Y2predicted > meshXYZ[loopCHECKFACET, 1, 1]
                            and YRpredicted > testy[loop]
                        )
                        or (
                            Y2predicted < meshXYZ[loopCHECKFACET, 1, 1]
                            and YRpredicted < testy[loop]
                        )
                        or (
                            meshXYZ[loopCHECKFACET, 1, 2]
                            - meshXYZ[loopCHECKFACET, 1, 0]
                        )
                        * (meshXYZ[loopCHECKFACET, 0, 2] - testx[loop])
                        == 0
                    ):
                        # testV2 = 1;   # The ray is on the same side of the 3-1 edge as the 2nd vertex.
                        pass
                    else:
                        # testV2 = 0;   # The ray is on the opposite side of the 3-1 edge to the 2nd vertex.
                        # As the check is for ALL three checks to be true, we can continue here, if only one check fails
                        continue

                    Y3predicted = meshXYZ[loopCHECKFACET, 1, 0] - (
                        (meshXYZ[loopCHECKFACET, 1, 0] - meshXYZ[loopCHECKFACET, 1, 1])
                        * (
                            meshXYZ[loopCHECKFACET, 0, 0]
                            - meshXYZ[loopCHECKFACET, 0, 2]
                        )
                        / (
                            meshXYZ[loopCHECKFACET, 0, 0]
                            - meshXYZ[loopCHECKFACET, 0, 1]
                        )
                    )
                    YRpredicted = meshXYZ[loopCHECKFACET, 1, 0] - (
                        (meshXYZ[loopCHECKFACET, 1, 0] - meshXYZ[loopCHECKFACET, 1, 1])
                        * (meshXYZ[loopCHECKFACET, 0, 0] - testx[loop])
                        / (
                            meshXYZ[loopCHECKFACET, 0, 0]
                            - meshXYZ[loopCHECKFACET, 0, 1]
                        )
                    )

                    if (
                        (
                            Y3predicted > meshXYZ[loopCHECKFACET, 1, 2]
                            and YRpredicted > testy[loop]
                        )
                        or (
                            Y3predicted < meshXYZ[loopCHECKFACET, 1, 2]
                            and YRpredicted < testy[loop]
                        )
                        or (
                            meshXYZ[loopCHECKFACET, 1, 0]
                            - meshXYZ[loopCHECKFACET, 1, 1]
                        )
                        * (meshXYZ[loopCHECKFACET, 0, 0] - testx[loop])
                        == 0
                    ):
                        # testV3 = 1;   # The ray is on the same side of the 1-2 edge as the 3rd vertex.
                        pass
                    else:
                        # testV3 = 0;   # The ray is on the opposite side of the 1-2 edge to the 3rd vertex.
                        # As the check is for ALL three checks to be true, we can continue here, if only one check fails
                        continue

                    nf = nf + 1
                    if nf <= len(facetCROSSLIST):
                        facetCROSSLIST[nf - 1] = loopCHECKFACET
                    else:
                        # Expand array if needed
                        facetCROSSLIST = np.concatenate(
                            [facetCROSSLIST, np.zeros(1000, dtype=int)]
                        )
                        facetCROSSLIST[nf - 1] = loopCHECKFACET

                # Use only values ~=0
                facetCROSSLIST = facetCROSSLIST[:nf]

                # - 3 - Find the z coordinate of the locations where the ray crosses each facet:
                gridCOzCROSS = np.zeros(nf)
                for i, loopFINDZ in enumerate(facetCROSSLIST):

                    # METHOD:
                    # 1. Define the equation describing the plane of the facet. For a
                    # more detailed outline of the maths, see:
                    # http://local.wasp.uwa.edu.au/~pbourke/geometry/planeeq/
                    #    Ax + By + Cz + D = 0
                    #    where  A = y1 (z2 - z3) + y2 (z3 - z1) + y3 (z1 - z2)
                    #           B = z1 (x2 - x3) + z2 (x3 - x1) + z3 (x1 - x2)
                    #           C = x1 (y2 - y3) + x2 (y3 - y1) + x3 (y1 - y2)
                    #           D = - x1 (y2 z3 - y3 z2) - x2 (y3 z1 - y1 z3) - x3 (y1 z2 - y2 z1)
                    # 2. For the x and y coordinates of the ray, solve these equations to find the z coordinate in this plane.

                    planecoA = (
                        meshXYZ[loopFINDZ, 1, 0]
                        * (meshXYZ[loopFINDZ, 2, 1] - meshXYZ[loopFINDZ, 2, 2])
                        + meshXYZ[loopFINDZ, 1, 1]
                        * (meshXYZ[loopFINDZ, 2, 2] - meshXYZ[loopFINDZ, 2, 0])
                        + meshXYZ[loopFINDZ, 1, 2]
                        * (meshXYZ[loopFINDZ, 2, 0] - meshXYZ[loopFINDZ, 2, 1])
                    )

                    planecoB = (
                        meshXYZ[loopFINDZ, 2, 0]
                        * (meshXYZ[loopFINDZ, 0, 1] - meshXYZ[loopFINDZ, 0, 2])
                        + meshXYZ[loopFINDZ, 2, 1]
                        * (meshXYZ[loopFINDZ, 0, 2] - meshXYZ[loopFINDZ, 0, 0])
                        + meshXYZ[loopFINDZ, 2, 2]
                        * (meshXYZ[loopFINDZ, 0, 0] - meshXYZ[loopFINDZ, 0, 1])
                    )

                    planecoC = (
                        meshXYZ[loopFINDZ, 0, 0]
                        * (meshXYZ[loopFINDZ, 1, 1] - meshXYZ[loopFINDZ, 1, 2])
                        + meshXYZ[loopFINDZ, 0, 1]
                        * (meshXYZ[loopFINDZ, 1, 2] - meshXYZ[loopFINDZ, 1, 0])
                        + meshXYZ[loopFINDZ, 0, 2]
                        * (meshXYZ[loopFINDZ, 1, 0] - meshXYZ[loopFINDZ, 1, 1])
                    )

                    planecoD = (
                        -meshXYZ[loopFINDZ, 0, 0]
                        * (
                            meshXYZ[loopFINDZ, 1, 1] * meshXYZ[loopFINDZ, 2, 2]
                            - meshXYZ[loopFINDZ, 1, 2] * meshXYZ[loopFINDZ, 2, 1]
                        )
                        - meshXYZ[loopFINDZ, 0, 1]
                        * (
                            meshXYZ[loopFINDZ, 1, 2] * meshXYZ[loopFINDZ, 2, 0]
                            - meshXYZ[loopFINDZ, 1, 0] * meshXYZ[loopFINDZ, 2, 2]
                        )
                        - meshXYZ[loopFINDZ, 0, 2]
                        * (
                            meshXYZ[loopFINDZ, 1, 0] * meshXYZ[loopFINDZ, 2, 1]
                            - meshXYZ[loopFINDZ, 1, 1] * meshXYZ[loopFINDZ, 2, 0]
                        )
                    )

                    if abs(planecoC) < 1e-14:
                        planecoC = 0

                    gridCOzCROSS[i] = (
                        -planecoD - planecoA * testx[loop] - planecoB * testy[loop]
                    ) / planecoC

                if len(gridCOzCROSS) == 0:
                    continue

                # Remove values of gridCOzCROSS which are outside of the mesh limits (including a 1e-12 margin for error).
                gridCOzCROSS = gridCOzCROSS[
                    (gridCOzCROSS >= meshZmin - 1e-12)
                    & (gridCOzCROSS <= meshZmax + 1e-12)
                ]

                # Round gridCOzCROSS to remove any rounding errors, and take only the unique values:
                gridCOzCROSS = np.round(gridCOzCROSS * 1e10) / 1e10

                # Replacement of the call to unique (gridCOzCROSS = unique(gridCOzCROSS);) by the following line:
                tmp = np.sort(gridCOzCROSS)
                if len(tmp) > 1:
                    I = np.concatenate(([True], tmp[1:] - tmp[:-1] != 0))
                    gridCOzCROSS = tmp[I]
                else:
                    gridCOzCROSS = tmp

                # - 4 - Label as being inside the mesh all the voxels that the ray passes through after crossing one facet before crossing another facet:

                if (
                    len(gridCOzCROSS) % 2 == 0
                ):  # Only rays which cross an even number of facets are voxelised

                    for loopASSIGN in range(1, len(gridCOzCROSS) // 2 + 1):
                        voxelsINSIDE = (
                            testz[loop] > gridCOzCROSS[2 * loopASSIGN - 2]
                            and testz[loop] < gridCOzCROSS[2 * loopASSIGN - 1]
                        )
                        OUTPUT[loop] = int(voxelsINSIDE)
                        if voxelsINSIDE:
                            break

                elif (
                    len(gridCOzCROSS) != 0
                ):  # Remaining rays which meet the mesh in some way are not voxelised, but are labelled for correction later.
                    correctionLIST.append(loop)

    # ======================================================
    # RESTORE DIVIDE-BY-ZERO WARNINGS TO THE ORIGINAL STATE
    # ======================================================
    # (handled automatically by context manager)

    # J.Korsawe: A correction is not possible as the testpoints need not to be
    #            ordered in any way.
    #            voxelise contains a correction algorithm which is appended here
    #            without changes in syntax.

    return OUTPUT.flatten().astype(int), np.array(correctionLIST, dtype=int)


def rotmatrix(v: np.ndarray, deg: float) -> np.ndarray:
    """
    Calculate the rotation matrix about v by deg degrees
    """

    deg = deg / 180 * np.pi
    if deg != 0:
        v = v / np.linalg.norm(v)
        v1 = v[0, 0] if v.ndim > 1 else v[0]
        v2 = v[0, 1] if v.ndim > 1 else v[1]
        v3 = v[0, 2] if v.ndim > 1 else v[2]
        ca = np.cos(deg)
        sa = np.sin(deg)
        D = np.array(
            [
                [
                    ca + v1 * v1 * (1 - ca),
                    v1 * v2 * (1 - ca) - v3 * sa,
                    v1 * v3 * (1 - ca) + v2 * sa,
                ],
                [
                    v2 * v1 * (1 - ca) + v3 * sa,
                    ca + v2 * v2 * (1 - ca),
                    v2 * v3 * (1 - ca) - v1 * sa,
                ],
                [
                    v3 * v1 * (1 - ca) - v2 * sa,
                    v3 * v2 * (1 - ca) + v1 * sa,
                    ca + v3 * v3 * (1 - ca),
                ],
            ]
        )
    else:
        D = np.eye(3)

    return D


def brain1020(
    node: np.ndarray,
    face: np.ndarray,
    initpoints: Optional[Union[Dict, np.ndarray]] = None,
    perc1: int = 10,
    perc2: int = 20,
    **kwargs,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray]:
    """
    Compute 10-20-like scalp landmarks with user-specified density on a head mesh

    Author: Qianqian Fang (q.fang at neu.edu)
    Python conversion: Preserves exact algorithm from MATLAB version

    Parameters:
    -----------
    node : ndarray
        Full head mesh node list
    face : ndarray
        Full head mesh element list- a 3-column array defines face list
        for the exterior (scalp) surface; a 4-column array defines the
        tetrahedral mesh of the full head.
    initpoints : dict or ndarray, optional
        One can provide the 3-D coordinates of the below
        5 landmarks: nz, iz, lpa, rpa, cz0 (cz0 is the initial guess of cz)
        initpoints can be a dict with the above landmark names
        as keys, or a 5x3 array defining these points in the above
        mentioned order (one can use the output landmarks as initpoints)
    perc1 : int, optional
        The percentage of geodesic distance towards the rim of
        the landmarks; this is the first number of the 10-20 or 10-10 or
        10-5 systems, in this case, it is 10 (for 10%). Default is 10.
    perc2 : int, optional
        The percentage of geodesic distance towards the center
        of the landmarks; this is the 2nd number of the 10-20 or 10-10 or
        10-5 systems, which are 20, 10, 5, respectively, default is 20
    **kwargs : dict
        Additional options:
        'display' : bool, default=True
            If True, plot landmarks and curves
        'cztol' : float, default=1e-6
            The tolerance for searching cz that bisects saggital and coronal reference curves
        'maxcziter' : int, default=10
            The maximum number of iterations to update cz to bisect both cm and sm curves
        'baseplane' : bool, default=True
            If True, create the reference curves along the primary control points (nz,iz,lpa,rpa)
        'minangle' : float, default=0
            If set to a positive number, this specifies the minimum angle (radian) between
            adjacent segments in the reference curves to avoid sharp turns

    Returns:
    --------
    landmarks : dict
        A dictionary storing all computed landmarks. The keys include two sections:
        1) 'nz','iz','lpa','rpa','cz': individual 3D positions defining
           the 5 principle reference points: nasion (nz), inion (iz),
           left-pre-auricular-point (lpa), right-pre-auricular-point
           (rpa) and vertex (cz) - cz is updated from initpoints to bisect
           the saggital and coronal ref. curves.
        2) landmarks along specific cross-sections, each cross section
           may contain more than 1 position.
    curves : dict
        A dictionary storing all computed cross-section curves. The
        keys are named similarly to landmarks, except that
        landmarks stores the 10-? points, and curves stores the
        detailed cross-sectional curves
    initpoints : ndarray
        A 5x3 array storing the principle reference points in the
        orders of 'nz','iz','lpa','rpa','cz'

    Notes:
    ------
    This function is part of brain2mesh toolbox (http://mcx.space/brain2mesh)
    License: GPL v3 or later, see LICENSE.txt for details

    This function requires a pre-installed Iso2Mesh Toolbox
    Download URL: http://github.com/fangq/iso2mesh
    Website: http://iso2mesh.sf.net
    """

    if node is None or face is None:
        raise ValueError("one must provide a head-mesh to call this function")

    if node.size == 0 or face.size == 0 or face.shape[1] <= 2 or node.shape[1] < 3:
        raise ValueError(
            "input node must have 3 columns, face must have at least 3 columns"
        )

    # Parse user options
    showplot = kwargs.get("display", False)
    baseplane = kwargs.get("baseplane", True)
    tol = kwargs.get("cztol", 1e-6)
    dosimplify = kwargs.get("minangle", 0)
    maxcziter = kwargs.get("maxcziter", 10)

    if isinstance(initpoints, list):
        initpoints = np.array(initpoints)

    # Handle case where initpoints has only 3 landmarks (nz, lpa, rpa)
    if initpoints is not None and (
        (isinstance(initpoints, dict) and "iz" not in initpoints)
        or (isinstance(initpoints, np.ndarray) and initpoints.shape[0] == 3)
    ):

        if isinstance(initpoints, dict):
            nz = np.array(initpoints["nz"]).flatten()
            lpa = np.array(initpoints["lpa"]).flatten()
            rpa = np.array(initpoints["rpa"]).flatten()
        else:
            nz = initpoints[0, :]
            lpa = initpoints[1, :]
            rpa = initpoints[2, :]

        # This assumes nz, lpa, rpa, iz are on the same plane to find iz on the head surface
        pa_mid = np.mean([lpa, rpa], axis=0)
        v0 = pa_mid - nz
        iz, e0 = ray2surf(node, face, nz, v0, ">")

        # To find cz, we assume that the vector from iz nz midpoint to cz is perpendicular
        # to the plane defined by nz, lpa, and rpa.
        iznz_mid = (nz + iz) * 0.5
        v0 = np.cross(nz - rpa, lpa - rpa)
        cz, e0 = ray2surf(node, face, iznz_mid, v0, ">")

        if isinstance(initpoints, dict):
            initpoints["iz"] = iz
            initpoints["cz"] = cz
        else:
            initpoints = np.vstack(
                [
                    initpoints[0:1, :],
                    iz.reshape(1, -1),
                    initpoints[1:3, :],
                    cz.reshape(1, -1),
                ]
            )

    # Convert initpoints input to a 5x3 array
    if isinstance(initpoints, dict):
        landmarks = {
            "nz": np.array(initpoints["nz"]).flatten(),
            "iz": np.array(initpoints["iz"]).flatten(),
            "lpa": np.array(initpoints["lpa"]).flatten(),
            "rpa": np.array(initpoints["rpa"]).flatten(),
            "cz": np.array(initpoints["cz"]).flatten(),
        }
        initpoints = np.vstack(
            [
                landmarks["nz"],
                landmarks["iz"],
                landmarks["lpa"],
                landmarks["rpa"],
                landmarks["cz"],
            ]
        )

    # Convert tetrahedral mesh into a surface mesh
    if face.shape[1] >= 4:
        face = volface(face[:, :4])[0]  # Use first 4 columns for tetrahedral faces

    # Remove nodes not located in the surface
    node, face, _ = removeisolatednode(node, face)

    # If initpoints is not sufficient, we need interactive selection
    # For now, we'll require full initpoints or raise an error
    if initpoints is None or initpoints.shape[0] < 5:
        raise ValueError(
            "initpoints must contain 5 landmarks: nz, iz, lpa, rpa, cz. "
            + "Interactive selection not implemented in Python version."
        )

    if showplot:
        print("Initial points:")
        print(initpoints)

    # Save input initpoints to landmarks output, cz is not finalized
    if initpoints.shape[0] >= 5:
        landmarks = {
            "nz": initpoints[0, :],
            "iz": initpoints[1, :],
            "lpa": initpoints[2, :],
            "rpa": initpoints[3, :],
            "cz": initpoints[4, :],
        }

    # At this point, initpoints contains {nz, iz, lpa, rpa, cz0}
    # Plot the head mesh
    if showplot:
        hh = plotmesh(node, face, alpha=0.6, color="wheat", linewidth=0)
        ax = hh["ax"][0]
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Brain 10-20 System Landmarks")

    lastcz = np.array([1, 1, 1]) * np.inf
    cziter = 0
    curves = {}

    # Find cz that bisects cm and sm curves within a tolerance, using UI 10-10 approach
    while np.linalg.norm(initpoints[4, :] - lastcz) > tol and cziter < maxcziter:

        # Step 1: nz, iz and cz0 to determine saggital reference curve
        nsagg = slicesurf(node, face, initpoints[[0, 1, 4], :])

        # Step 1.1: get cz1 as the mid-point between iz and nz
        slen, nsagg, _ = polylinelen(
            nsagg, initpoints[0, :], initpoints[1, :], initpoints[4, :]
        )
        if dosimplify:
            nsagg, slen = polylinesimplify(nsagg, dosimplify)
        idx, weight, cz = polylineinterp(slen, np.sum(slen) * 0.5, nsagg)
        initpoints[4, :] = cz[0, :]

        # Step 1.2: lpa, rpa and cz1 to determine coronal reference curve, update cz1
        curves["cm"] = slicesurf(node, face, initpoints[[2, 3, 4], :])
        len_cm, curves["cm"], _ = polylinelen(
            curves["cm"], initpoints[2, :], initpoints[3, :], initpoints[4, :]
        )
        if dosimplify:
            curves["cm"], len_cm = polylinesimplify(curves["cm"], dosimplify)
        idx, weight, coro = polylineinterp(len_cm, np.sum(len_cm) * 0.5, curves["cm"])
        lastcz = initpoints[4, :].copy()
        initpoints[4, :] = coro[0, :]
        cziter += 1
        if showplot:
            print(
                f"cz iteration {cziter} error {np.linalg.norm(initpoints[4, :] - lastcz):.2e}"
            )

    # Set the finalized cz to output
    landmarks["cz"] = initpoints[4, :].copy()

    if showplot:
        print("Finalized points:")
        print(initpoints)

    # Step 2: subdivide saggital (sm) and coronal (cm) ref curves
    perc_range = np.arange(perc1, 100 - perc1 + perc2, perc2) * 0.01
    idx, weight, coro = polylineinterp(
        len_cm, np.sum(len_cm) * perc_range, curves["cm"]
    )
    landmarks["cm"] = coro  # t7, c3, cz, c4, t8

    curves["sm"] = slicesurf(node, face, initpoints[[0, 1, 4], :])
    slen, curves["sm"], _ = polylinelen(
        curves["sm"], initpoints[0, :], initpoints[1, :], initpoints[4, :]
    )
    if dosimplify:
        curves["sm"], slen = polylinesimplify(curves["sm"], dosimplify)
    idx, weight, sagg = polylineinterp(slen, np.sum(slen) * perc_range, curves["sm"])
    landmarks["sm"] = sagg  # fpz, fz, cz, pz, oz

    # Step 3: fpz, t7 and oz to determine left 10% axial reference curve
    landmarks["aal"], curves["aal"], landmarks["apl"], curves["apl"] = slicesurf3(
        node,
        face,
        landmarks["sm"][0, :],
        landmarks["cm"][0, :],
        landmarks["sm"][-1, :],
        perc2 * 2,
    )

    # Step 4: fpz, t8 and oz to determine right 10% axial reference curve
    landmarks["aar"], curves["aar"], landmarks["apr"], curves["apr"] = slicesurf3(
        node,
        face,
        landmarks["sm"][0, :],
        landmarks["cm"][-1, :],
        landmarks["sm"][-1, :],
        perc2 * 2,
    )

    # Show plots of the landmarks
    if showplot:
        ax.plot(
            curves["sm"][:, 0],
            curves["sm"][:, 1],
            curves["sm"][:, 2],
            "r-",
            linewidth=2,
            label="Sagittal",
        )
        ax.plot(
            curves["cm"][:, 0],
            curves["cm"][:, 1],
            curves["cm"][:, 2],
            "g-",
            linewidth=2,
            label="Coronal",
        )
        ax.plot(
            curves["aal"][:, 0],
            curves["aal"][:, 1],
            curves["aal"][:, 2],
            "k-",
            linewidth=1,
        )
        ax.plot(
            curves["aar"][:, 0],
            curves["aar"][:, 1],
            curves["aar"][:, 2],
            "k-",
            linewidth=1,
        )
        ax.plot(
            curves["apl"][:, 0],
            curves["apl"][:, 1],
            curves["apl"][:, 2],
            "b-",
            linewidth=1,
        )
        ax.plot(
            curves["apr"][:, 0],
            curves["apr"][:, 1],
            curves["apr"][:, 2],
            "b-",
            linewidth=1,
        )

        ax.scatter(
            landmarks["sm"][:, 0],
            landmarks["sm"][:, 1],
            landmarks["sm"][:, 2],
            c="red",
            s=50,
            marker="o",
            label="Sagittal landmarks",
        )
        ax.scatter(
            landmarks["cm"][:, 0],
            landmarks["cm"][:, 1],
            landmarks["cm"][:, 2],
            c="green",
            s=50,
            marker="o",
            label="Coronal landmarks",
        )
        ax.scatter(
            landmarks["aal"][:, 0],
            landmarks["aal"][:, 1],
            landmarks["aal"][:, 2],
            c="black",
            s=50,
            marker="o",
        )
        ax.scatter(
            landmarks["aar"][:, 0],
            landmarks["aar"][:, 1],
            landmarks["aar"][:, 2],
            c="magenta",
            s=50,
            marker="o",
        )
        ax.scatter(
            landmarks["apl"][:, 0],
            landmarks["apl"][:, 1],
            landmarks["apl"][:, 2],
            c="black",
            s=50,
            marker="o",
        )
        ax.scatter(
            landmarks["apr"][:, 0],
            landmarks["apr"][:, 1],
            landmarks["apr"][:, 2],
            c="magenta",
            s=50,
            marker="o",
        )

    # Step 5: computing all anterior coronal cuts, moving away from the medial cut (cm) toward frontal
    idxcz = closestnode(landmarks["sm"], landmarks["cz"])[0]
    skipcount = int(np.floor(10 / perc2))

    for i in range(
        1, landmarks["aal"].shape[0] - skipcount + 1
    ):  # MATLAB: for i = 1:size(landmarks.aal, 1) - skipcount
        step = (
            (perc2 * 25)
            * 0.1
            * (
                1
                + (
                    (perc2 < 20 + perc2 < 10)
                    and i == landmarks["aal"].shape[0] - skipcount
                )
            )
        )
        # Subtract 1 from idxcz when using as index (convert from 1-based to 0-based)
        cal_landmarks, leftpart, car_landmarks, rightpart = slicesurf3(
            node,
            face,
            landmarks["aal"][i - 1, :],
            landmarks["sm"][idxcz - 1 - i, :],
            landmarks["aar"][i - 1, :],
            step,
        )

        landmarks[f"cal_{i}"] = cal_landmarks
        landmarks[f"car_{i}"] = car_landmarks

        if showplot:
            ax.plot(leftpart[:, 0], leftpart[:, 1], leftpart[:, 2], "k-", linewidth=1)
            ax.plot(
                rightpart[:, 0], rightpart[:, 1], rightpart[:, 2], "k-", linewidth=1
            )
            ax.scatter(
                cal_landmarks[:, 0],
                cal_landmarks[:, 1],
                cal_landmarks[:, 2],
                c="yellow",
                s=30,
                marker="o",
            )
            ax.scatter(
                car_landmarks[:, 0],
                car_landmarks[:, 1],
                car_landmarks[:, 2],
                c="cyan",
                s=30,
                marker="o",
            )

    # Step 6: computing all posterior coronal cuts, moving away from the medial cut (cm) toward occipital
    for i in range(
        1, landmarks["apl"].shape[0] - skipcount + 1
    ):  # MATLAB: for i = 1:size(landmarks.apl, 1) - skipcount
        step = (
            (perc2 * 25)
            * 0.1
            * (
                1
                + (
                    (perc2 < 20 + perc2 < 10)
                    and i == landmarks["apl"].shape[0] - skipcount
                )
            )
        )
        # Subtract 1 from idxcz when using as index (convert from 1-based to 0-based)
        cpl_landmarks, leftpart, cpr_landmarks, rightpart = slicesurf3(
            node,
            face,
            landmarks["apl"][i - 1, :],
            landmarks["sm"][idxcz - 1 + i, :],
            landmarks["apr"][i - 1, :],
            step,
        )

        landmarks[f"cpl_{i}"] = cpl_landmarks
        landmarks[f"cpr_{i}"] = cpr_landmarks

        if showplot:
            ax.plot(leftpart[:, 0], leftpart[:, 1], leftpart[:, 2], "k-", linewidth=1)
            ax.plot(
                rightpart[:, 0], rightpart[:, 1], rightpart[:, 2], "k-", linewidth=1
            )
            ax.scatter(
                cpl_landmarks[:, 0],
                cpl_landmarks[:, 1],
                cpl_landmarks[:, 2],
                c="yellow",
                s=30,
                marker="o",
            )
            ax.scatter(
                cpr_landmarks[:, 0],
                cpr_landmarks[:, 1],
                cpr_landmarks[:, 2],
                c="cyan",
                s=30,
                marker="o",
            )

    # Step 7: create the axial cuts across principle ref. points: left: nz, lpa, iz, right: nz, rpa, iz
    if baseplane and perc2 <= 10:
        (
            landmarks["paal"],
            curves["paal"],
            landmarks["papl"],
            curves["papl"],
        ) = slicesurf3(
            node, face, landmarks["nz"], landmarks["lpa"], landmarks["iz"], perc2 * 2
        )
        (
            landmarks["paar"],
            curves["paar"],
            landmarks["papr"],
            curves["papr"],
        ) = slicesurf3(
            node, face, landmarks["nz"], landmarks["rpa"], landmarks["iz"], perc2 * 2
        )

        if showplot:
            ax.plot(
                curves["paal"][:, 0],
                curves["paal"][:, 1],
                curves["paal"][:, 2],
                "k-",
                linewidth=1,
            )
            ax.plot(
                curves["paar"][:, 0],
                curves["paar"][:, 1],
                curves["paar"][:, 2],
                "k-",
                linewidth=1,
            )
            ax.plot(
                curves["papl"][:, 0],
                curves["papl"][:, 1],
                curves["papl"][:, 2],
                "k-",
                linewidth=1,
            )
            ax.plot(
                curves["papr"][:, 0],
                curves["papr"][:, 1],
                curves["papr"][:, 2],
                "k-",
                linewidth=1,
            )

            ax.scatter(
                landmarks["paal"][:, 0],
                landmarks["paal"][:, 1],
                landmarks["paal"][:, 2],
                c="yellow",
                s=30,
                marker="o",
            )
            ax.scatter(
                landmarks["papl"][:, 0],
                landmarks["papl"][:, 1],
                landmarks["papl"][:, 2],
                c="cyan",
                s=30,
                marker="o",
            )
            ax.scatter(
                landmarks["paar"][:, 0],
                landmarks["paar"][:, 1],
                landmarks["paar"][:, 2],
                c="yellow",
                s=30,
                marker="o",
            )
            ax.scatter(
                landmarks["papr"][:, 0],
                landmarks["papr"][:, 1],
                landmarks["papr"][:, 2],
                c="cyan",
                s=30,
                marker="o",
            )

    return landmarks, curves, initpoints


def label2tpm(
    vol: np.ndarray, names: Optional[Union[List[str], Dict[int, str]]] = None
) -> Dict[str, np.ndarray]:
    """
    Converting a multi-label volume to binary tissue probabilistic maps (TPMs)

    Author: Qianqian Fang (q.fang at neu.edu)
    Python conversion: Preserves exact algorithm from MATLAB version

    Parameters:
    -----------
    vol : ndarray
        A 2-D or 3-D array of integer values.
    names : list or dict, optional
        A list of strings defining the names of each label,
        alternatively, a dictionary with label (integer) as
        the key and name as the value. The names can be a subset of all
        labels.

    Returns:
    --------
    seg : dict
        A dictionary with subfields of 3D or 4D uint8 array; the subfield
        names are defined via the optional names input; the unnamed
        labels will be named as 'label_#'.

    Notes:
    ------
    This function is part of brain2mesh toolbox (http://mcx.space/brain2mesh)
    License: GPL v3 or later, see LICENSE.txt for details
    """

    if not np.issubdtype(vol.dtype, np.number):
        raise TypeError("input must be a numerical array")

    # Get unique values excluding 0, equivalent to setdiff(sort(unique(vol(:))), 0)
    vol_flat = vol.flatten(order="F")  # Flatten using Fortran order as requested
    unique_vals = np.unique(vol_flat)
    val = np.setdiff1d(unique_vals, 0)  # Remove 0 from unique values
    val = np.sort(val)  # Ensure sorted order

    # Check if input should be converted to labels first
    if len(val) > vol.size * (0.1**vol.ndim):
        raise ValueError("please convert the input to labels first")

    seg = {}
    for i in range(
        len(val)
    ):  # MATLAB: for i = 1:length(val), Python: for i in range(len(val))
        nm = f"label_{val[i]}"  # sprintf('label_%d', val(i)) equivalent

        if names is not None:
            if isinstance(names, list):
                # MATLAB: if (i <= length(names)), Python: if (i < len(names))
                if i < len(names):
                    nm = names[i]
            elif isinstance(names, dict):
                # MATLAB: elseif (isa(names, 'containers.Map') && isKey(names, val(i)))
                if val[i] in names:
                    nm = names[val[i]]

        # MATLAB: seg.(nm) = uint8(vol == val(i))
        seg[nm] = (vol == val[i]).astype(np.uint8)

    return seg


def tpm2label(
    seg: Union[Dict[str, np.ndarray], List[np.ndarray], np.ndarray],
    segorder: Optional[List[str]] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:
    """
    Converting tissue probabilistic maps (TPMs) to a multi-label volume

    Author: Qianqian Fang (q.fang at neu.edu)
    Python conversion: Preserves exact algorithm from MATLAB version

    Parameters:
    -----------
    seg : dict, list, or ndarray
        A dictionary, a list, or a 3D or 4D array; if seg is a list
        or a dictionary, their elements must be 2D/3D arrays of the same
        sizes;
    segorder : list, optional
        If seg is a dictionary, segorder allows one to assign output
        labels using customized order instead of the creation order

    Returns:
    --------
    vol : ndarray
        A 2-D or 3-D array of the same type/size of the input arrays. The
        label for each voxel is determined by the index to the highest
        value in TPM of the same voxel. If a voxel is a background voxel
        - i.e. zeros for all TPMs, it stays 0
    names : list, optional
        A list storing the names of the labels (if input is a
        dictionary), the first string is the name for label 1, and so on
        (only returned if input is a dictionary)

    Notes:
    ------
    This function is part of brain2mesh toolbox (http://mcx.space/brain2mesh)
    License: GPL v3 or later, see LICENSE.txt for details
    """

    mask = seg
    names = []

    # Handle dictionary input (equivalent to isstruct(seg) in MATLAB)
    if isinstance(seg, dict):
        if segorder is not None:
            # Reorder fields according to segorder (equivalent to orderfields(seg, segorder))
            seg = orderfields_python(seg, segorder)

        # Get field names (equivalent to fieldnames(seg))
        names = list(seg.keys())

        # Extract values using cellfun equivalent
        # MATLAB: mask = cellfun(@(x) seg.(x), names, 'UniformOutput', false);
        mask = [seg[x] for x in names]

    # Handle list input (equivalent to iscell(mask) in MATLAB)
    if isinstance(mask, list):
        # Concatenate along new dimension (equivalent to cat(ndims(mask{1}) + 1, mask{:}))
        if len(mask) > 0:
            # Get number of dimensions of first element
            ndims_first = mask[0].ndim
            # Concatenate along the next dimension
            mask = np.stack(mask, axis=ndims_first)
        else:
            mask = np.array([])

    # Check if input is numeric (equivalent to isnumeric(mask) in MATLAB)
    if not np.issubdtype(mask.dtype, np.number):
        raise TypeError(
            "input must be a list/dict array with numeric elements of matching dimensions"
        )

    # Find maximum values and indices along last dimension
    # MATLAB: [newmask, vol] = max(mask, [], ndims(mask));
    last_dim = (
        mask.ndim - 1
    )  # ndims(mask) in MATLAB is mask.ndim in Python, but for axis we need ndim-1
    newmask = np.max(mask, axis=last_dim)
    vol = (
        np.argmax(mask, axis=last_dim) + 1
    )  # +1 to convert from 0-based to 1-based indexing

    # Set background voxels to 0 (equivalent to vol .* (sum(mask, ndims(mask)) > 0))
    background_mask = np.sum(mask, axis=last_dim) > 0
    vol = vol * background_mask.astype(vol.dtype)

    # Return results based on input type
    if isinstance(seg, dict):
        return vol, names
    else:
        return vol


def orderfields_python(struct_dict, order: List[str]):
    """
    Python equivalent of MATLAB's orderfields function

    Reorders dictionary fields according to specified order.

    Parameters:
    -----------
    struct_dict : dict
        Dictionary to reorder
    order : list
        List specifying the desired field order

    Returns:
    --------
    ordered_dict : dict
        Dictionary with fields reordered according to order
    """

    ordered_dict = OrderedDict()

    # First add fields in the specified order
    for field_name in order:
        if field_name in struct_dict:
            ordered_dict[field_name] = struct_dict[field_name]

    # Then add any remaining fields that weren't in the order list
    for field_name, field_value in struct_dict.items():
        if field_name not in ordered_dict:
            ordered_dict[field_name] = field_value

    return ordered_dict
