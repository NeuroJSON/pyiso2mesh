![](https://iso2mesh.sourceforge.net/images/iso2mesh_2015_banner.png)

# pyiso2mesh - One-liner 3D Surface and Tetrahedral Mesh Generation Toolbox

* **Copyright**: (C) Qianqian Fang (2024-2025) <q.fang at neu.edu>
* **License**: GNU Public License V3 or later
* **Version**: 0.4.0
* **URL**: [https://pypi.org/project/iso2mesh/](https://pypi.org/project/iso2mesh/)
* **Homepage**: [https://iso2mesh.sf.net](https://iso2mesh.sf.net)
* **Github**: [https://github.com/NeuroJSON/pyiso2mesh](https://github.com/NeuroJSON/pyiso2mesh)
* **Acknowledgement**: This project is supported by the US National Institute of Health (NIH)
  grants [U24-NS124027](https://reporter.nih.gov/project-details/10308329) and 
  [R01-CA204443](https://reporter.nih.gov/project-details/10982160)


![Python Module](https://github.com/NeuroJSON/pyiso2mesh/actions/workflows/build_all.yml/badge.svg)

## Table of Contents
- [Introduction](#introduction)
- [How to Install](#how-to-install)
- [Runtime Dependencies](#runtime-dependencies)
- [Build Instructions](#build-instructions)
  - [Build Dependencies](#build-dependencies)
  - [Build Steps](#build-steps)
  - [Run built-in unit-tests](#run-built-in-unit-tests)
- [How to Use](#how-to-use)
- [Iso2Mesh function port status](#iso2mesh-function-port-status)
- [How to Cite](#how-to-cite)
- [Acknowledgement](#acknowledgement)

## Introduction

Iso2Mesh is a versatile 3D mesh generation toolbox,
originally developed for MATLAB and GNU Octave in 2007.
It is designed for the easy creation of high-quality surface and
tetrahedral meshes from 3D volumetric images. It includes
over 200 mesh processing scripts and programs, which can operate
independently or in conjunction with external open-source
meshing tools. The Iso2Mesh toolbox can directly convert
3-D image stacks—including binary, segmented, or grayscale
images such as MRI or CT scans—into high-quality volumetric
meshes. This makes it especially suitable for multi-modality
medical imaging data analysis and multi-physics modeling.

The `iso2mesh` Python module provides a re-implementation of Iso2Mesh
in the native Python language, following algorithms
similar to those in the MATLAB/Octave versions of Iso2Mesh.

## How to Install

* PIP: `python3 -m pip install iso2mesh`
* PIP+Git (latest version): `python3 -m pip install git+https://github.com/NeuroJSON/pyiso2mesh.git`

MacOS users: you need to run the following commands to install this module

```
python3 -m venv /tmp/pyiso2mesh-venv
source /tmp/pyiso2mesh-venv/bin/activate
python3 -m pip install iso2mesh
```

## Runtime Dependencies

* **numpy**: `pyiso2mesh` relies heavily on vectorized NumPy
  matrix operations, similar to those used in the MATLAB version of Iso2Mesh.
* **matplotlib**: Used for plotting results. Install with `pip install matplotlib`.
* **jdata** (optional): A lightweight module to load JSON-encoded volume/mesh data,
  including dynamically downloading data from [NeuroJSON.io](https://neurojson.io) `pip install jdata`.

Many core meshing functions in `pyiso2mesh` require the set of mesh processing
executables provided in the Iso2Mesh package under the `iso2mesh/bin` folder.
These binaries are not needed at the time when installing `pyiso2mesh`; when
any of them becomes needed during mesh processing, `pyiso2mesh` dynamically
downloads these external tools and store those under the user's home directory
(`$HOME/iso2mesh-tools`) for future use. This download operation only runs once.

## Build Instructions

### Build Dependencies

* **Operating System**: The module is written in pure Python and is portable across platforms, 
including Windows, Linux, and macOS.

### Build Steps

1. Install the `build` module: `python3 -m pip install --upgrade build`

2. Clone the repository:

```
git clone --recursive https://github.com/NeuroJSON/pyiso2mesh.git
cd pyiso2mesh
```

3. Type `python3 -m build` to build the package

4. A platform-independent `noarch` module will be built locally. You should see a package
   named `iso2mesh-x.x.x-py2.py3-none-any.whl` in the `dist/` subfolder.

5. You can install the locally built package using:
   `python3 -m pip install --force-reinstall iso2mesh-*.whl`

### Run built-in unit-tests

If you want to modify the source, and verify that it still produces correct results, you can run
the built-in unit-test script inside the downloaded git repository by using this command
inside the `pyiso2mesh` root folder

```
python3 -m unittest test.run_test
```


## How to Use

`pyiso2mesh` inherits the "trademark" **one-liner mesh generator** style from its MATLAB/Octave counterpart
and maintains high compatibility with Iso2Mesh in terms of function names, input/output parameters,
and node/element ordering and indexing conventions.

All index matrices, such as `face` or `elem`, in the generated mesh data are 1-based (i.e.,
the lowest index is 1, not 0). This design ensures compatibility with the MATLAB/Octave Iso2Mesh outputs.

```python3
import iso2mesh as i2m
import numpy as np

# creating basic grid-like meshes
no, el = i2m.meshgrid5([0,1], [0,2], [1,2])
i2m.plotmesh(no, el)

no, el = i2m.meshgrid6([0,1], [0,2], [1,2])
i2m.plotmesh(no, el)

# meshing a box and plotting with selector
no, fc, el = i2m.meshabox([0,0,0], [30, 20, 10], 2)
i2m.plotmesh(no, el, 'z < 5')

# computing various mesh data

fc2 = i2m.volface(el)
ed1 = i2m.surfedge(fc[:-1,:])
fvol = i2m.elemvolume(no, fc)
evol = i2m.elemvolume(no, el)
facenb = i2m.faceneighbors(el)
snorm = i2m.surfacenorm(no, fc)
cv = i2m.meshcentroid(no, el)
cf = i2m.meshcentroid(no, fc)

# plotting nodes with markers
i2m.plotmesh(cf, 'r.')

# cleaning a surface mesh
no1, fc1 = i2m.meshcheckrepair(no, fc)

# smoothing a surface mesh
no2 = i2m.sms(no1, fc1, 20)

i2m.plotmesh(no2, fc1)

# meshing a cylinder
no, fc, el = i2m.meshacylinder([0,0,0], [0, 0, 10], 2, 50)
i2m.plotmesh(no, el, 'x < 0', edgecolor='r')

# creating and plotting polyhedral solids (PLCs)
mesh = i2m.latticegrid([0,1],[0,1,2], [0,2])
i2m.plotmesh(mesh[0], mesh[1], alpha=0.5, linestyle='--')

# mesh and label PLC based domains using tetgen1.5
mesh2 = i2m.s2m(mesh[0], mesh[1], 1, 0.03, method='tetgen1.5')
i2m.plotmesh(mesh2[0], mesh2[1], alpha=0.5)
```

`pyiso2mesh` subdivides all functions into sub-modules (**core, geometry, plot,
io, trait, modify, utils, register, raster**) that can be individually
imported. For example, if one wants to create tetrahedral meshes from a 3-D binary
array, one can use

```python3
from iso2mesh.core import v2m, v2s
from iso2mesh.plot import plotmesh
import numpy as np

img = np.zeros([60,60,60], dtype=np.uint8)
img[20:41, 10:51, 30:] = 1

no, el, fc = v2m(img, [], 3, 100, 'cgalmesh')
plotmesh(no, el)

no, fc, _, _ = v2s(img, 0.5, {'distbound': 0.2})
ax = plotmesh(no, fc, 'y < 30', alpha=0.5,  edgecolor='none')
plotmesh(no, fc, 'y > 30', parent = ax)
```

## Iso2Mesh function port status

The progress of converting MATLAB-based Iso2Mesh functions to Python is
tracked in https://github.com/NeuroJSON/pyiso2mesh/issues/1

| Ported | Unit-tested |     | Ported | Unit-tested |
| ------ | ------ | --- | ------ | ------ |
| > All-in-one pipeline shortcuts       | |  | > File I/O | |
| ✅ `v2m.m` | ✅ tested |                    | ✅ `saveasc.m` | ⭕️ tested |
| ✅ `v2s.m` | ✅ tested |                    | ✅ `savedxf.m` | ⭕️ tested |
| ✅ `s2m.m` | ✅ tested |                    | ✅ `savestl.m` | ⭕️ tested |
| ✅ `s2v.m` | ✅ tested |                    | ✅ `savebinstl.m` | ⭕️ tested |
| ✅ `m2v.m` | ✅ tested |                    | ✅ `saveinr.m` | ⭕️ tested |
| ✅ `sms.m` | ✅ tested |                    | ✅ `saveoff.m` | ✅ tested |
| > Streamlined mesh generation| |            | ⭕️ `savesmf.m` | ⭕️ tested |
| ✅ `vol2mesh.m` | ✅ tested |               | ✅ `savesurfpoly.m` | ✅ tested |
| ✅ `vol2surf.m` | ✅ tested |               | ⭕️ `savegts.m` | ⭕️ tested |
| ✅ `surf2mesh.m` | ✅ tested |              | ⭕️ `readgts.m` | ⭕️ tested |
| ✅ `surf2vol.m` | ✅ tested |               | ⭕️ `savemsh.m` | ⭕️ tested |
| ✅ `mesh2vol.m` | ✅ tested |               | ⭕️ `savevrml.m` | ⭕️ tested |
| > Iso2mesh main function backend| |         | ✅ `readasc.m` | ⭕️ tested |
| ✅ `binsurface.m` | ✅ tested |             | ⭕️ `readinr.m` | ⭕️ tested |
| ✅ `cgalv2m.m` | ✅ tested |                | ✅ `readmedit.m` | ⭕️ tested |
| ✅ `cgals2m.m` | ✅ tested |                | ✅ `readoff.m` | ✅ tested |
| ✅ `vol2restrictedtri.m` | ✅ tested |      | ⭕️ `readsmf.m` | ⭕️ tested |
| ✅ `surf2volz.m` | ✅ tested |              | ✅ `readtetgen.m` | ✅ tested |
| ✅ `mesh2mask.m` | ✅ tested |              | ✅ `deletemeshfile.m` | ✅ tested |
| > Iso2mesh primitive meshing|          |   | ✅ `mcpath.m` | ✅ tested |
| ✅ `meshabox.m` | ✅ tested |               | ✅ `mwpath.m` | ✅ tested |
| ✅ `meshasphere.m` | ✅ tested |            | ✅ `savemedit.m` | ✅ tested |
| ✅ `meshanellip.m` | ✅ tested |            | ⭕️ `savejson.m` | ⭕️ tested |
| ✅ `meshunitsphere.m` | ✅ tested |         | ⭕️ `loadjson.m` | ⭕️ tested |
| ✅ `meshacylinder.m` | ✅ tested |          | ⭕️ `saveubjson.m` | ⭕️ tested |
| ✅ `meshgrid5.m` | ✅ tested |              | ⭕️ `loadubjson.m` | ⭕️ tested |
| ✅ `meshgrid6.m` | ✅ tested |              | ⭕️ `loadmsgpack.m` | ⭕️ tested |
| ✅ `latticegrid.m` | ✅ tested |            | ⭕️ `savemsgpack.m` | ⭕️ tested |
| ✅ `extrudecurve.m` | ⭕️ tested |           | ⭕️ `savebj.m` | ⭕️ tested |
| ✅ `meshcylinders.m` | ✅ tested |          | ⭕️ `loadbj.m` | ⭕️ tested |
| > Mesh decomposition and query| |           | ⭕️ `savemphtxt.m` | ⭕️ tested |
| ✅ `finddisconnsurf.m` | ✅ tested |        | ⭕️ `savetetgenele.m` | ⭕️ tested |
| ✅ `surfedge.m` | ✅ tested |               | ⭕️ `savetetgennode.m` | ⭕️ tested |
| ✅ `volface.m` | ✅ tested |                | ⭕️ `saveabaqus.m` | ⭕️ tested |
| ✅ `extractloops.m` | ✅ tested |           | ⭕️ `savenirfast.m` | ⭕️ tested |
| ✅ `meshconn.m` | ✅ tested |               | ⭕️ `readnirfast.m` | ⭕️ tested |
| ✅ `meshcentroid.m` | ✅ tested |           | ⭕️ `readnifti.m` | ⭕️ tested |
| ✅ `nodevolume.m` | ✅ tested |             | ⭕️ `readmptiff.m` | ⭕️ tested |
| ✅ `elemvolume.m` | ✅ tested |             | ⭕️ `loadjsnirf.m` | ⭕️ tested |
| ✅ `neighborelem.m` | ✅ tested |           | ⭕️ `savejsnirf.m` | ⭕️ tested |
| ✅ `layersurf.m` | ⭕️ tested |              | ⭕️ `loadsnirf.m` | ⭕️ tested |
| ✅ `faceneighbors.m` | ✅ tested |          | ⭕️ `savesnirf.m` | ⭕️ tested |
| ✅ `edgeneighbors.m` | ✅ tested |          | ⭕️ `readobjmesh.m` | ⭕️ tested |
| ✅ `maxsurf.m` | ⭕️ tested |                | ⭕️ `loadjmesh.m` | ⭕️ tested |
| ✅ `flatsegment.m` | ⭕️ tested |            | ⭕️ `readobjmesh.m` | ⭕️ tested |
| ✅ `orderloopedge.m` | ⭕️ tested |          | > Volumetric image pre-processing| |
| ✅ `mesheuler.m` | ✅ tested |              | ✅ `volgrow.m` | ✅ tested |
| ✅ `bbxflatsegment.m` | ⭕️ tested |         | ✅ `volshrink.m` | ✅ tested |
| ✅ `surfplane.m` | ⭕️ tested |              | ✅ `volopen.m` | ✅ tested |
| ✅ `surfinterior.m` | ⭕️ tested |           | ✅ `volclose.m` | ✅ tested |
| ✅ `surfpart.m` | ⭕️ tested |               | ✅ `fillholes3d.m` | ✅ tested |
| ✅ `surfseeds.m` | ⭕️ tested |              | ⭕️ `bwislands.m` | ⭕️ tested |
| ✅ `meshquality.m` | ✅ tested |            | ⭕️ `laplacefill.m` | ⭕️ tested |
| ✅ `meshedge.m` | ✅ tested |               | ⭕️ `deislands2d.m` | ⭕️ tested |
| ✅ `meshface.m` | ✅ tested |               | ⭕️ `deislands3d.m` | ⭕️ tested |
| ✅ `surfacenorm.m` | ✅ tested |            | ⭕️ `ndgaussian.m` | ⭕️ tested |
| ✅ `nodesurfnorm.m` | ✅ tested |           | ⭕️ `ndimfilter.m` | ⭕️ tested |
| ✅ `uniqedges.m` | ✅ tested |              | ⭕️ `imedge3d.m` | ⭕️ tested |
| ✅ `uniqfaces.m` | ✅ tested |              | ⭕️ `internalpoint.m` | ⭕️ tested |
| ✅ `advancefront.m` | ⭕️ tested |           | ⭕️ `smoothbinvol.m` | ⭕️ tested |
| ✅ `innersurf.m` | ⭕️ tested |              | ✅ `thickenbinvol.m` | ✅ tested |
| ✅ `outersurf.m` | ⭕️ tested |              | ✅ `thinbinvol.m` | ✅ tested |
| ✅ `surfvolume.m` | ✅ tested |             | ⭕️ `maskdist.m` | ⭕️ tested |
| ✅ `insurface.m` | ✅ tested |              | > Mesh plotting| |
| > Mesh processing and reparing| |           | ✅ `plotmesh.m` | ✅ tested |
| ✅ `meshcheckrepair.m` | ✅ tested |        | ✅ `plotsurf.m` | ✅ tested |
| ✅ `meshreorient.m` | ✅ tested |           | ✅ `plottetra.m` | ✅ tested |
| ✅ `removedupelem.m` | ✅ tested |          | ✅ `plotedges.m` | ✅ tested |
| ✅ `removedupnodes.m` | ✅ tested |         | ✅ `qmeshcut.m` | ✅ tested |
| ✅ `removeisolatednode.m` | ✅ tested |      | > Miscellaneous functions| |
| ✅ `removeisolatedsurf.m` | ⭕️ tested |     | ⭕️ `surfdiffuse.m` | ⭕️ tested |
| ✅ `surfaceclean.m` | ⭕️ tested |           | ⭕️ `volmap2mesh.m` | ⭕️ tested |
| ✅ `getintersecttri.m` | ⭕️ tested |        | ⭕️ `isoctavemesh.m` | ⭕️ tested |
| ✅ `delendelem.m` | ⭕️ tested |             | ⭕️ `getvarfrom.m` | ⭕️ tested |
| ✅ `surfreorient.m` | ✅ tested |           | ✅ `raytrace.m` | ✅ tested |
| > Mesh registration | |                     | ⭕️ `linextriangle.m` | ⭕️ tested |
| ✅ `proj2mesh.m` | ⭕️ tested |              | ⭕️ `getplanefrom3pt.m` | ⭕️ tested |
| ✅ `dist2surf.m` | ⭕️ tested |              | ✅ `getexeext.m` | ✅ tested |
| ✅ `regpt2surf.m` | ⭕️ tested |             | ✅ `fallbackexeext.m` | ✅ tested |
| ✅ `affinemap.m` | ⭕️ tested |              | ⭕️ `iso2meshver.m` | ⭕️ tested |
| > Polyline handling| |                      | ⭕️ `raysurf.m` | ⭕️ tested |
| ✅ `slicesurf.m` | ⭕️ tested |              | ⭕️ `getoptkey.m` | ⭕️ tested |
| ✅ `slicesurf3.m` | ⭕️ tested |             | ✅ `rotatevec3d.m` | ⭕️ tested |
| ✅ `polylinelen.m` | ⭕️ tested |            | ✅ `rotmat2vec.m` | ⭕️ tested |
| ✅ `polylinesimplify.m` | ⭕️ tested |       | ✅ `varargin2struct.m` | ⭕️ tested |
| ✅ `polylineinterp.m` | ⭕️ tested |         | ✅ `jsonopt.m` | ⭕️ tested |
| ✅ `closestnode.m` | ⭕️ tested |            | ⭕️ `mergestruct.m` | ⭕️ tested |
| > Mesh resampling and optimization| |       | ⭕️ `orthdisk.m` | ⭕️ tested |
| ✅ `meshresample.m` | ✅ tested |           | ⭕️ `nestbracket2dim.m` | ⭕️ tested |
| ✅ `remeshsurf.m` | ✅ tested |             | ⭕️ `memmapstream.m` | ⭕️ tested |
| ✅ `smoothsurf.m` | ✅ tested |             | ⭕️ `aos2soa.m` | ⭕️ tested |
| ✅ `sortmesh.m` | ⭕️ tested |               | ⭕️ `soa2aos.m` | ⭕️ tested |
| ✅ `mergemesh.m` | ✅ tested |              | > Brain2mesh toolbox | |
| ✅ `meshrefine.m` | ✅ tested |             | ✅ `brain2mesh.m` | ⭕️ tested |
| ✅ `mergesurf.m` | ⭕️ tested |              | ✅ `brain1020.m` | ⭕️ tested |
| ✅ `surfboolean.m` | ✅ tested |            | ✅ `intriangulation.m` | ⭕️ tested |
| ✅ `fillsurf.m` | ⭕️ tested |               | ✅ `label2tpm.m` | ⭕️ tested |
| ✅ `highordertet.m` | ✅ tested |           | ✅ `tpm2label.m` | ⭕️ tested |
| ✅ `elemfacecenter.m` | ✅ tested |
| ✅ `barydualmesh.m` | ✅ tested |
| ✅ `meshinterp.m` | ⭕️ tested |
| ✅ `meshremap.m` | ⭕️ tested |
| ✅ `extrudesurf.m` | ⭕️ tested |


## How to Cite

Users of the pyiso2mesh toolbox should consider citing the below publications related to iso2mesh.

The original conference paper that described the **iso2mesh** toolbox workflow was published in 2009:

- Qianqian Fang and David Boas, "Tetrahedral mesh generation from volumetric binary and gray-scale images," 
  Proceedings of IEEE International Symposium on Biomedical Imaging 2009, pp. 1142-1145, 2009

A journal paper, published in 2020 describing the **brain2mesh** toolbox contains a much more
up-to-date description of iso2mesh in the context of complex brain mesh generation

- Anh Phong Tran, Shijie Yan and Qianqian Fang*, (2020) "Improving model-based fNIRS analysis using
  mesh-based anatomical and light-transport models," Neurophotonics, 7(1), 015008

If one utilized the `brain1020` function in your research, you should cite the below paper where
the workflow of this function was described

- Ashlyn McCann, Edward Xu, Fan-Yu Yen, Noah Joseph, Qianqian Fang, "Creating anatomically derived, 
  standardized, customizable, and three-dimensional printable head caps for functional neuroimaging," 
  Neurophoton. 12(1) 015016 (18 March 2025) https://doi.org/10.1117/1.NPh.12.1.015016, PMID: 39257741; PMCID: PMC11383710.


## Acknowledgement

The `pyiso2mesh` module was converted from the MATLAB/Octave version of
of Iso2Mesh (https://github.com/fangq/iso2mesh) written by the same author.

We utilized large-language-model (LLM) and AI chatbot in the initial MATLAB-to-Python
conversion, with specific instructions to **faithfully replicate** the algorithms
developed in the MATLAB code to avoid introducing external copyrighted materials
into this toolbox. With the assumption that the AI chatbot functions primarily
as an automated syntax translator without altering the originality of the code,
the upstream author of the original MATLAB-based Iso2Mesh retains the full copyright
of this derived Python module.

The initial translation was further manually tested, adjusted and restructured
to produce matching outputs as the original MATLAB toolbox. Dr. Edward Xu
<xu.ed at northeastern.edu> had contributed to the initial conversion and
testing of a subset of functions in the `geometry.py` and `trait.py` units.

The development of this software is supported by the US National Institute of Health (NIH)
under research awards [U24-NS124027](https://reporter.nih.gov/project-details/10308329) and 
[R01-CA204443](https://reporter.nih.gov/project-details/10982160), with the author,
Dr. Qianqian Fang, serving as the principle investigator on both awards.
