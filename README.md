![](https://neurojson.org/wiki/upload/neurojson_banner_long.png)

# pyiso2mesh - Versatile 3D Surface and Tetrahedral Mesh Generation and Processing Toolbox

* Copyright: (C) Qianqian Fang (2025) \<q.fang at neu.edu>
* License: GNU Public License V3 or later
* Version: 0.1.0
* URL: [https://pypi.org/project/pyiso2mesh/](https://pypi.org/project/pyiso2mesh/)
* Github: [https://github.com/NeuroJSON/pyiso2mesh](https://github.com/NeuroJSON/pyiso2mesh)

![Python Module](https://github.com/NeuroJSON/pyiso2mesh/actions/workflows/build_package.yml/badge.svg)\\

Iso2Mesh is a versatile 3D mesh generation toolbox,
originally developed for MATLAB and GNU Octave in 2007.
It is designed for the easy creation of high-quality surface and
tetrahedral meshes from 3D volumetric images. It includes
over 200 mesh processing scripts and programs, which can operate
independently or in conjunction with external open-source
meshing tools. The Iso2Mesh toolbox can directly convert
3D image stacks—including binary, segmented, or grayscale
images such as MRI or CT scans—into high-quality volumetric
meshes. This makes it especially suitable for multi-modality
medical imaging data analysis and multi-physics modeling.

This module provides a Python re-implementation of Iso2Mesh.
Most functions are written in native Python, following algorithms
nearly identical to those in the MATLAB/Octave versions of Iso2Mesh.

## How to Install

* PIP: `python3 -m pip install iso2mesh`, see [https://pypi.org/project/iso2mesh/](https://pypi.org/project/iso2mesh/)
* PIP+Git: `python3 -m pip install git+https://github.com/NeuroJSON/pyiso2mesh.git`

## Runtime Dependencies

* **numpy**: `pyiso2mesh` relies heavily on vectorized NumPy
  matrix operations, similar to those used in the MATLAB version of Iso2Mesh.
* **matplotlib**: Used for plotting results. Install with `pip install matplotlib`.
* (optional) **pyvista** and **tetgen**: Required for generating tetrahedral meshes from surfaces. Install with `pip install pyvista tetgen`.
* (optional) **jdata**: Only needed for reading/writing JNIfTI output files. Install with `pip install jdata`
  on any operating system. On Debian-based Linux distributions, you can also install it to the system interpreter
  using `sudo apt-get install python3-jdata`. See [https://pypi.org/project/jdata/](https://pypi.org/project/jdata/) for more details.

## Build Instructions

### Build Dependencies

* **Operating System**: `pyiso2mesh` can be built on most operating systems, including Windows, Linux, and macOS.
  The module is written in pure Python and is portable across platforms.

### Build Steps

1. Install the `build` module: `python3 -m pip install --upgrade build`

2. Clone the repository:

```bash
   git clone --recursive https://github.com/NeuroJSON/pyiso2mesh.git
   cd pyiso2mesh
```

3. Type `python3 -m build` to build the package

4. A platform-independent `noarch` module will be built locally. You should see a package
   named `iso2mesh-x.x.x-py2.py3-none-any.whl` in the `dist/` subfolder.

5. You can install the locally built package using:
   `python3 -m pip install --force-reinstall iso2mesh-*.whl`

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
no, fc, el = i2m.meshacylinder([0,0,0], [0, 0, 10], 2)
i2m.plotmesh(no, el, 'x < 0')

# creating and plotting polyhedral solids (PLCs)
mesh = i2m.latticegrid([0,1],[0,1,2], [0,2])
i2m.plotmesh(mesh[0], mesh[1].tolist())
```
