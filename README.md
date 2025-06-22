![](https://neurojson.org/wiki/upload/neurojson_banner_long.png)

# pyiso2mesh - One-liner 3D Surface and Tetrahedral Mesh Generation Toolbox

* Copyright: (C) Qianqian Fang (2024-2025) <q.fang at neu.edu>, Edward Xu (2024) <xu.ed at northeastern.edu>
* License: GNU Public License V3 or later
* Version: 0.3.1
* URL: [https://pypi.org/project/iso2mesh/](https://pypi.org/project/iso2mesh/)
* Github: [https://github.com/NeuroJSON/pyiso2mesh](https://github.com/NeuroJSON/pyiso2mesh)

![Python Module](https://github.com/NeuroJSON/pyiso2mesh/actions/workflows/build_all.yml/badge.svg)

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
