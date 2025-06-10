![](https://neurojson.org/wiki/upload/neurojson_banner_long.png)

# pyiso2mesh - Versatile 3-D surface and tetrahedral mesh generation and processing toolbox

- Copyright: (C) Qianqian Fang (2025) <q.fang at neu.edu>
- License: GNU Public License V3 or later
- Version: 0.1.1
- URL: https://pypi.org/project/pyiso2mesh/
- Github: https://github.com/NeuroJSON/pyiso2mesh

![Python Module](https://github.com/NeuroJSON/pyiso2mesh/actions/workflows/build_package.yml/badge.svg)\


Iso2Mesh is a versatile 3-D mesh generation toolbox,
initially developed for MATLAB and GNU Octave since 2007. 
Iso2Mesh is designed for easy creation of high quality surface and 
tetrahedral meshes from 3D volumetric images. It contains 
over 200 mesh processing scripts/programs, working 
either independently or interacting with external free 
meshing utilities. Iso2Mesh toolbox can directly convert
a 3D image stack, including binary, segmented or gray-scale 
images such as MRI or CT scans, into quality volumetric 
meshes. This makes it particularly suitable for multi-modality 
medical imaging data analysis and multi-physics modeling.

This module provides a Python re-implementation of Iso2Mesh.
The majority of the functions are written in the native Python
language, following nearly identical algorithm as the corresponding
functions in the MATLAB/Octave versions of Iso2Mesh.

## How to Install

* PIP: ```python3 -m pip install iso2mesh```, see https://pypi.org/project/iso2mesh/
* PIP+Git: ```python3 -m pip install git+https://github.com/NeuroJSON/pyiso2mesh.git```


## Runtime Dependencies
* **numpy**: `pyiso2mesh` functions are extensively built upon vecterized numpy
matrix operations, similar to the style used in MATLAB based Iso2Mesh.
* **matplotlib**: For plotting the results. To install, run either `pip install matplotlib`.
* (optional) **pyvista** and **tetgen** are needed to create tetrahedral mesh from surfaces. To install, use `pip install pyvista tetgen`
* (optional) **jdata**: Only needed to read/write JNIfTI output files. To install, use pip: `pip install jdata` 
on all operating systems; For Debian-based Linux distributions, you can also install to the system interpreter 
using apt-get: `sudo apt-get install python3-jdata`. See https://pypi.org/project/jdata/ for more details. 


## Build Instructions

### Build Dependencies
* **Operating System**: pyiso2mesh can be compiled on most OSes, including Windows, Linux and MacOS.
The module is written in the pure Python language, and thus is portable for all platforms.

### Build Steps
1. Install the `build` module for Python `python3 -m pip install --upgrade build`

2. Clone the repository
    ```bash
        git clone --recursive https://github.com/NeuroJSON/pyiso2mesh.git
        cd pyiso2mesh
    ```
3. A platform independent `noarch` module is successfully built locally, you should see the package 
with name `iso2mesh-x.x.x-py2.py3-none-any.whl` under the `dist/` subfolder. 

4. You can the install the locally built package using `python3 -m pip install --force-reinstall iso2mesh-*.whl`


## How to use

`pyiso2mesh` inherits the "trade-mark" `one-liner mesh generator` style from its MATLAB/Octave counterpart,
and maintains a high-compatibility to Iso2Mesh in terms of the function names, input/output parameters,
as well as the node/element ordering/indexing conventions.

All index matrices, such as `face` or `elem`, of the generated mesh data are 1-based (meaning that the
lowest index is 1 instead of 0). This is designed to be compatible with the MATLAB/Octave based Iso2Mesh
outputs.


```python3
import iso2mesh as i2m
import numpy as np

no, el = i2m.meshgrid5([0,1], [0,2], [1,2])
i2m.plotmesh(no, el)

no, el = i2m.meshgrid6([0,1], [0,2], [1,2])
i2m.plotmesh(no, el)

no, fc, el = i2m.meshabox([0,0,0], [30, 20, 10], 2)
i2m.plotmesh(no, el)
```
