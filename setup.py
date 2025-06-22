from setuptools import setup

with open("README.md", "r") as fh:
    readme = fh.read()

setup(
    name="iso2mesh",
    packages=["iso2mesh"],
    version="0.3.1",
    license='GPLv3+',
    description="Image-based 3D Surface and Volumetric Mesh Generator",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Qianqian Fang",
    author_email="fangqq@gmail.com",
    maintainer="Qianqian Fang",
    url="https://github.com/NeuroJSON/pyiso2mesh",
    keywords=[
        "Iso2Mesh",
        "Mesh generation",
        "FEM",
        "MATLAB",
        "Mesh-based Monte Carlo",
        "CGAL",
        "Tetgen",
        "CSG",
    ],
    platforms="any",
    install_requires=["numpy>=1.8.0", "matplotlib"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
