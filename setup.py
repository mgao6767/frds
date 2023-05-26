import sys
from sysconfig import get_path
from setuptools import setup, find_packages, Extension

import numpy

if sys.platform in ("linux", "linux2"):  # linux
    extra_compile_args = ["-std=c++17", "-lstdc++"]
elif sys.platform == "darwin":  # OS X
    extra_compile_args = ["-std=c++17", "-lstdc++"]
elif sys.platform == "win32":  # Windows
    extra_compile_args = ["/O2", "/std:c++17"]

mod_isolation_forest = Extension(
    "frds.algorithms.isolation_forest.iforest_ext",
    include_dirs=[get_path("platinclude"), numpy.get_include()],
    sources=[
        "src/frds/algorithms/isolation_forest/iforest_module.cpp",
    ],
    extra_compile_args=extra_compile_args,
    language="c++",
)

ext_modules = [mod_isolation_forest]

setup(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
)
