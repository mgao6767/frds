from setuptools import setup, find_namespace_packages, Extension
from setuptools.command.build_ext import build_ext
from distutils.sysconfig import get_python_inc
import sys

try:
    import numpy
except ImportError:
    print("Numpy needs to be installed.")
    sys.exit(1)

from frds import meta

if sys.platform == "linux" or sys.platform == "linux2":
    # linux
    extra_compile_args = ["-std=c++17", "-lstdc++"]
elif sys.platform == "darwin":
    # OS X
    extra_compile_args = ["-std=c++17", "-lstdc++"]
elif sys.platform == "win32":
    # Windows
    extra_compile_args = ["/O2", "/std:c++17"]

requires = [
    "scipy",
    "pandas",
    "sqlalchemy",
    "psycopg2-binary",
    "fredapi",
]

mod_isolation_forest = Extension(
    "frds.algorithms.isolation_forest.iforest_ext",
    include_dirs=[get_python_inc(True), numpy.get_include()],
    sources=[
        "frds/algorithms/isolation_forest/src/iforest_module.cpp",
    ],
    extra_compile_args=extra_compile_args,
    language="c++",
)

setup(
    name="frds",
    version=meta["__version__"],
    description=meta["__description__"],
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author=meta["__author__"],
    author_email=meta["__author_email__"],
    url=meta["__github_url__"],
    packages=find_namespace_packages(),
    package_data={
        "": ["LICENSE", "README.md", "*.cpp", "*.hpp"],
    },
    install_requires=requires,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    license="MIT",
    ext_modules=[mod_isolation_forest],
    cmdclass={"build_ext": build_ext},
)
