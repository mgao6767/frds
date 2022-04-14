import sys
from sysconfig import get_path

from setuptools import setup, find_namespace_packages, Extension
from setuptools.command.build_ext import build_ext

from frds import (
    __version__,
    __description__,
    __author__,
    __author_email__,
    __github_url__,
)

try:
    import numpy
except ImportError:
    print("Numpy needs to be installed.")
    sys.exit(1)

if sys.platform in ("linux", "linux2"):
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
    include_dirs=[get_path("platinclude"), numpy.get_include()],
    sources=[
        "frds/algorithms/isolation_forest/src/iforest_module.cpp",
    ],
    extra_compile_args=extra_compile_args,
    language="c++",
)

mod_trth_parser = Extension(
    "frds.mktstructure.trth_parser",
    include_dirs=[get_path("platinclude")],
    sources=["frds/mktstructure/trth_parser.c"],
    language="c",
)


with open("README.md", "r", encoding="utf-8") as f:
    README = f.read()

setup(
    name="frds",
    version=__version__,
    description=__description__,
    long_description=README,
    long_description_content_type="text/markdown",
    author=__author__,
    author_email=__author_email__,
    url=__github_url__,
    packages=find_namespace_packages(),
    package_data={
        "": ["LICENSE", "README.md", "*.cpp", "*.hpp", "*.c"],
    },
    entry_points={"console_scripts": ["frds-mktstructure=frds.mktstructure.main:main"]},
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
    ext_modules=[
        mod_isolation_forest,
        mod_trth_parser,
    ],
    cmdclass={"build_ext": build_ext},
)
