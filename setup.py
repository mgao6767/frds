from setuptools import setup, find_namespace_packages, Extension
import numpy
import frds
import sys

requires = [
    "pandas",
    "sqlalchemy",
    "psycopg2-binary",
]

mod_isolation_forest = Extension(
    "frds.algorithms.isolation_forest.iforest_ext",
    include_dirs=[*sys.path, numpy.get_include()],
    sources=[
        "frds/algorithms/isolation_forest/src/iforest_module.cpp",
    ],
    extra_compile_args=["/O2", "/std:c++17"],
    language="c++",
)

setup(
    name="frds",
    version=frds.__version__,
    description=frds.__description__,
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author=frds.__author__,
    author_email=frds.__author_email__,
    url=frds.__github_url__,
    packages=find_namespace_packages(),
    package_data={
        "": ["LICENSE", "README.md"],
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
)
