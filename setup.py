from setuptools import setup, find_namespace_packages
from setuptools.command.install import install
import pathlib
import shutil
import os

requires = ["pandas", "sqlalchemy", "psycopg2-binary", "PyQt5"]


class PostInstallCommand(install):
    def run(self):
        install.run(self)
        base_dir = str(pathlib.Path("~/frds").expanduser())
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(os.path.join(base_dir, "data"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "result"), exist_ok=True)
        if not os.path.exists(os.path.join(base_dir, "config.ini")):
            shutil.copy("frds/config.ini", base_dir)


setup(
    name="frds",
    version="0.4.3",
    description="Financial Research Data Services",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Mingze Gao",
    author_email="adrian.gao@outlook.com",
    url="https://github.com/mgao6767/frds/",
    packages=find_namespace_packages(),
    package_data={
        "": ["LICENSE", "README.md", "*.ico"],
        "frds": ["config.ini"],
    },
    install_requires=requires,
    cmdclass={"install": PostInstallCommand},
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
)
