from setuptools import setup, find_namespace_packages
from setuptools.command.install import install
from frds import default_config, custom_config
import shutil

requires = [
    'pandas',
    'sqlalchemy',
    'psycopg2-binary'
]


class PostInstallCommand(install):
    def run(self):
        install.run(self)
        shutil.copyfile(default_config, custom_config)


setup(name='frds',
      version='0.1.3',
      description='Financial Research Data Services',
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      author='Mingze Gao',
      author_email='adrian.gao@outlook.com',
      url='https://github.com/mgao6767/frds/',
      packages=find_namespace_packages(),
      package_data={
          "": ["config.ini"],
      },
      install_requires=requires,
      cmdclass={
          'install': PostInstallCommand
      },
      classifiers=[
          "License :: OSI Approved :: MIT License",
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.8",
          'Intended Audience :: End Users/Desktop',
          'Intended Audience :: Financial and Insurance Industry',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Information Analysis',
      ],
      license='MIT')
