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
      version='0.1',
      description='Financial Research Data Services',
      author='Mingze Gao',
      author_email='adrian.gao@outlook.com',
      packages=find_namespace_packages(),
      install_requires=requires,
      cmdclass={
          'install': PostInstallCommand
      })
