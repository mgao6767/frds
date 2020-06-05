from setuptools import setup, find_namespace_packages

requires = [
    'pandas',
    'sqlalchemy',
    'psycopg2-binary'
]

setup(name='frds',
      version='0.1',
      description='Financial Research Data Services',
      author='Mingze Gao',
      author_email='adrian.gao@outlook.com',
      packages=find_namespace_packages(),
      install_requires=requires)
