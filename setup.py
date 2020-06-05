from setuptools import setup

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
      packages=['frds'],
      install_requires=requires)
