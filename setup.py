#!/usr/bin/env python
from setuptools import setup, find_packages

def readme():
  with open('README.rst') as f:
    return f.read()

setup(name='BayesianBlocks',
    version='0.1',
    description='Bayesian Blocks Analysis Software for HEP applications',
    url='https://github.com/brovercleveland/BayesianBlocks',
    author='Brian Pollack',
    author_email='brianleepollack@gmail.com',
    license='NU',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False)
