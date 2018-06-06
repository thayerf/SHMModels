#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='SHMModels',
      version='1.0',
      description='Mechanistic modeling of somatic hypermutation',
      author='Julia Fukuyama',
      author_email='julia.fukuyama@gmail.com',
      url='http://www.github.com/matsengrp/SHMModels',
      packages=find_packages(exclude=['test', 'analysis']),
      package_data={'SHMModels': ['data/*']}
)
