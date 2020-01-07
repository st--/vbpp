#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: skip-file

from setuptools import setup

requirements = [
    'numpy',
    'scipy',
    'gpflow<2.0',
    'tensorflow<2.0',
]

setup(
    name='vbpp',
    version='0.0.1',
    author="ST John",
    #author_email="",
    description="Variational Bayes for Point Processes using GPflow",
    license="Apache License 2.0",
    #keywords="",
    url="http://github.com/st--/vbpp",
    install_requires=requirements,
    packages=['vbpp'],
)
