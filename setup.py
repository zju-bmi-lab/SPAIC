# coding: utf-8
from setuptools import setup

# Metadata goes in setup.cfg. There are here for GitHub's dependency graph
setup(
    name='SPAIC',
    install_requires=[
        'torch>=1.5.0',
        'matplotlib',
        'numpy',
        'tqdm',
        'scipy',
        'pyyaml',
        'h5py',
    ],
)
