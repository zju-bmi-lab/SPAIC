# coding: utf-8

# python -m build

from setuptools import setup
from setuptools import find_packages

setup(
        name='SPAIC',
        version='0.2.0.7.7',
        description='This is a test of setup for SPAIC',
        author='zjlab',
        packages=find_packages(),
        include_package_data=False,
        install_requires=[
                                'torch>=1.5.0',
                                'matplotlib',
                                'numpy',
                                'tqdm',
                                'scipy',
                                'h5py',
                           ],
        python_requires='>=3.6'
)