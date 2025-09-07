"""
Setup script for ulens_lsst package.
"""
from setuptools import setup, find_packages

setup(
    name='ulens_lsst',
    version='1.0.0',
    description='LSST Microlensing LSST-like Light Curve Simulation Pipeline',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Karen Nowogrodzki',
    author_email='nowo.karen@gmail.com',
    url='https://github.com/CosmoObs/microlensing/simulation_Rubin/dp0_rubin/ulens_lsst',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.23.0',
        'pandas>=1.5.0',
        'pyarrow>=7.0.0',
        'astropy>=5.0.0'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'ulens-lsst = ulens_lsst.run_pipeline:main',
        ],
    },
)
