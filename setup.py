from distutils.core import setup
from setuptools import find_packages

import os

cwd = os.getcwd()

setup(
    name='afsrr',
    version='1.0',
    description=(
        'This is an official implementation of the code used in the paper'
        ' "Atrial Fibrillation Screening During Sinus-Rhythm via Analysis of Cardiac Dynamics" - '
        'Yonatan Elul, Noam Keidar, Yael Drori, Alex M. Bronstein, Assaf Schuster, Yael Yaniv.'
    ),
    author='Yonatan Elul',
    author_email='renedal@gmail.com',
    url='https://github.com/YonatanElul/afsrr.git',
    license='MIT License',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: AI/ML Researchers, Researchers/Developers working on AF screening',
        'Topic :: Software Development :: AF Screening',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    package_dir={'afsrr': os.path.join(cwd, 'afsrr')},
    packages=find_packages(
        exclude=[
            'data',
            'logs',
        ]
    ),
    install_requires=[
        'numpy==1.26.2',
        'scipy==1.13.1',
        'scikit-learn==1.3.2',
        'wfdb==3.4.1',
        'matplotlib==3.8.2',
        'torch>=2.0.1',
        'tqdm==4.64.0',
        'h5py==3.6.0',
        'pandas==2.1.4',
        'seaborn==0.13.2',
        'ishneholterlib==2020.5.29',
    ],
)
