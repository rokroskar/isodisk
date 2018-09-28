#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

install_requirements = ['scipy', 'numpy', 'matplotlib', 'emcee']

setup_requirements = [
    'pytest-runner',
]

test_requirements = [
    'pytest',
]

setup(
    author="Rok Roškar",
    author_email='rok.roskar@sdsc.ethz.ch',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Analysis of spinning globs of gas, stars, and dark matter.",
    install_requires=install_requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='isodisk',
    name='isodisk',
    packages=find_packages(include=['isodisk']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/rokroskar/isodisk',
    version='0.1.0',
    zip_safe=False,
    dependency_links=['https://github.com/pynbody/pynbody/tarball/master#egg=pynbody'])
