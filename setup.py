# Copyright (C) 2018  Charlie Hoy <charlie.hoy@ligo.org>
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

from distutils.core import setup
from setuptools import find_packages

with open("README.md", "r") as f:
    readme = f.read()

setup(name='pesummary',
      version='0.1.2',
      description='Python package to produce summary pages for Parameter '
                  'estimation codes',
      author='Charlie Hoy',
      author_email='charlie.hoy@ligo.org',
      url='https://git.ligo.org/charlie.hoy/pesummary',
      download_url='https://git.ligo.org/charlie.hoy/pesummary',
      install_requires=[
        'h5py',
        'numpy',
        'corner',
        'matplotlib',
        'deepdish',
        'pandas',
        'pygments',
        'astropy',
        'lalsuite',
        'pytest'],
      include_package_data=True,
      packages=find_packages(),
      package_dir={'pesummary': 'pesummary'},
      package_data={'pesummary': ['js/*.js', 'css/*.css']},
      scripts=['pesummary/bin/summarypages.py'],
      classifiers=[
          "Programming Language :: Python :: 3.6"],
      license='MIT',
      long_description=readme,
      long_description_content_type='text/markdown')
