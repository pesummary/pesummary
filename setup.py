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

from setuptools import setup
import subprocess

version = "0.1.6"


def full_description():
    """Get the full readme
    """
    with open("README.md", "r") as f:
        readme = f.read()
    return readme


def write_version_file(version):
    """Add the version number and the git hash to the file
    'pesummary.__init__.py'

    Parameters
    ----------
    version: str
        the release version of the code that you are running
    """
    try:
        git_log = subprocess.check_output(
            ["git", "log", "-1", "--pretty=format:%h"]).decode("utf-8")
    except Exception as e:
        print("Unable to obtain git version information, because %s" % (e))
        git_log = ""

    with open("pesummary/.version", "w") as f:
        f.writelines(["%s %s" % (version, git_log)])
    return ".version"


readme = full_description()
version_file = write_version_file(version)

setup(name='pesummary',
      version=version,
      description='Python package to produce summary pages for Parameter '
                  'estimation codes',
      author='Charlie Hoy',
      author_email='charlie.hoy@ligo.org',
      url='https://git.ligo.org/lscsoft/pesummary',
      download_url='https://git.ligo.org/lscsoft/pesummary',
      install_requires=[
          'numpy==1.15.4',
          'h5py',
          'matplotlib',
          'corner',
          'tables==3.4.4',
          'deepdish',
          'pandas',
          'pygments',
          'pluggy==0.9.0',
          'astropy==2.0.9',
          'lalsuite',
          'configparser'],
      include_package_data=True,
      packages=['pesummary', 'pesummary.core', 'pesummary.core.webpage',
                'pesummary.core.plots', 'pesummary.core.file',
                'pesummary.gw', 'pesummary.gw.file',
                'pesummary.gw.plots', 'pesummary.utils', 'cli'],
      package_data={'pesummary': ['core/js/*.js', 'core/css/*.css', version_file]},
      entry_points={
          'console_scripts': [
              'summaryconvert=pesummary.file.one_format:main',
              'summarypages=cli.summarypages:main',
              'summaryplots=cli.summaryplots:main']},
      classifiers=[
          "Programming Language :: Python :: 3.5",
          "Programming Language :: Python :: 3.6"],
      license='MIT',
      long_description=readme,
      long_description_content_type='text/markdown')
