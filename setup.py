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

import builtins
import warnings
from distutils import log
from pathlib import Path

import versioneer
from setuptools import setup
from setuptools.command import (
    build_py,
    sdist,
)

# tell python we're in setup.py
builtins._PESUMMARY_SETUP = True

version = versioneer.get_version()
version_file = Path("pesummary") / ".version"


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
    from pesummary._version_helper import GitInformation, PackageInformation

    git_info = GitInformation()
    packages = PackageInformation()

    with version_file.open("w") as f:
        f.writelines(["# pesummary version information\n\n"])
        f.writelines(["version = %s\n" % (version)])
        f.writelines(["last_release = %s\n" % (git_info.last_version)])
        f.writelines(["\ngit_hash = %s\n" % (git_info.hash)])
        f.writelines(["git_author = %s\n" % (git_info.author)])
        f.writelines(["git_status = %s\n" % (git_info.status)])
        f.writelines(["git_builder = %s\n" % (git_info.builder)])
        f.writelines(["git_build_date = %s\n\n" % (git_info.build_date)])
        f.writelines(["# Install information\n\n"])
        f.writelines(["install_path = %s\n" % (packages.package_dir)])
    return ".version"


class _VersionedCommand(object):
    def run(self):
        log.info("generating {}".format(version_file))
        try:
            write_version_file(version)
        except Exception as exc:
            if not version_file.is_file():
                raise
            warnings.warn("failed to generate .version file, will reuse existing copy")
        super().run()


class VersionedSdist(_VersionedCommand, sdist.sdist):
    pass


class VersionedBuildPy(_VersionedCommand, build_py.build_py):
    pass


readme = full_description()

setup(name='pesummary',
      version=version,
      description='Python package to produce summary pages for Parameter '
                  'estimation codes',
      author='Charlie Hoy',
      author_email='charlie.hoy@ligo.org',
      url='https://git.ligo.org/lscsoft/pesummary',
      download_url='https://git.ligo.org/lscsoft/pesummary',
      cmdclass={
           "sdist": VersionedSdist,
           "build_py": VersionedBuildPy,
      },
      install_requires=[
          'numpy>=1.15.4',
          'h5py',
          'matplotlib',
          'seaborn',
          'statsmodels',
          'corner',
          'tables',
          'deepdish',
          'pandas',
          'pygments',
          'astropy>=3.2.3',
          'lalsuite>=6.70.0',
          'gwpy',
          'configparser',
          'plotly'],
      include_package_data=True,
      packages=['pesummary', 'pesummary.core', 'pesummary.core.webpage',
                'pesummary.core.plots', 'pesummary.core.file',
                'pesummary.core.file.formats', 'pesummary.gw',
                'pesummary.gw.file', 'pesummary.gw.file.formats',
                'pesummary.gw.plots', 'pesummary.gw.webpage', 'pesummary.utils',
                'pesummary.conf', 'pesummary.cli', 'pesummary.io'],
      package_data={
          'pesummary': [version_file.name],
          'pesummary.core': ['js/*.js', 'css/*.css'],
          'pesummary.conf': ['matplotlib_rcparams.sty'],
          'pesummary.gw.plots': ['cylon.csv'],
      },
      entry_points={
          'console_scripts': [
              'summaryclassification=pesummary.cli.summaryclassification:main',
              'summaryclean=pesummary.cli.summaryclean:main',
              'summarycombine=pesummary.cli.summarycombine:main',
              'summarydetchar=pesummary.cli.summarydetchar:main',
              'summarymodify=pesummary.cli.summarymodify:main',
              'summarypages=pesummary.cli.summarypages:main',
              'summarypageslw=pesummary.cli.summarypageslw:main',
              'summarypipe=pesummary.cli.summarypipe:main',
              'summaryplots=pesummary.cli.summaryplots:main',
              'summarypublication=pesummary.cli.summarypublication:main',
              'summaryrecreate=pesummary.cli.summaryrecreate:main',
              'summaryreview=pesummary.cli.summaryreview:main',
              'summaryversion=pesummary.cli.summaryversion:main']},
      classifiers=[
          "Programming Language :: Python :: 3.5",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7"],
      license='MIT',
      long_description=readme,
      long_description_content_type='text/markdown')
