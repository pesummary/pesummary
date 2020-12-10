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
    from pesummary._version_helper import make_version_file

    return make_version_file(
        version_file=version_file, version=version, add_install_path=False
    )


cmdclass = versioneer.get_cmdclass()


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


class VersionedSdist(_VersionedCommand, cmdclass["sdist"]):
    pass


class VersionedBuildPy(_VersionedCommand, cmdclass["build_py"]):
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
          'seaborn>=0.11.0',
          'statsmodels',
          'corner',
          'tables',
          'deepdish',
          'pandas',
          'pygments',
          'astropy>=3.2.3',
          'lalsuite>=6.70.0',
          'python-ligo-lw',
          'ligo-gracedb',
          'gwpy>=2.0.2',
          'plotly',
          'tqdm>=4.44.0'],
      include_package_data=True,
      packages=['pesummary', 'pesummary.core', 'pesummary.core.webpage',
                'pesummary.core.plots', 'pesummary.core.file',
                'pesummary.core.file.formats', 'pesummary.core.notebook',
                'pesummary.gw', 'pesummary.gw.notebook',
                'pesummary.gw.file', 'pesummary.gw.file.formats',
                'pesummary.gw.plots', 'pesummary.gw.webpage', 'pesummary.utils',
                'pesummary.conf', 'pesummary.cli', 'pesummary.io',
                'pesummary.tests'],
      package_data={
          'pesummary': [version_file.name],
          'pesummary.core': ['js/*.js', 'css/*.css'],
          'pesummary.conf': ['matplotlib_rcparams.sty'],
          'pesummary.gw.plots': ['cylon.csv'],
          'pesummary.tests': ['*.sh', '*.ini', '*.xml', 'files/*.ini',
                              'files/*.txt'],
      },
      entry_points={
          'console_scripts': [
              'summaryclassification=pesummary.cli.summaryclassification:main',
              'summaryclean=pesummary.cli.summaryclean:main',
              'summarycombine=pesummary.cli.summarycombine:main',
              'summarycompare=pesummary.cli.summarycompare:main',
              'summarydetchar=pesummary.cli.summarydetchar:main',
              'summarygracedb=pesummary.cli.summarygracedb:main',
              'summaryjscompare=pesummary.cli.summaryjscompare:main',
              'summarymodify=pesummary.cli.summarymodify:main',
              'summarypages=pesummary.cli.summarypages:main',
              'summarypageslw=pesummary.cli.summarypageslw:main',
              'summarypipe=pesummary.cli.summarypipe:main',
              'summaryplots=pesummary.cli.summaryplots:main',
              'summarypublication=pesummary.cli.summarypublication:main',
              'summaryrecreate=pesummary.cli.summaryrecreate:main',
              'summaryreview=pesummary.cli.summaryreview:main',
              'summarytest=pesummary.cli.summarytest:main',
              'summaryversion=pesummary.cli.summaryversion:main']},
      classifiers=[
          "Programming Language :: Python :: 3.5",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8"],
      license='MIT',
      long_description=readme,
      long_description_content_type='text/markdown')
