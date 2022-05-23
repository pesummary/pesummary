# Licensed under an MIT style license -- see LICENSE.md

import os
import warnings
from distutils import log
from pathlib import Path
import setuptools
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist
import setuptools_scm
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from pesummary._version_helper import make_version_file

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]

version = setuptools_scm.get_version()
version_file = Path("pesummary") / ".version"


class _VersionedCommand(object):
    def run(self):
        log.info("generating {}".format(version_file))
        try:
            _ = make_version_file(
                version_file=version_file, version=version,
                add_install_path=False
            )
        except Exception as exc:
            if not version_file.is_file():
                raise
            warnings.warn("failed to generate .version file, will reuse existing copy")
        super().run()


class VersionedSdist(_VersionedCommand, sdist):
    pass


class VersionedBuildPy(_VersionedCommand, build_py):
    pass


setup_params = dict(
    cmdclass={
        "sdist": VersionedSdist,
        "build_py": VersionedBuildPy,
    }
)
if __name__ == '__main__':
    dist = setuptools.setup(**setup_params)
