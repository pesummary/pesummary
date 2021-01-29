# Licensed under an MIT style license -- see LICENSE.md

import os
import shutil

import argparse

from pesummary.core.plots import plot
from pesummary.gw.plots import plot as gwplot
from pesummary.utils.array import Array
from subprocess import CalledProcessError

from astropy.coordinates import (CartesianRepresentation, SkyCoord,
                                 SphericalRepresentation)
from astropy.table import Table, setdiff
from astropy.utils.misc import NumpyRNGContext
from astropy import units as u
import numpy as np
from scipy import stats
import pytest
import matplotlib

from ligo.skymap.io.hdf5 import read_samples, write_samples
from ligo.skymap.tool.tests import run_entry_point


import numpy as np
import matplotlib
from matplotlib import rcParams
import pytest

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]
rcParams["text.usetex"] = False
np.random.seed(150914)


@pytest.fixture
def samples(tmpdir):
    mean = SkyCoord(ra=stats.uniform(0, 360).rvs() * u.hourangle,
                    dec=np.arcsin(stats.uniform(-1, 1).rvs()) * u.radian,
                    distance=stats.uniform(100, 200).rvs()).cartesian.xyz.value
    eigvals = stats.uniform(0, 1).rvs(3)
    eigvals *= len(eigvals) / eigvals.sum()
    cov = stats.random_correlation.rvs(eigvals) * 100
    pts = stats.multivariate_normal(mean, cov).rvs(200)
    pts = SkyCoord(pts, representation_type=CartesianRepresentation)
    pts.representation_type = SphericalRepresentation
    time = stats.uniform(-0.01, 0.01).rvs(200) + 1e9
    table = Table({
        'ra': pts.ra.rad, 'dec': pts.dec.rad, 'distance': pts.distance.value,
        'time': time
    })
    filename = str(tmpdir / 'samples.hdf5')
    write_samples(table, filename, path='/posterior_samples')
    return filename


def test_ligo_skymap(samples, tmpdir):
    run_entry_point('ligo-skymap-from-samples', '--seed', '150914',
                    samples, '-o', str(tmpdir),
                    '--instruments', 'H1', 'L1', 'V1', '--objid', 'S1234')
    table = Table.read(str(tmpdir / 'skymap.fits'), format='fits')
    _samples = Table.read(samples)
    fig = gwplot._ligo_skymap_plot(
        _samples["ra"], _samples["dec"], dist=_samples["distance"],
        savedir=str(tmpdir), label="pesummary"
    )
    pesummary_table = Table.read(
        str(tmpdir / 'pesummary_skymap.fits'), format='fits'
    )
    diff = setdiff(table, pesummary_table)
    assert not len(diff)
