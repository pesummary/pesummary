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

import subprocess
import os
import time
import numpy as np

from pesummary.core.finish import FinishingTouches
from pesummary.gw.inputs import PostProcessing
from pesummary.utils.utils import logger


class GWFinishingTouches(FinishingTouches):
    """Class to handle the finishing touches

    Parameters
    ----------
    ligo_skymap_PID: dict
        dictionary containing the process ID for the ligo.skymap subprocess
        for each analysis
    """
    def __init__(self, inputs, ligo_skymap_PID=None):
        super(GWFinishingTouches, self).__init__(inputs)
        self.ligo_skymap_PID = ligo_skymap_PID
        self.generate_ligo_skymap_statistics()

    def generate_ligo_skymap_statistics(self):
        """Extract key statistics from the ligo.skymap fits file
        """
        FAILURE = False
        if self.ligo_skymap_PID is None:
            return
        samples_dir = os.path.join(self.webdir, "samples")
        for label in self.labels:
            _path = os.path.join(samples_dir, "{}_skymap.fits".format(label))
            while not os.path.isfile(_path):
                try:
                    output = subprocess.check_output(
                        ["ps -p {}".format(self.ligo_skymap_PID[label])],
                        shell=True
                    )
                    cond1 = "summarypages" not in str(output)
                    cond2 = "defunct" in str(output)
                    if cond1 or cond2:
                        if not os.path.isfile(_path):
                            FAILURE = True
                        break
                except (subprocess.CalledProcessError, KeyError):
                    FAILURE = True
                    break
                time.sleep(60)
            if FAILURE:
                continue
            ess = subprocess.Popen(
                "ligo-skymap-stats {} -p 50 90 -o {}".format(
                    os.path.join(samples_dir, "{}_skymap.fits".format(label)),
                    os.path.join(
                        samples_dir, "{}_skymap_stats.dat".format(label)
                    )
                ), shell=True
            )
            ess.wait()
            self.save_skymap_stats_to_metafile(
                label, os.path.join(samples_dir, "{}_skymap_stats.dat".format(label))
            )
            self.save_skymap_data_to_metafile(
                label, os.path.join(samples_dir, "{}_skymap.fits".format(label))
            )

    def save_skymap_stats_to_metafile(self, label, filename):
        """Save the skymap statistics to the PESummary metafile

        Parameters
        ----------
        label: str
            the label of the analysis that the skymap statistics corresponds to
        filename: str
            name of the file that contains the skymap statistics for label
        """
        logger.info("Adding ligo.skymap statistics to the metafile")
        skymap_data = np.genfromtxt(filename, names=True, skip_header=True)
        keys = skymap_data.dtype.names
        command_line = (
            "summarymodify --webdir {} --samples {} "
            "--delimiter / --kwargs {} --overwrite".format(
                self.webdir, os.path.join(self.webdir, "samples", "posterior_samples.h5"),
                " ".join(["{}/{}:{}".format(label, key, skymap_data[key]) for key in keys])
            )
        )
        ess = subprocess.Popen(command_line, shell=True)
        ess.wait()

    def save_skymap_data_to_metafile(self, label, filename):
        """Save the skymap data to the PESummary metafile

        Parameters
        ----------
        label: str
            the label of the analysis that the skymap corresponds to
        filename: str
            name of the fits file that contains the skymap for label
        """
        logger.info("Adding ligo.skymap data to the metafile")
        command_line = (
            "summarymodify --webdir {} --samples {} "
            "--store_skymap {}:{} --overwrite".format(
                self.webdir, os.path.join(self.webdir, "samples", "posterior_samples.h5"),
                label, filename
            )
        )
        ess = subprocess.Popen(command_line, shell=True)
        ess.wait()
