# Licensed under an MIT style license -- see LICENSE.md

import subprocess
import os
import time
import numpy as np

from pesummary.core.parser import convert_dict_to_namespace
from pesummary.core.finish import FinishingTouches
from pesummary.gw.inputs import PostProcessing
from pesummary.utils.utils import logger
from pesummary.cli.summarymodify import _main, command_line

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


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

        _dict = {
            "webdir": self.webdir,
            "samples": [os.path.join(self.webdir, "samples", "posterior_samples.h5")],
            "kwargs": {label: ["{}:{}".format(key, float(skymap_data[key])) for key in keys]},
            "overwrite": True
        }
        opts = convert_dict_to_namespace(_dict, add_defaults=command_line())
        _main(opts)

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

        _dict = {
            "webdir": self.webdir,
            "samples": [os.path.join(self.webdir, "samples", "posterior_samples.h5")],
            "overwrite": True, "store_skymap": {label: filename}
        }
        opts = convert_dict_to_namespace(_dict, add_defaults=command_line())
        _main(opts)
