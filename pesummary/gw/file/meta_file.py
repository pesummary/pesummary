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

from pesummary.utils.utils import logger
from pesummary.core.file.meta_file import (
    _MetaFile, recursively_save_dictionary_to_hdf5_file,
    DEFAULT_HDF5_KEYS as CORE_HDF5_KEYS
)
from pesummary.gw.inputs import GWPostProcessing

DEFAULT_HDF5_KEYS = CORE_HDF5_KEYS


class _GWMetaFile(_MetaFile):
    """This class handles the creation of a meta file storing all information
    from the analysis

    Attributes
    ----------
    meta_file: str
        name of the meta file storing all information
    """
    def __init__(
        self, samples, labels, config, injection_data, file_versions,
        file_kwargs, calibration=None, psd=None, approximant=None, webdir=None,
        result_files=None, hdf5=False, existing_version=None,
        existing_label=None, existing_samples=None, existing_psd=None,
        existing_calibration=None, existing_approximant=None,
        existing_config=None, existing_injection=None,
        existing_metadata=None, priors={}, outdir=None, existing=None,
        existing_priors={}, existing_metafile=None, package_information={},
        mcmc_samples=False, skymap=None, existing_skymap=None,
        filename=None
    ):
        self.calibration = calibration
        self.psds = psd
        self.approximant = approximant
        self.existing_psd = existing_psd
        self.existing_calibration = existing_calibration
        self.existing_approximant = existing_approximant
        self.skymap = skymap
        self.existing_skymap = existing_skymap
        super(_GWMetaFile, self).__init__(
            samples, labels, config, injection_data, file_versions,
            file_kwargs, webdir=webdir, result_files=result_files, hdf5=hdf5,
            priors=priors, existing_version=existing_version,
            existing_label=existing_label, existing_samples=existing_samples,
            existing_injection=existing_injection,
            existing_metadata=existing_metadata,
            existing_config=existing_config, existing_priors=existing_priors,
            outdir=outdir, package_information=package_information,
            existing=existing, existing_metafile=existing_metafile,
            mcmc_samples=mcmc_samples, filename=filename
        )
        if self.calibration is None:
            self.calibration = {label: {} for label in self.labels}
        if self.psds is None:
            self.psds = {label: {} for label in self.labels}

    def _make_dictionary(self):
        """Generate a single dictionary which stores all information
        """
        super(_GWMetaFile, self)._make_dictionary()
        for num, label in enumerate(self.labels):
            cond = all(self.calibration[label] != j for j in [{}, None])
            if self.calibration != {} and cond:
                self.data[label]["calibration_envelope"] = {
                    key: item for key, item in self.calibration[label].items()
                    if item is not None
                }
            else:
                self.data[label]["calibration_envelope"] = {}
            if self.psds != {} and all(self.psds[label] != j for j in [{}, None]):
                self.data[label]["psds"] = {
                    key: item for key, item in self.psds[label].items() if item
                    is not None
                }
            else:
                self.data[label]["psds"] = {}
            if self.approximant is not None and self.approximant[label] is not None:
                self.data[label]["approximant"] = self.approximant[label]
            else:
                self.data[label]["approximant"] = {}
            if self.skymap is not None and len(self.skymap):
                if self.skymap[label] is not None:
                    self.data[label]["skymap"] = {
                        "data": self.skymap[label],
                        "meta_data": {
                            key: item for key, item in
                            self.skymap[label].meta_data.items()
                        }
                    }

    @staticmethod
    def save_to_hdf5(
        data, labels, samples, meta_file, no_convert=False, mcmc_samples=False
    ):
        """Save the metafile as a hdf5 file
        """
        _MetaFile.save_to_hdf5(
            data, labels, samples, meta_file, no_convert=no_convert,
            extra_keys=CORE_HDF5_KEYS, mcmc_samples=mcmc_samples
        )


class GWMetaFile(GWPostProcessing):
    """This class handles the creation of a metafile storing all information
    from the analysis
    """
    def __init__(self, inputs):
        super(GWMetaFile, self).__init__(inputs)
        logger.info("Starting to generate the meta file")
        if self.add_to_existing:
            existing = self.existing
            existing_metafile = self.existing_metafile
            existing_samples = self.existing_samples
            existing_labels = self.existing_labels
            existing_psd = self.existing_psd
            existing_calibration = self.existing_calibration
            existing_config = self.existing_config
            existing_approximant = self.existing_approximant
            existing_injection = self.existing_injection_data
            existing_version = self.existing_file_version
            existing_metadata = self.existing_file_kwargs
            existing_priors = self.existing_priors
        else:
            existing_metafile = None
            existing_samples = None
            existing_labels = None
            existing_psd = None
            existing_calibration = None
            existing_config = None
            existing_approximant = None
            existing_injection = None
            existing_version = None
            existing_metadata = None
            existing = None
            existing_priors = {}

        meta_file = _GWMetaFile(
            self.samples, self.labels, self.config, self.injection_data,
            self.file_version, self.file_kwargs, calibration=self.calibration,
            psd=self.psd, hdf5=self.hdf5, webdir=self.webdir,
            result_files=self.result_files, existing_version=existing_version,
            existing_label=existing_labels, existing_samples=existing_samples,
            existing_psd=existing_psd, existing_calibration=existing_calibration,
            existing_approximant=existing_approximant,
            existing_injection=existing_injection,
            existing_metadata=existing_metadata,
            existing_config=existing_config, priors=self.priors,
            existing_priors=existing_priors, existing=existing,
            existing_metafile=existing_metafile, approximant=self.approximant,
            package_information=self.package_information,
            mcmc_samples=self.mcmc_samples, skymap=self.skymap,
            existing_skymap=self.existing_skymap, filename=self.filename
        )
        meta_file.make_dictionary()
        if not self.hdf5:
            meta_file.save_to_json(meta_file.data, meta_file.meta_file)
        else:
            meta_file.save_to_hdf5(
                meta_file.data, meta_file.labels, meta_file.samples,
                meta_file.meta_file, mcmc_samples=meta_file.mcmc_samples
            )
        meta_file.save_to_dat()
        meta_file.write_marginalized_posterior_to_dat()
        logger.info(
            "Finishing generating the meta file. The meta file can be viewed "
            "here: {}".format(meta_file.meta_file)
        )
