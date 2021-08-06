# Licensed under an MIT style license -- see LICENSE.md

import numpy as np
from pesummary.utils.utils import logger
from pesummary.core.file.meta_file import (
    _MetaFile, recursively_save_dictionary_to_hdf5_file,
    DEFAULT_HDF5_KEYS as CORE_HDF5_KEYS
)
from pesummary.gw.inputs import GWPostProcessing

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]
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
        filename=None, external_hdf5_links=False, hdf5_compression=None,
        history=None, descriptions=None, gwdata=None
    ):
        self.calibration = calibration
        self.psds = psd
        self.approximant = approximant
        self.existing_psd = existing_psd
        self.existing_calibration = existing_calibration
        self.existing_approximant = existing_approximant
        self.skymap = skymap
        self.existing_skymap = existing_skymap
        self.gwdata = gwdata
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
            mcmc_samples=mcmc_samples, filename=filename,
            external_hdf5_links=external_hdf5_links,
            hdf5_compression=hdf5_compression, history=history,
            descriptions=descriptions
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
        if self.gwdata is not None and len(self.gwdata):
            try:
                from gwpy.types.io.hdf5 import format_index_array_attrs
                from pesummary.utils.dict import Dict
                self.data["strain"] = {}
                for key, item in self.gwdata.items():
                    if item is None:
                        continue
                    _name = item.name if item.name is not None else "unknown_name"
                    self.data["strain"][key] = Dict(
                        {_name: item.value},
                        extra_kwargs=format_index_array_attrs(item)
                    )
            except Exception:
                logger.warn(
                    "Failed to store the gravitational wave strain data"
                )

    @staticmethod
    def save_to_hdf5(
        data, labels, samples, meta_file, no_convert=False, mcmc_samples=False,
        external_hdf5_links=False, compression=None, _class=None, gwdata=None
    ):
        """Save the metafile as a hdf5 file
        """
        if gwdata is not None and len(gwdata):
            extra_keys = CORE_HDF5_KEYS + ["strain"]
        else:
            extra_keys = CORE_HDF5_KEYS
        _MetaFile.save_to_hdf5(
            data, labels, samples, meta_file, no_convert=no_convert,
            extra_keys=extra_keys, mcmc_samples=mcmc_samples, _class=_class,
            external_hdf5_links=external_hdf5_links, compression=compression
        )


class _TGRMetaFile(_GWMetaFile):
    """Class to create a single file to contain TGR data
    """
    def __init__(
        self, samples, labels, imrct_data=None, webdir=None, outdir=None,
        file_kwargs={}
    ):
        super(_TGRMetaFile, self).__init__(
            samples, labels, None, None, None, None, webdir=webdir,
            outdir=outdir, filename="tgr_samples.h5", hdf5=True
        )
        self.tgr_data = {"imrct": imrct_data}
        self.file_kwargs = file_kwargs
        for key in self.tgr_data.keys():
            if key not in self.file_kwargs:
                self.file_kwargs[key] = {}

    @staticmethod
    def convert_posterior_samples_to_numpy(labels, samples, mcmc_samples=False):
        """Convert a dictionary of multiple posterior samples from a
        column-major dictionary to a row-major numpy array

        Parameters
        ----------
        labels: list
            list of unique labels for each analysis
        samples: MultiAnalysisSamplesDict
            dictionary of multiple posterior samples to convert to a numpy
            array.
        mcmc_samples: Bool, optional
            if True, the dictionary contains seperate mcmc chains

        Examples
        --------
        >>> dictionary = MultiAnalysisSamplesDict(
        ...     {"label": {"mass_1": [1,2,3], "mass_2": [1,2,3]}}
        ... )
        >>> dictionary = _Metafile.convert_posterior_samples_to_numpy(
        ...     dictionary.keys(), dictionary
        ... )
        >>> print(dictionary)
        ... {"label": rec.array([(1., 1.), (2., 2.), (3., 3.)],
        ...           dtype=[('mass_1', '<f4'), ('mass_2', '<f4')])}
        """
        _convert_function = _GWMetaFile._convert_posterior_samples_to_numpy
        converted_samples = {label: {} for label in labels}
        for label in labels:
            for key in ["inspiral", "postinspiral"]:
                if "{}:{}".format(label, key) in samples.keys():
                    _samples_key = "{}:{}".format(label, key)
                else:
                    _samples_key = key
                converted_samples[label][key] = _convert_function(
                    samples[_samples_key], mcmc_samples=False
                )
        return converted_samples

    @staticmethod
    def save_to_hdf5(*args, **kwargs):
        """Save the metafile as a hdf5 file
        """
        return _GWMetaFile.save_to_hdf5(*args, _class=_TGRMetaFile, **kwargs)

    def _make_dictionary(self):
        """Generate a single dictionary which stores all information
        """
        dictionary = self._dictionary_structure
        dictionary.update(
            {
                label: {key: {} for key in self.tgr_data.keys()} for label in
                self.labels
            }
        )
        for label in self.labels:
            dictionary[label]["posterior_samples"] = {}
            if self.tgr_data["imrct"] is not None:
                for analysis in ["inspiral", "postinspiral"]:
                    try:
                        _samples = self.samples["{}:{}".format(label, analysis)]
                    except KeyError:
                        _samples = self.samples[analysis]
                    parameters = _samples.keys()
                    samples = np.array([_samples[i] for i in parameters]).T
                    dictionary[label]["posterior_samples"][analysis] = {
                        "parameter_names": list(parameters),
                        "samples": samples.tolist()
                    }
                deviations = "final_mass_final_spin_deviations"
                _imrct_data = self.tgr_data["imrct"][label][deviations]
                dictionary[label]["imrct"] = {
                    "final_mass_deviation": _imrct_data.x,
                    "final_spin_deviation": _imrct_data.y,
                    "pdf": _imrct_data.probs,
                }
                _kwargs = self.file_kwargs["imrct"]
                if label in _kwargs.keys():
                    _kwargs = _kwargs[label]
                dictionary[label]["imrct"]["meta_data"] = _kwargs
        self.data = dictionary


class GWMetaFile(GWPostProcessing):
    """This class handles the creation of a metafile storing all information
    from the analysis
    """
    def __init__(self, inputs, history=None):
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
            existing_skymap=self.existing_skymap, filename=self.filename,
            external_hdf5_links=self.external_hdf5_links,
            hdf5_compression=self.hdf5_compression, history=history,
            gwdata=self.gwdata, descriptions=self.descriptions
        )
        meta_file.make_dictionary()
        if not self.hdf5:
            meta_file.save_to_json(meta_file.data, meta_file.meta_file)
        else:
            meta_file.save_to_hdf5(
                meta_file.data, meta_file.labels, meta_file.samples,
                meta_file.meta_file, mcmc_samples=meta_file.mcmc_samples,
                external_hdf5_links=meta_file.external_hdf5_links,
                compression=meta_file.hdf5_compression,
                gwdata=meta_file.gwdata
            )
        meta_file.save_to_dat()
        meta_file.write_marginalized_posterior_to_dat()
        logger.info(
            "Finishing generating the meta file. The meta file can be viewed "
            "here: {}".format(meta_file.meta_file)
        )


class TGRMetaFile(_GWMetaFile):
    """Class to create a single file to contain TGR data
    """
    def __init__(
        self, samples, labels, imrct_data=None, webdir=None, outdir=None,
        file_kwargs={}
    ):
        logger.info("Starting to generate the meta file")
        meta_file = _TGRMetaFile(
            samples, labels, imrct_data=imrct_data, webdir=webdir,
            outdir=outdir, file_kwargs=file_kwargs
        )
        meta_file.make_dictionary()
        meta_file.save_to_hdf5(
            meta_file.data, meta_file.labels, meta_file.samples,
            meta_file.meta_file, mcmc_samples=False
        )
        logger.info(
            "Finished generating the meta file. The meta file can be viewed "
            "here: {}".format(meta_file.meta_file)
        )
