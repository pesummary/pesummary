=================
The read function
=================

`pesummary` adds extra functionality to read in `gw` specific result files. For
details about the `core` functionality see
`The core read function <../core/read.html>`_.

Reading a result file
---------------------

Below is how you might read in a result file,

.. code-block:: python

    >>> from pesummary.gw.file.read import read
    >>> lalinference = read("lalinference_result_file.hdf5")
    >>> bilby = read("bilby_result_file.json")

Parameter translation
---------------------

Different sampling softwares use different notations for the same parameters.
Consequently, `pesummary` tries to convert all parameters into a 'standard
format'. This means that parameters like, 'mass2', 'm2', 'mass_2' etc are all
converted to the 'standard' 'mass_2' parameter name. For details about what
each of the 'standard' parameter names mean, see
`The standard pesummary names <parameters.html>`_.

    >>> import h5py
    >>> lalinference_file = h5py.File("lalinference_result_file.hdf5")
    >>> posterior_samples = lalinference_file['lalinference/lalinference_mcmc/posterior_samples')
    >>> print(posterior_samples.dtype.names())
    ('H1_cplx_snr_amp', 'H1_cplx_snr_arg', 'H1_optimal_snr', 'H1_spcal_amp_0', 'H1_spcal_amp_1', 'H1_spcal_amp_2', 'H1_spcal_amp_3', 'H1_spcal_amp_4', 'H1_spcal_amp_5', 'H1_spcal_amp_6', 'H1_spcal_amp_7', 'H1_spcal_amp_8', 'H1_spcal_amp_9', 'H1_spcal_phase_0', 'H1_spcal_phase_1', 'H1_spcal_phase_2', 'H1_spcal_phase_3', 'H1_spcal_phase_4', 'H1_spcal_phase_5', 'H1_spcal_phase_6', 'H1_spcal_phase_7', 'H1_spcal_phase_8', 'H1_spcal_phase_9', 'L1_cplx_snr_amp', 'L1_cplx_snr_arg', 'L1_optimal_snr', 'L1_spcal_amp_0', 'L1_spcal_amp_1', 'L1_spcal_amp_2', 'L1_spcal_amp_3', 'L1_spcal_amp_4', 'L1_spcal_amp_5', 'L1_spcal_amp_6', 'L1_spcal_amp_7', 'L1_spcal_amp_8', 'L1_spcal_amp_9', 'L1_spcal_phase_0', 'L1_spcal_phase_1', 'L1_spcal_phase_2', 'L1_spcal_phase_3', 'L1_spcal_phase_4', 'L1_spcal_phase_5', 'L1_spcal_phase_6', 'L1_spcal_phase_7', 'L1_spcal_phase_8', 'L1_spcal_phase_9', 'V1_cplx_snr_amp', 'V1_cplx_snr_arg', 'V1_optimal_snr', 'V1_spcal_amp_0', 'V1_spcal_amp_1', 'V1_spcal_amp_2', 'V1_spcal_amp_3', 'V1_spcal_amp_4', 'V1_spcal_amp_5', 'V1_spcal_amp_6', 'V1_spcal_amp_7', 'V1_spcal_amp_8', 'V1_spcal_amp_9', 'V1_spcal_phase_0', 'V1_spcal_phase_1', 'V1_spcal_phase_2', 'V1_spcal_phase_3', 'V1_spcal_phase_4', 'V1_spcal_phase_5', 'V1_spcal_phase_6', 'V1_spcal_phase_7', 'V1_spcal_phase_8', 'V1_spcal_phase_9', 'azimuth', 'deltalogl', 'logl', 'loglH1', 'loglL1', 'loglV1', 'logpost', 'logprior', 'matched_filter_snr', 'nullLogL', 'optimal_snr', 'phase', 'phi12', 'phi_jl', 'q', 't0', 'temperature', 'time', 'nLocalTemps', 'randomSeed', 'ra', 'dec', 'dist', 'psi', 'mc', 'a1', 'a2', 'tilt1', 'tilt2', 'alpha', 'theta_jn', 'chain_log_evidence', 'chain_delta_log_evidence', 'chain_log_noise_evidence', 'chain_log_bayes_factor')
    >>> print(lalinference.parameters)
    ['H1_matched_filter_abs_snr', 'H1_matched_filter_snr_angle', 'H1_optimal_snr', 'H1_spcal_amp_0', 'H1_spcal_amp_1', 'H1_spcal_amp_2', 'H1_spcal_amp_3', 'H1_spcal_amp_4', 'H1_spcal_amp_5', 'H1_spcal_amp_6', 'H1_spcal_amp_7', 'H1_spcal_amp_8', 'H1_spcal_amp_9', 'H1_spcal_phase_0', 'H1_spcal_phase_1', 'H1_spcal_phase_2', 'H1_spcal_phase_3', 'H1_spcal_phase_4', 'H1_spcal_phase_5', 'H1_spcal_phase_6', 'H1_spcal_phase_7', 'H1_spcal_phase_8', 'H1_spcal_phase_9', 'L1_matched_filter_abs_snr', 'L1_matched_filter_snr_angle', 'L1_optimal_snr', 'L1_spcal_amp_0', 'L1_spcal_amp_1', 'L1_spcal_amp_2', 'L1_spcal_amp_3', 'L1_spcal_amp_4', 'L1_spcal_amp_5', 'L1_spcal_amp_6', 'L1_spcal_amp_7', 'L1_spcal_amp_8', 'L1_spcal_amp_9', 'L1_spcal_phase_0', 'L1_spcal_phase_1', 'L1_spcal_phase_2', 'L1_spcal_phase_3', 'L1_spcal_phase_4', 'L1_spcal_phase_5', 'L1_spcal_phase_6', 'L1_spcal_phase_7', 'L1_spcal_phase_8', 'L1_spcal_phase_9', 'V1_matched_filter_abs_snr', 'V1_matched_filter_snr_angle', 'V1_optimal_snr', 'V1_spcal_amp_0', 'V1_spcal_amp_1', 'V1_spcal_amp_2', 'V1_spcal_amp_3', 'V1_spcal_amp_4', 'V1_spcal_amp_5', 'V1_spcal_amp_6', 'V1_spcal_amp_7', 'V1_spcal_amp_8', 'V1_spcal_amp_9', 'V1_spcal_phase_0', 'V1_spcal_phase_1', 'V1_spcal_phase_2', 'V1_spcal_phase_3', 'V1_spcal_phase_4', 'V1_spcal_phase_5', 'V1_spcal_phase_6', 'V1_spcal_phase_7', 'V1_spcal_phase_8', 'V1_spcal_phase_9', 'azimuth', 'deltalogl', 'log_likelihood', 'loglH1', 'loglL1', 'loglV1', 'logpost', 'log_prior', 'network_matched_filter_snr', 'nullLogL', 'network_optimal_snr', 'phase', 'phi_12', 'phi_jl', 'mass_ratio', 't0', 'temperature', 'geocent_time', 'nLocalTemps', 'randomSeed', 'ra', 'dec', 'luminosity_distance', 'psi', 'chirp_mass', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'alpha', 'theta_jn', 'chain_log_evidence', 'chain_delta_log_evidence', 'chain_log_noise_evidence', 'chain_log_bayes_factor']

Changing file format
--------------------

With the `gw` read function, there is extra functionality to convert to
alernative file formats. This includes, for example, converting to a
lalinference file format,

.. code-block:: python

    >>> bilby.to_lalinference(outdir="./", filename="converted_bilby")

This will then convert the bilby format to a lalinference hdf5 file format. This
also means that, where possible, `pesummary` will convert the parameter names
from the `bilby` convention, to the `lalinference` convention. If you would
prefer to convert to a lalinference dat file, this is possible with the extra
`dat=True` option,

.. code-block:: python

    >>> bilby.to_lalinference(outdir="./", filename="converted_bilby", dat=True)

This will then convert the `bilby` result file into the more traditional
`posterior_samples.dat` file produced by `lalinference`.

Parameter conversion
--------------------

For `gw` specific result files, there are many posteriors that we are interested
in which can be generated from the posteriors already collected from the
sampler. This includes, for example, the `total_mass` posterior from the
`chirp_mass` and `mass_ratio` posterior samples. `pesummary` includes a
comprehensive conversion module which includes lots of useful conversion
functions. See `The Conversion class <Conversion.html>`_ for details.

We may convert the stored posterior samples with the
`.generate_all_posterior_samples()` method, see below:

.. code-block:: python

    >>> lalinference.samples_dict["mass_1"]
    KeyError: "mass_1 not in dictionary."
    >>> lalinference.generate_all_posterior_samples()
    >>> lalinference.samples_dict["mass_1"]
    Array([...])
