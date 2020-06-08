==============
summarycompare
==============

The `summarycompare` executable is designed to compare multiple result files.
This can be multiple PESummary metafiles or simply two dat files containing
posterior samples. 

To see help for this executable please run:

.. code-block:: console

    $ summarycompare --help

.. program-output:: summarycompare --help


Further details
---------------

The `sumamrycompare` executable loads in each result file with the `pesummary.io`
module and compares the properties. If the properties are a dictionary, we
recursively search through the entries, until a string, float or numpy array
can be compared between the two result files.

We print the differences to stdout through `logger`. If there are no differences
between the result files, no information is printed to stdout. However, this
can be modified by passing the `-v`/`--verbose` command line argument.

Examples
--------

Below we show an example where we compare the same metafiles:

.. code-block:: bash

    $ summarycompare --samples posterior_samples.h5 posterior_samples.h5 \
                     --properties_to_compare posterior_samples config priors
    2020-06-08  13:25:34 PESummary INFO    : Command line arguments: Namespace(compare=['posterior_samples', 'config', 'priors'], samples=['webpage/samples/posterior_samples.h5', 'webpage/samples/posterior_samples.h5'], verbose=False)
    $

As expected, nothing is returned as there are no differences. If we ran with
the verbose option:

.. code-block:: bash

    $ summarycompare --samples posterior_samples.h5 posterior_samples.h5 \
                     --properties_to_compare posterior_samples config priors \
                     --verbose
    2020-06-08  13:26:51 PESummary INFO    : Command line arguments: Namespace(compare=['posterior_samples', 'config', 'priors'], samples=['webpage/samples/posterior_samples.h5', 'webpage/samples/posterior_samples.h5'], verbose=True)
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/H1_matched_filter_abs_snr'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/H1_matched_filter_snr_angle'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/H1_optimal_snr'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/H1_spcal_amp_0'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/H1_spcal_amp_1'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/H1_spcal_amp_2'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/H1_spcal_amp_3'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/H1_spcal_amp_4'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/H1_spcal_amp_5'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/H1_spcal_amp_6'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/H1_spcal_amp_7'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/H1_spcal_amp_8'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/H1_spcal_amp_9'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/H1_spcal_phase_0'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/H1_spcal_phase_1'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/H1_spcal_phase_2'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/H1_spcal_phase_3'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/H1_spcal_phase_4'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/H1_spcal_phase_5'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/H1_spcal_phase_6'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/H1_spcal_phase_7'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/H1_spcal_phase_8'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/H1_spcal_phase_9'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/L1_matched_filter_abs_snr'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/L1_matched_filter_snr_angle'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/L1_optimal_snr'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/L1_spcal_amp_0'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/L1_spcal_amp_1'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/L1_spcal_amp_2'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/L1_spcal_amp_3'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/L1_spcal_amp_4'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/L1_spcal_amp_5'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/L1_spcal_amp_6'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/L1_spcal_amp_7'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/L1_spcal_amp_8'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/L1_spcal_amp_9'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/L1_spcal_phase_0'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/L1_spcal_phase_1'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/L1_spcal_phase_2'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/L1_spcal_phase_3'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/L1_spcal_phase_4'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/L1_spcal_phase_5'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/L1_spcal_phase_6'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/L1_spcal_phase_7'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/L1_spcal_phase_8'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/L1_spcal_phase_9'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/V1_matched_filter_abs_snr'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/V1_matched_filter_snr_angle'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/V1_optimal_snr'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/V1_spcal_amp_0'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/V1_spcal_amp_1'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/V1_spcal_amp_2'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/V1_spcal_amp_3'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/V1_spcal_amp_4'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/V1_spcal_amp_5'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/V1_spcal_amp_6'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/V1_spcal_amp_7'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/V1_spcal_amp_8'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/V1_spcal_amp_9'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/V1_spcal_phase_0'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/V1_spcal_phase_1'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/V1_spcal_phase_2'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/V1_spcal_phase_3'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/V1_spcal_phase_4'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/V1_spcal_phase_5'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/V1_spcal_phase_6'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/V1_spcal_phase_7'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/V1_spcal_phase_8'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/V1_spcal_phase_9'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/azimuth'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/deltalogl'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/log_likelihood'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/loglH1'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/loglL1'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/loglV1'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/logpost'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/log_prior'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/network_matched_filter_snr'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/nullLogL'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/network_optimal_snr'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/phase'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/phi_12'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/phi_jl'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/mass_ratio'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/t0'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/temperature'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/geocent_time'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/nLocalTemps'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/randomSeed'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/ra'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/dec'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/luminosity_distance'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/psi'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/chirp_mass'
    2020-06-08  13:27:00 PESummary DEBUG   : The result files match for entry: 'samples_dict/one/a_1'
    .
    .
    .

We now see alot more information and can see which entries have been compared.

Of course, this does not just have to be used to compare PESummary metafiles. We
can also compare two files containing only the posterior samples, for example:

.. code-block:: python

    >>> import numpy as np
    >>> from pesummary.io import write
    >>> parameters = ["a", "b", "c", "d"]
    >>> data = np.random.random([100, 4])
    >>> write(parameters, data, file_format="dat", filename="example1.dat")
    >>> parameters2 = ["a", "b", "c", "d", "e"]
    >>> data2 = np.random.random([100, 5])
    >>> write(parameters2, data2, file_format="json", filename="example2.json")

.. code-block:: bash

    $ summarycompare --samples example1.dat example2.json \
                     --properties_to_compare posterior_samples -v
    2020-06-08  13:36:45 PESummary INFO    : Command line arguments: Namespace(compare=['posterior_samples'], samples=['example1.dat', 'example2.json'], verbose=False)
    2020-06-08  13:36:47 PESummary WARNING : Failed to find 'log_likelihood' in result file. Setting every sample to have log_likelihood 0
    2020-06-08  13:36:47 PESummary INFO    : Failed to read in example2.json with the <bound method PESummary.load_file of <class 'pesummary.gw.file.formats.pesummary.PESummaryDeprecated'>> class because __init__() got an unexpected keyword argument 'disable_prior_conversion'
    2020-06-08  13:36:47 PESummary WARNING : Using the default load because example2.json failed the following checks: is_bilby_json_file, is_pesummary_json_file, is_pesummary_json_file_deprecated
    2020-06-08  13:36:47 PESummary WARNING : Failed to find 'log_likelihood' in result file. Setting every sample to have log_likelihood 0
    2020-06-08  13:36:47 PESummary INFO    : The result files differ for the following entry: 'samples_dict/a'. The maximum difference is: 0.863436584455099
    2020-06-08  13:36:47 PESummary INFO    : The result files differ for the following entry: 'samples_dict/b'. The maximum difference is: 0.866345243653647
    2020-06-08  13:36:47 PESummary INFO    : The result files differ for the following entry: 'samples_dict/c'. The maximum difference is: 0.892060295600003
    2020-06-08  13:36:47 PESummary INFO    : The result files differ for the following entry: 'samples_dict/d'. The maximum difference is: 0.8092611957933932
    2020-06-08  13:37:31 PESummary DEBUG   : The result files match for entry: 'samples_dict/log_likelihood'
