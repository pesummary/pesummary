==============================
Understanding the results file
==============================

Information stored
------------------

Once you have run the :code:`summarypages.py` executable, with a single results file, a `posterior_samples.h5` metafile is written and saved in the `samples/` directory of your web directory. This results file will have the following structure,

.. code-block:: console

   label/
     -> approximant/
       -> injection_data
       -> injection_parameters
       -> parameter_names
       -> samples

The `label/` is of the form :code:`${GRACEDB}_${DETECTORS}`. If no `gracedb` event tag is given, the label is simply :code:`${DETECTORS}`. If :code:`PESummary` is unable to find any information for the detector network chosen in the given results file (`results_file.h5`), the label will be a number running from 0 to infinity. The `approximant/` is the approximant that was used to generate the samples. Inside the `approximant/` group, all information for that run is stored. All information about any injections are stored in the `injection_data` and `injection_parameters` datasets. The posterior samples are stored in the `samples` dataset and the corresponding parameters are stored in the `parameter_names` dataset.

.. note::
   It is impossible for two runs to have the same :code:`${LABEL}_${APPROXIMANT}`. If, for example, you are running the same gracedb event, with the same detector network and the same approximant (but with different prior settings for example), the label will be chosen such that these two runs can be distinguished. For instance, if we are using the gracedb event tage `G180714` and a `H1L1` detector network with the `IMRPhenomPv2` approximant, one label would be `G180714_H1L1_0` and the other would be `G180714_H1L1_1`.

The injected parameters and the injected values can be found with,

.. code-block:: python

   >>> import h5py
   >>> f = h5py.File("posterior_samples.h5")
   >>> path = "label/approximant"
   >>> injections = {i:j for i,j in zip(f["%s/injection_parameters" %(path)], f["%s/injection_data" %(path)])}
   >>> print(injections)
   {b'mass_ratio': nan, b'geocent_time': nan}

If no information about the injection can be found by :code:`PESummary`, then a `nan` is passed. 

If you would like to recover the posterior samples for a specific parameter (for instance chirp mass), then this can be done with,

.. code-block:: python

   >>> import h5py
   >>> f = h5py.File("posterior_samples.h5")
   >>> path = "label/approximant"
   >>> parameters = [i for i in f["%s/parameter_names" %(path)]]
   >>> ind = parameters.index("chirp_mass")
   >>> samples_for_chirp_mass = [i[ind] for i in f["%s/samples" %(path)]]

Multiple results files
----------------------

If you have either passed multiple results files to the :code:`summarypages.py` executable or you have added to an existing summary page (for information about this see `using PESummary <executable.rst>`_, the combined `posterior_samples.h5` metafile will take one of the following forms.

If the results files that you have passed have the same label but different approximants, the `posterior_samples.h5` file will have the following structure,

.. code-block:: console

   label/
     -> approximant/
       -> injection_data
       -> injection_parameters
       -> parameter_names
       -> samples
     -> approximant2/
       -> injection_data
       -> injection_parameters
       -> parameter_names
       -> samples

If the result files that you have pass have different labels, the `posterior_samples.h5` file will have the following structure,

.. code-block:: console

   label/
     -> approximant/
       -> injection_data
       -> injection_parameters
       -> parameter_names
       -> samples
   label2/
     -> approximant2/
       -> injection_data
       -> injection_parameters
       -> parameter_names
       -> samples
