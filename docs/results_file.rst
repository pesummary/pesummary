==============================
Understanding the results file
==============================

Information stored
------------------

Once you have run the :code:`summarypages` executable, with a single results file, a `posterior_samples.h5` metafile is written and saved in the `samples/` directory of your web directory. This results file will have the following structure,

.. code-block:: console

   posterior_samples/
       -> label/
           -> injection_data
           -> injection_parameters
           -> parameter_names
           -> samples

The default `label/` is of the form :code:`${TIME}_${RESULTS_FILE}. You are able to pass your own labels by using the `--labels` flag. Inside the `label/` group, all information for that run is stored. All information about any injections are stored in the `injection_data` and `injection_parameters` datasets. The posterior samples are stored in the `samples` dataset and the corresponding parameters are stored in the `parameter_names` dataset.

.. note::
   It is possible for two runs to have the same :code:`${TIME}_${RESULTS_FILE}`. If, this is the case, then the label will be chosen such that these two runs can be distinguished. For instance, if pass the `label` for multiple results files, then the actual label that will be used is `label_0` and `label_1`.

Extracting information from metafile
------------------------------------

All information can be extracted from the metafile using the code below,

.. literalinclude:: /../examples/results_file/extract_information.py 
   :language: python                                                            
   :linenos: 

Multiple results files
----------------------

If you have either passed multiple results files to the :code:`summarypages` executable or you have added to an existing summary page (for information about this see `using PESummary <executable.rst>`_, the combined `posterior_samples.h5` metafile will take one of the following forms.

If the results files that you have passed have the same label but different approximants, the `posterior_samples.h5` file will have the following structure,

.. code-block:: console

   posterior_samples/
       -> label_1/
           -> injection_data
           -> injection_parameters
           -> parameter_names
           -> samples
       -> label_2
           -> injection_data
           -> injection_parameters
           -> parameter_names
           -> samples
