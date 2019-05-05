===============
Using PESummary
===============

PESummary executables
----------------------

The primary user-interface for this code is a command line tool :code:`summarypages` which is available after following the `installation instructions <installation.html>`_. To see help for using this tool, please run,

.. code-block:: console

   $ summarypages --help

You will then see that there are general command line arguments,

.. code-block:: console

    usage: summarypages [-h] [-w DIR] [-b DIR] [-s SAMPLES [SAMPLES ...]]
                        [-c CONFIG [CONFIG ...]] [--email EMAIL] [--dump]
                        [--add_to_existing] [-e EXISTING]
                        [-i INJ_FILE [INJ_FILE ...]]
                        [--labels LABELS [LABELS ...]] [-v] [--save_to_hdf5]
                        [-a APPROXIMANT [APPROXIMANT ...]] [--sensitivity]
                        [--gracedb GRACEDB] [--psd PSD [PSD ...]]
                        [--calibration CALIBRATION [CALIBRATION ...]] [--gw]

    optional arguments:
      -h, --help            show this help message and exit
      -w DIR, --webdir DIR  make page and plots in DIR
      -b DIR, --baseurl DIR
                            make the page at this url
      -s SAMPLES [SAMPLES ...], --samples SAMPLES [SAMPLES ...]
                            Posterior samples hdf5 file
      -c CONFIG [CONFIG ...], --config CONFIG [CONFIG ...]
                            configuration file associcated with each samples file.
      --email EMAIL         send an e-mail to the given address with a link to the
                            finished page.
      --dump                dump all information onto a single html page
      --add_to_existing     add new results to an existing html page
      -e EXISTING, --existing_webdir EXISTING
                            web directory of existing output
      -i INJ_FILE [INJ_FILE ...], --inj_file INJ_FILE [INJ_FILE ...]
                            path to injetcion file
      --labels LABELS [LABELS ...]
                            labels used to distinguish runs
      -v, --verbose         print useful information for debugging purposes
      --save_to_hdf5        save the meta file in hdf5 format

    Options specific for gravitational wave results files:
      -a APPROXIMANT [APPROXIMANT ...], --approximant APPROXIMANT [APPROXIMANT ...]
                            waveform approximant used to generate samples
      --sensitivity         generate sky sensitivities for HL, HLV
      --gracedb GRACEDB     gracedb of the event
      --psd PSD [PSD ...]   psd files used
      --calibration CALIBRATION [CALIBRATION ...]
                            files for the calibration envelope
      --gw                  run with the gravitational wave pipeline

Running PESummary
-----------------

To run :code:`PESummary`, you first need to generate a results file using a sample generating code of your choice. Instructions on the format of the input results file can be seen `here <file_format.html>`_. 

Once you have at least one results file, (for clarity, lets call it :code:`results_file.h5`), you then need to decide where you would like to store the webpages. If you are working on an LDG cluster, then please use your :code:`public_html` directory. You can then generate webpages with,

.. code-block:: console

   $ summarypages --webdir /home/albert.einstein/public_html/projects/my_awesome_example
                  --samples ./results_file.h5

This will then generate all derived posterior samples, all plots, and all webpages. Information about the progress of your job will be printed. If you would require further details, please use the :code:`--verbose` flag.

All files are written to seperate directories. The basic structure is as follows,

.. code-block:: console

   home.html
   config/
     -> config.ini
   css/
     -> image_styles.css
     -> side_bar.css
   js/
     -> combine_corner.js
     -> grab.js
     -> modal.js
     -> multi_dropbar.js
     -> multiple_posteriors.js
     -> search.js
     -> side_bar.js
   plots/
     ->
   html/
     ->
   samples/
     -> posterior_samples.h5

The `plots/`, `html/` and `samples/` directories are all empty initially, and are populated as the job progresses. The home page of the generate webpage can be opened by viewing the `home.html` file in your browser. For details about the output pages please refer to `understanding the webpages <summarypage.html>`_. :code:`PESummary` also stored all information about the run in the `posterior_samples.h5` file. For details about this file, please refer to `understanding the results file <results_file.html>`_.

If you wish to get an email alert notifying you when the summary page has finished, then please use the :code:`--email` flag followed by the email address you wish to get the information sent to.

Running with multiple result files
----------------------------------

:code:`PESummary` offers the opporunity to combine multiple results files into a single summary page. To pass multiple result files, simply list them after the :code:`--samples` named argument. 

.. code-block:: console

   $ summarypages --webdir /home/albert.einstein/public_html/projects/combing_results_files
                  --samples ./results_file.h5 ./results_file2.h5
    
As well as generating all derived posterior distributions for both results files, :code:`PESummary` will produce both all plots for each results file as well as comparison plots. Here, histograms showing the distributions for all parmeters that are common to both results files are shown. 

:code:`PESummary` will also generate a single `posterior_samples.h5` metafile containing all information about both runs. For information about the structure of this metafile, please refer to `understanding the results file <results_file.html>`_.

Adding to an existing webpage
-----------------------------

If you have already generated a summary page using :code:`PESummary`, you are able to add to this summary page by using the :code:`existing_webdir` named argument in replacement of the :code:`webdir` named argument. For clarity, let us assume that you have already ran the the :code:`summarypages` executable with two results files (`results_file.h5` and `results_file2.h5`) in the web directory `/home/albert.einstein/public_html/existing` and you would like to add a further results file (`results_file3.h5`) then you can do this with,

.. code-block:: console

   $ summarypages --existing_webdir /home/albert.einstein/public_html/existing
                  --samples ./results_file3.h5

Here, :code:`PESummary` will first derive all posterior samples available from `results_file3.h5`. It will then generate all plots for `results_file3.h5`. :code:`PESummary` will then read the `posterior_samples.h5` file located in the `/home/albert.einstein/public_html/LVC/existing/samples` directory to grab all samples from `results_file.h5` and `results_file2.h5`. Comparison plots will then be generated to compare all files and a new webpage is generated to show the information. Finally, the samples from `results_file3.h5` are incorporated into the `posterior_samples.h5` metafile.
