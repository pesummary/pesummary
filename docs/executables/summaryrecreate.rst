===============
summaryrecreate
===============

The `summaryrecreate` executable is designed to allow the user to recreate
the analysis that was used to generate a PESummary metafile. This executable
reads the metafile, extracts the stored configuration file, psd and
calibration data, and launches a job with exactly the same settings. Of course,
if you would like, you are able to modify the configuration file with new/
improved settings. This can be done with the `--config_override` command line
argument. The launched job will then use this modified configuration file. This
makes it very easy for you to rerun a given analysis, but with a different
signal model, or psd for example.

To see help for this executable please run:

.. code-block:: console

    $ summaryrecreate --help

.. program-output:: summaryrecreate --help

GW190425 example
----------------

Below is an example on how to recreate the PhenomPNRT-HS analysis that was done
for GW190425 but changing the approximant, roq and webdir settings:

.. code-block:: console

    $ curl https://dcc.ligo.org/public/0165/P2000026/001/GW190425_posterior_samples.h5 -o GW190425_posterior_samples.h5
    $ PWD=`pwd`
    $ summaryrecreate --rundir ./rerun --samples GW190425_posterior_samples.h5 \
                      --config_override roq:False webdir:${PWD}/webpage approx:IMRPhenomPv2pseudoFourPN \
                      --labels PhenomPNRT-HS --code lalinference
