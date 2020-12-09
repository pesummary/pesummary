Welcome to PESummary's gw package
=================================

PESummary's gw specific package contains GW functionality,
including converting posterior distributions, deriving event classifications and
GW specific plots.

Unified input/output
++++++++++++++++++++

.. toctree::
    :maxdepth: 1

    ../core/write
    read

Parameter definitions
+++++++++++++++++++++

.. toctree::
    :maxdepth: 1

    parameters

Manipulating posterior samples
++++++++++++++++++++++++++++++

.. toctree::
    :maxdepth: 1

    ../core/Array
    SamplesDict
    ../core/MCMCSamplesDict
    MultiAnalysisSamplesDict
    Conversion

Manipulating PDFs
+++++++++++++++++

.. toctree::
    :maxdepth: 1

    ../core/pdf
    ../core/ProbabilityDict
    ../core/ProbabilityDict2D

Manipulating PSD and Calibration data
+++++++++++++++++++++++++++++++++++++

.. toctree::
    :maxdepth: 1

    psd
    calibration

The pesummary metafile
++++++++++++++++++++++

.. toctree::
    :maxdepth: 2

    pesummary_file

Fetching data
+++++++++++++

.. toctree::
    :maxdepth: 2

    fetch

Waveforms
+++++++++

.. toctree::
    :maxdepth: 1

    tutorials/waveforms

Plotting
++++++++

.. toctree::
    :maxdepth: 1

    ../core/seaborn
    ../core/bounded_kdes
    violin
    tutorials/plotting_from_metafile
    tutorials/population_scatter_plot_GWTC-1
    tutorials/GWTC1_plots
    tutorials/interaction_with_ligo_skymap
    tutorials/plot_waveform_on_strain_data

Releasing samples
+++++++++++++++++

.. toctree::
    :maxdepth: 1

    tutorials/release_notebook

Executables
+++++++++++

PESummary boasts several executables to make it even easier to use, see below
for details:

.. toctree::
    :maxdepth: 1

    cli/summaryclassification
    ../core/cli/summaryclean
    ../core/cli/summarycombine
    ../core/cli/summarycompare
    cli/summarydetchar
    cli/summarygracedb
    ../core/cli/summarymodify
    ../core/cli/summarypages
    ../core/cli/summarypageslw
    cli/summarypipe
    ../core/cli/summarypublication
    cli/summaryrecreate
    cli/summaryreview
    ../core/cli/summaryversion

The webpages
++++++++++++

As the `summarypages <../core/cli/summarypages.html>`_ executable is the main
executable provided by `pesummary`, we will show and explain each of the
output pages. We will use the pages that were produced from Listing 5 in the
`pesummary paper <https://arxiv.org/pdf/2006.06639.pdf>`_ (`GW190412` webpages).

.. toctree::
    :maxdepth: 1

    summarypages/home
    summarypages/publication
    summarypages/logging
    summarypages/version
    summarypages/downloads
    summarypages/about
    summarypages/IMRPhenomPv3HM/home
    summarypages/IMRPhenomPv3HM/corner
    summarypages/IMRPhenomPv3HM/mass_1
    summarypages/IMRPhenomPv3HM/interactive
    summarypages/IMRPhenomPv3HM/classification
    summarypages/comparison/home
    summarypages/comparison/mass_1
    summarypages/comparison/interactive
    summarypages/watermark
