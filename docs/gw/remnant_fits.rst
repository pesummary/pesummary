========================
Calculating Remnant Fits
========================

PESummary has multiple methods for calculating the remnant properties of a
binary. These include averaging multiple fits tuned to numerical relativity, and
using waveform specific fitting functions. Below we go into more details about
how each of these can be used with the `summarypages` executable and/or
the `convert` function directly.

Average NR fits
---------------

By default, the average of the fits tuned to numerical relativity will always
be calculated for you. We input the spins defined at a given reference frequency
and return the `final_mass_non_evolved`, `final_spin_non_evolved` among other
quantities. However, we also offer the ability to evolve the spins up to the
ISCO frequency and input these to the fitting functions. The returned posterior
distributions are then `final_mass`, `final_spin` among others (note the lack
of `non_evolved` in the parameter name) and will consequently be more accurate
than the corresponding `non_evolved` quantities. The evolved spin fits
can be achieve with the following command `summarypages` executable:

.. code-block:: console

    $ summarypages --webdir ./evolved_spin --samples example.hdf5 \
                   --evolve_spins --gw

or:

.. code-block:: python

    >>> from pesummary.gw.conversions import convert
    >>> data = convert(data_table, evolve_spins='ISCO')


This then calls the following functions:

.. autofunction:: pesummary.gw.conversions.nrutils.bbh_final_mass_average

.. autofunction:: pesummary.gw.conversions.nrutils.bbh_final_spin_average_precessing

.. autofunction:: pesummary.gw.conversions.nrutils.bbh_final_spin_average_non_precessing

.. autofunction:: pesummary.gw.conversions.nrutils.bbh_peak_luminosity_average

NRSurrogate fits
----------------

Alternatively, we may use an NRSurrogate remnant fit to evolve the spins and
calculate the remnant properties. This can be done with the following
`summarypages` executable:

.. code-block:: console

    $ summarypages --webdir ./NRSurrogate --samples example.hdf5 \
                   --NRSur_fits --gw

By default, the `NRSur7dq4Remnant` is used. The user may choose their own
NRSurrogate remnant model by passing it from the command line:

.. code-block:: console

    $ summarypages --webdir ./NRSurrogate --samples example.hdf5 \
                   --NRSur_fits NRSur3dq8Remnant --gw

Using the `convert` function:

.. code-block:: python

    >>> data = convert(data_table, NRSur_fits='NRSur7dq4Remnant')

This then calls the following function:

.. autofunction:: pesummary.gw.conversions.nrutils.NRSur_fit

A list of available NRSurrogate remnant models can be found
`here <https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/eval__fits_8py_source.html>`_.

.. note::

    This will require the LAL_DATA_PATH to be set correctly. This will require
    cloning https://git.ligo.org/lscsoft/lalsuite-extra and pointing to
    lalsuite-extra/data/lalsimulation:

        export LAL_DATA_PATH=~/lalsuite-extra/data/lalsimulation

Waveform specific waveform fits
-------------------------------

Finally specific waveform approximants can be used to evaluate the remnant fits.
This means that the functions used when evaluating the waveform are directly
used for the posterior samples. For example, we can use the SEOBNRv4PHM
approximant to calculate the remnant properties with the following
`summarypages` executable:

.. code-block:: console

    $ summarypages --webdir ./waveform --samples example.hdf5 \
                   --waveform_fits --gw --approximant SEOBNRv4PHM

Sometimes, it can take a while to evaluate these fits for computationally
expensive waveform models. Therefore we offer parallelisation to reduce wall
time. This can be done with the following:

.. code-block:: console

    $ summarypages --webdir ./waveform --samples example.hdf5 \
                   --waveform_fits --gw --approximant SEOBNRv4PHM \
                   --multi_process 20


`f_low` is required for this conversion. If `f_low` cannot be extracted from the
result file, the code will return a ValueError and ask for an `f_low` to be
passed from the command line. This can be done with the following:

.. code-block:: console

    $ summarypages --webdir ./waveform --samples example.hdf5 \
                   --waveform_fits --gw --approximant SEOBNRv4PHM \
                   --multi_process 20 --f_low 3.0


Using the `convert` function:

    >>> data = convert(data_table, waveform_fits=True, approximant="SEOBNRv4PHM")

This then calls the following function:

.. autofunction:: pesummary.gw.conversions.remnant.final_mass_of_merger_from_waveform

.. autofunction:: pesummary.gw.conversions.remnant.final_spin_of_merger_from_waveform 
