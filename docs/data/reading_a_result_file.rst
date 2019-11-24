====================
Reading result files
====================

PESummary provides two functions for reading GW and non-GW specific result
files. The GW specific class maintains all the functionality of the core class
but adds a few extra features.

For details about reading in a PESummary metafile see
`Reading a PESummary metafile <reading_the_metafile.html>`_.

The `core` package
------------------

To read in a result file, the `pesummary.core.file.read.read` function should
be used,

.. code-block:: python

    >>> from pesummary.core.file.read import read
    >>> data = read("posterior_samples.txt")

Once read, you then have access to all the samples via the `samples_dict`
property

.. code-block:: python

    >>> print(data.samples_dict)
    idx    a    b    c    d
    0      10   20   30   40
    1      20   10   20   10
    .      .    .    .    .
    .      .    .    .    .
    14     40   30   10   20
    15     10   40   15   67
    >>> a = data.samples_dict["a"]
    >>> print(a)
    Array([10, 20, 30, 40, 10, 20, 35, 64, 82, 10, 45, 23, 76, 40, 10])
    >>> print(a.average(type="mean"))
    Array(34.3333333)
    >>> print(a.average(type="median"))
    Array(30.)

The `gw` package
----------------

You can read a GW result file with the `pesummary.core.file.read.read` function
above, but it is recommended to use the `pesummary.gw.file.read.read` function
as it provides some additional functionality. For instance, you are able to
calculate posteriors for all derived quantities by using the following,

.. code-block:: python

    >>> from pesummary.gw.file.read import read
    >>> data = read("posterior_samples.hdf5")
    >>> print(data.parameters)
    ['mass_1', 'mass_2', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'luminosity_distance', 'dec', 'ra', 'theta_jn', 'psi', 'phase', 'geocent_time', 'log_likelihood', 'log_prior']
    >>> data.generate_all_posterior_samples()
    >>> print(data.parameters)
    ['mass_1', 'mass_2', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'luminosity_distance', 'dec', 'ra', 'theta_jn', 'psi', 'phase', 'geocent_time', 'log_likelihood', 'log_prior', 'mass_ratio', 'total_mass', 'chirp_mass', 'symmetric_mass_ratio', 'iota', 'spin_1x', 'spin_1y', 'spin_1z', 'spin_2x', 'spin_2y', 'spin_2z', 'phi_1', 'phi_2', 'chi_eff', 'chi_p', 'cos_tilt_1', 'cos_tilt_2', 'redshift', 'comoving_distance', 'mass_1_source', 'mass_2_source', 'total_mass_source', 'chirp_mass_source', 'cos_theta_jn', 'cos_iota']

The `generate_all_posterior_samples` function simply passes the list of
parameters and samples to the `pesummary.gw.file.conversion._Conversion` class.
For details about this conversion module, see `here <conversion.html>`_.
