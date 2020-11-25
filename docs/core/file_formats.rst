====================
Allowed file formats
====================

`pesummary` is able to read in a multitude of file formats through its
`universal read function <../io/read.html>`_. Below we document the allowed
file formats.

dat
---

A :code:`dat` file is read in through the `pesummary.core.file.formats.dat.read_dat`
function. This function loads a :code:`dat` file through the
`np.genfromtxt <https://numpy.org/devdocs/user/basics.io.genfromtxt.html>`_
function. By default, we assume :code:`delimiter=None`, meaning that each line is
split by a single or multiple white spaces. Each column is assumed to contain
samples drawn from a single posterior distribution.

.. autofunction:: pesummary.core.file.formats.dat.read_dat

csv
---

A :code:`csv` file is read in through the `pesummary.core.file.formats.csv.read_csv`
function. This function loads a :code:`csv` file through the
`np.genfromtxt <https://numpy.org/devdocs/user/basics.io.genfromtxt.html>`_
function with :code:`delimiter=','`. Each column is assumed to contain
samples drawn from a single posterior distribution.

.. autofunction:: pesummary.core.file.formats.csv.read_csv

json
----

A :code:`json` file is read in through the `pesummary.core.file.formats.json.read_json`
function. By default, we assume :code:`path_to_samples=None` meaning that the :code:`json`
file is recursively searched to find a `posterior` or `posterior_samples` group.
Each key of this group should be a parameter name and value, samples drawn from
the posterior distribution.

.. autofunction:: pesummary.core.file.formats.json.read_json

hdf5
----

A :code:`hdf5` file is read in through the `pesummary.core.file.formats.hdf5.read_hdf5`
function. By default, we assume :code:`path_to_samples=None` meaning that the :code:`hdf5`
file is recursively searched to find a `posterior` or `posterior_samples` group.
Within this group, posterior samples should be stored stored as a
`numpy structured array <https://numpy.org/doc/stable/user/basics.rec.html>`_
with columns corresponding to samples drawn from a single posterior distribution.

.. autofunction:: pesummary.core.file.formats.hdf5.read_hdf5

sql
---

An :code:`sql` database is read in through the
`pesummary.core.file.formats.sql.read_sql` function. If
:code:`path_to_samples=None` all tables in the :code:`sql` database are
extracted. Parameter names are the column descriptions and each
column corresponds to the samples drawn from a single posterior distribution.

.. autofunction:: pesummary.core.file.formats.sql.read_sql

npy
---

A :code:`npy` file is read in through the
`pesummary.core.file.formats.numpy.read_numpy` function. This function loads a
:code:`npy` file through the
`np.load <https://numpy.org/doc/stable/reference/generated/numpy.load.html>`_
function. The data stored in the :code:`npy` file must be a
`Structured array <https://numpy.org/doc/stable/user/basics.rec.html>`_
with each column assumed to contain samples drawn from a single posterior
distributions.

.. autofunction:: pesummary.core.file.formats.numpy.read_numpy

bilby
-----

A file produced by the `bilby <https://lscsoft.docs.ligo.org/bilby/>`_ parameter
estimation code, is read in through the
`pesummary.core.file.formats.bilby.read_bilby` function. We use the
`bilby.core.result.read_in_result <https://lscsoft.docs.ligo.org/bilby/bilby-output.html?highlight=read_in_result#reading-in-a-result-file>`_
function to load the file. Posterior samples are then extracted through the
`.posterior` property.

.. autofunction:: pesummary.core.file.formats.bilby.read_bilby
