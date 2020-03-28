==============================================
Making a result file compatible with PESummary
==============================================

In order to first use PESummary, you need a result file which is compatible.
The current allowed formats are `hdf5/h5`, `JSON`, `dat` and `txt` but if you
would prefer to have a file format that is not one of these, raise an `issue`_
and we will add this feature!

.. _issue: https://git.ligo.org/lscsoft/pesummary/issues

hdf5/h5
-------

`PESummary` offers two functions for reading in HDF5 files. One using `deepdish`
and one using `h5py`. Both functions require that your posterior samples are
saved in a group called 'posterior' or 'posterior_samples' with the parameters
saved as the keys. `PESummary` finds the path to the `posterior` group
recursively so it does not care about the overall structure of your hdf5 file.
For instance, if you are using deepdish to save your results file, you would
save it as the following:

.. code-block:: python

  >>> import deepdish
  >>> data = {"posterior": {"a": [1,2,3,4,5], "b": [1,2,3,4,5]}}
  >>> deepdish.io.save("posterior_samples.h5", data)

If you are using `h5py` to save your results file, you would save it as the
following:

.. code-block:: python

  >>> import h5py
  >>> import numpy as np
  >>> f = h5py.File("posterior_samples.h5", "w")
  >>> data = np.array(
  ...     [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)],
  ...     dtype=[("a", "f"), ("b", "f")]
  ... )
  >>> f.create_dataset("posterior", data=data)
  >>> f.close()

json
----

`PESummary` will also accept `json` files. As with `hdf5` files, PESummary
requires that the posterior samples are saved in a group called 'posterior'
or 'posterior_samples'. You can save your data to `json` by running,

.. code-block:: python

  >>> import json
  >>> data = {"posterior": {"a": [1,2,3,4,5], "b": [1,2,3,4,5]}}
  >>> with open("posterior_samples.json", "w") as f:
  ...    json.dump(data, f, indent=4, sort_keys=True)

dat/txt
-------

`PESummary` accepts `.dat`/`.txt` files if they have been saved with column
names indicating the names of the parameters and each row containing a single
sample. For instance, you can save your data to `dat` by using the `numpy`
package with the following,

.. code-block:: python

  >>> import numpy as np
  >>> parameters = ["a", "b"]
  >>> data = [[1, 1], [2,2], [3,3], [4,4], [5,5]]
  >>> np.savetxt("posterior_samples.dat", np.array(data), fmt="%s", delimiter=" ", header=" ".join(parameters), comments="")
