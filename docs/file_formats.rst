==============================
File formats that can be input
==============================

`PESummary` accepts multiple file formats for the results file. It determines which function to use based on the file extension. Below are details of the currently accepted file formats,

hdf5/h5 
-------

`PESummary` offers two functions for reading in HDF5 files. One using `deepdish` and one using `h5py`. Both functions require that your posterior samples are saved in a group called 'posterior' with the parameters saved as the keys. `PESummary` finds the path to the `posterior` group recursively so it does not care about the overall structure of your hdf5 file. In other words, your results file could be of the form `code/posterior_samples/posterior` or `posterior`. For instance, if you are using deepdish to save your results file, you would save it as the following:

.. code-block:: python

  >>> import deepdish
  >>> data = {"posterior": {"a": [1,2,3,4,5], "b": [1,2,3,4,5]}}
  >>> deepdish.io.save("posterior_samples.h5", data)

If you are using `h5py` to save your results file, you would save it as the following:

.. code-block:: python

  >>> import h5py
  >>> import numpy as np
  >>> f = h5py.File("posterior_samples.h5", "w")
  >>> data = np.array([(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)], dtype=[("a", "f"), ("b", "f")])
  >>> f.create_dataset("posterior", data=data)
  >>> f.close()

json
----

`PESummary` will also accept `json` files. We require that the posterior samples are saved in a group called `posterior` with the parameters saves the keys. `PESummary` finds the path to the `posterior` group recursively, so it does not care about the overall structure of the json file. In other words, your results file could be of the form `code/posterior_samples/posterior` or `posterior`. You can save your data to `json` by running,

.. code-block:: python

  >>> import json
  >>> data = {"posterior": {"a": [1,2,3,4,5], "b": [1,2,3,4,5]}}
  >>> with open("posterior_samples.json", "w") as f:
  ...    json.dump(data, f, indent=4, sort_keys=True)

dat
---

`PESummary` accepts `.dat` files as long as the data was saved by using the `saveastxt` function in the `numpy` package with the column names indicating the names of the parameters. You can save your data to `dat` by running,

.. code-block:: python

  >>> import numpy as np
  >>> data = [[1, 1], [2,2], [3,3], [4,4], [5,5]]
  >>> np.savetxt("posterior_samples.dat", np.array(data), fmt="%s", delimiter=" ", header="a b", comments="")
