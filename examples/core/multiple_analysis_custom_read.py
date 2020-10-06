# Here we show how to read in a result file which cannot be read in using
# the pesummary builtin functions. This result file contains multiple
# analyses
import numpy as np
import h5py
from pesummary.core.file.formats.base_read import MultiAnalysisRead
from pesummary.io import read

# First let us create a dictionary containing some posterior samples
data = {
    "run_1": {
        "a": np.random.uniform(1, 5, 1000),
        "b": np.random.uniform(1, 5, 1000)
    }, "run_2": {
        "a": np.random.uniform(1, 2, 1000),
        "b": np.random.uniform(1, 2, 1000)
    }
}

# Next, lets make a file which cannot be read in with pesummary
f = h5py.File("example.h5", "w")
samples = f.create_group("my_posterior_samples")
a = samples.create_group("a")
b = samples.create_group("b")
a.create_dataset("run_1", data=data["run_1"]["a"])
a.create_dataset("run_2", data=data["run_2"]["a"])
b.create_dataset("run_1", data=data["run_1"]["b"])
b.create_dataset("run_2", data=data["run_2"]["b"])
f.close()

# We now show that it cannot be read in using the pesummary inbuilt functions
try:
    f = read("example.h5")
    raise
except Exception:
    print("Failed to read in 'example.h5' with the pesummary inbuilt functions")

# Next, lets define a class which inherits from the SingleAnalysisRead class
# that is able to extract the posterior samples from our custom result file


class CustomReadClass(MultiAnalysisRead):
    """Class to read in our custom file

    Parameters
    ----------
    path_to_results_file: str
        path to the result file you wish to read in
    """
    def __init__(self, path_to_results_file, **kwargs):
        super(CustomReadClass, self).__init__(path_to_results_file, **kwargs)
        self.load(self.custom_load_function)

    def custom_load_function(self, path, **kwargs):
        """Function to load data from a custom hdf5 file
        """
        import h5py

        f = h5py.File(path, 'r')
        parameters = list(f["my_posterior_samples"].keys())
        labels = list(f["my_posterior_samples"][parameters[0]].keys())
        samples = [
            np.array([
                f["my_posterior_samples"][param][label] for param in
                parameters
            ]).T for label in labels
        ]
        f.close()
        # Return a dictionary of data
        return {
            "parameters": [parameters] * 2, "samples": samples,
            "labels": labels
        }


# Now, lets read in the result file
f = read("example.h5", cls=CustomReadClass)
print("Read in with class: {}".format(f.__class__))

# Now lets confirm that we have extracted the samples correctly
np.testing.assert_almost_equal(data["run_1"]["a"], f.samples_dict["run_1"]["a"])
np.testing.assert_almost_equal(data["run_1"]["b"], f.samples_dict["run_1"]["b"])
np.testing.assert_almost_equal(data["run_2"]["a"], f.samples_dict["run_2"]["a"])
np.testing.assert_almost_equal(data["run_2"]["b"], f.samples_dict["run_2"]["b"])

# We can now use the builtin pesummary plotting methods
samples = f.samples_dict
fig, _, _, _ = samples.plot(["a", "b"], type="reverse_triangle")
fig.savefig("test.png")
