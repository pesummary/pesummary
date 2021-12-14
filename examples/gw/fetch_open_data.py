from pesummary.gw.fetch import fetch_open_samples
import os

# For GWTC-1, the posterior samples for a given event are stored in a h5 file.
# We can download the data by simply specifying the event name
data = fetch_open_samples("GW150914")
print(data)

# For GWTC-2, the LIGO/Virgo/Kagra collaboration releases a tarball for each
# event which contains multiple files. We may download and unpack the tarball
# with
path_to_directory = fetch_open_samples(
    "GW190412", catalog="GWTC-2", unpack=True, read_file=False,
    delete_on_exit=False, outdir="./"
)
import glob
print(glob.glob(os.path.join(path_to_directory, "*")))

# If we wanted to open a specific file within the tarball, we may specify the
# file with the path kwarg
data = fetch_open_samples(
    "GW190412", catalog="GWTC-2", unpack=True, read_file=True,
    delete_on_exit=False, outdir="./", path="GW190412.h5"
)
print(data)
