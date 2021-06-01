from pesummary.utils.samples_dict import MultiAnalysisSamplesDict

# Define a dictionary of kwargs which are used to load the result files. These
# kwargs are passed directly to the `pesummary.io.read` function.
load_kwargs = {
    "princeton": dict(file_format="princeton"),
    "pycbc": dict(path_to_samples="samples")
}

# Load the data into a MultiAnalysisSamplesDict class using the `.from_files`
# class method. Here we are passing URLs. PESummary will download the file
# from the provided URL automatically. If you have already downloaded the
# result files, you can simply pass the path to the file
data = MultiAnalysisSamplesDict.from_files(
    {
        "princeton": "https://github.com/jroulet/O2_samples/raw/master/GW150914.npy",
        "pycbc": (
            "https://github.com/gwastro/2-ogc/raw/master/posterior_samples/"
            "H1L1V1-EXTRACT_POSTERIOR_150914_09H_50M_45UTC-0-1.hdf"
        ),
        "GWTC-1": "https://dcc.ligo.org/public/0157/P1800370/005/GW150914_GWTC-1.hdf5"
    }, **load_kwargs
)

# For some cases, the result file might be missing quantities you wish to
# compare. We therefore run the `.generate_all_posterior_samples` function
# (which runs PESummary's conversion module) to calculate alternative quantities
for key, samples in data.items():
    samples.generate_all_posterior_samples()

# Next we generate a `reverse_triangle` plot which compares the mass ratio
# and effective spin posterior samples
fig, _, _, _ = data.plot(["mass_ratio", "chi_eff"], type="reverse_triangle", module="gw")
fig.savefig("comparison_for_GW150914.png")
fig.close()
