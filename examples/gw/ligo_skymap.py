from pesummary.io import read
import matplotlib.pyplot as plt

f = read("posterior_samples.h5", package="gw")
label = f.labels[0]

# If the pesummary file has the skymap data already
# stored in the metafile, we may directly use the plot
# function
try:
    skymap = f.skymap[label]
    fig = skymap.plot()
    plt.show()
except (KeyError, AttributeError):
    print("No skymap data stored in the metafile")

# Otherwise, we may produce a skymap on the fly using
# the posterior samples already stored. This may take
# some time
posterior_samples = f.samples_dict[label]
fig = posterior_samples.plot(type="skymap")
plt.show()
