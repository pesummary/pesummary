from pesummary.gw.fetch import fetch_open_samples
import matplotlib.pyplot as plt
import requests
import time

TESTING = True


def generate_skymap(samples, **kwargs):
    """Generate a skymap from a SamplesDict object

    Parameters
    ----------
    samples: pesummary.utils.samples_dict.SamplesDict
        samples you wish to generate a skymap for
    **kwargs: dict, optional
        all additional kwargs are passed to the `.plot()` method
    """
    return samples.plot(type="skymap", **kwargs)


f = fetch_open_samples(
    "GW190814", catalog="GWTC-2", unpack=True, path="GW190814.h5"
)
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
if TESTING:
    import multiprocessing

    p = multiprocessing.Process(target=generate_skymap, args=(posterior_samples,))
    p.start()
    # Only let the skymap generation run for 120s
    time.sleep(120)
    p.terminate()
    p.join()
else:
    fig = posterior_samples.plot(type="skymap")
    plt.show()
