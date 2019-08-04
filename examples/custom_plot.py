import matplotlib.pyplot as plt

# You need to define an '__all__' variable which lists all of the functions
# that you would like PESummary to evaluate

__single_plots__ = ["mass_1_histogram_plot"]
__comparison_plots__ = ["mass_1_comparison_histogram_plot"]

# PESummary will pass 2 variables to your function. A list of parameters
# and a list of samples. Therefore, the only arguments that your function can
# take are a list of parameters and an nd list of samples. Your function
# must then grab the information required from these arguments


def mass_1_histogram_plot(parameters, samples):
    """Function to plot a 1d histogram for mass_1

    Parameters
    ----------
    parameters: list
        list of parameters
    samples: nd list
        nd list of samples
    """
    # In order to plot a histogram for solely 'mass_1', you need to get the
    # samples from the nd list
    ind = parameters.index("mass_1")
    mass_1_samples = [i[ind] for i in samples]
    fig = plt.figure()
    plt.hist(mass_1_samples, bins=10, density=True)
    # You need to return a matplotlib.figure.Figure class. This figure will
    # then be saved by PESummary and added to the webpages
    return fig


def combine_mass_1_histogram_plot(parameters, samples):
    """Make a histogram of mass_1 for a set of different posteriors

    Parameters
    ----------
    parameters: list
        list of parameters
    samples: nd list
        nd list of samples for each posterior
    """
    fig = plt.figure()
    for ii in range(len(samples)):
        index = parameters.index("mass_1")
        mass_1_samples = samples[ii][index]
        plt.hist(mass_1_samples, bins=10, density=True, histtype='step')
    return fig
