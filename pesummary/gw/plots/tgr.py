# Licensed under an MIT style license -- see LICENSE.md

import os

__author__ = [
    "Charlie Hoy <charlie.hoy@ligo.org>",
    "Aditya Vijaykumar <aditya.vijaykumar@ligo.org>",
]
DEFAULT_PLOT_KWARGS = dict(
    grid=True, smooth=4, type="triangle", fontsize=dict(label=20, legend=14),
    fig_kwargs=dict(wspace=0.2, hspace=0.2),
)


def imrct_plot(
    imrct_deviations,
    samples=None,
    evolve_spins=False,
    make_diagnostic_plots=False,
    inspiral_string="inspiral",
    postinspiral_string="postinspiral",
    cmap="YlOrBr",
    levels=[0.68, 0.95],
    level_kwargs={"colors": ["k", "k"]},
    xlabel=r"$\Delta M_{\mathrm{f}} / \bar{M}_{\mathrm{f}}$",
    ylabel=r"$\Delta a_{\mathrm{f}} / \bar{a}_{\mathrm{f}}$",
    _default_plot_kwargs=DEFAULT_PLOT_KWARGS,
    **plot_kwargs
):
    """Generate a triangle plot showing the IMR deviation parameters and
    diagnostic plots if specified.

    Parameters
    ----------
    imrct_deviations: ProbabilityDict2D
        Output of imrct_deviation_parameters_from_final_mass_final_spin
    samples: MultiAnalysisSamplesDict, optional
        Dictionary containing inspiral and postinspiral samples. Default None
    evolve_spins: Bool, optional
        if True, use the evolved spin remnant properties for the diagnostic
        plots. Default False
    make_diagnostic_plots: Bool, optional
        if True, generate 3 diagnostic triangle plots, one showing the
        a_1-a_2 posterior distribution, another showing the
        mass_1-mass_2 posterior distribution and another showing the
        remnant properties. Default False
    inspiral_string: str, optional
        The label of the inspiral analysis in the MultiAnalysisSamplesDict.
        Default 'inspiral'
    postinspiral_string: str, optional
        The label of the postinspiral analysis in the MultiAnalysisSamplesDict.
        Default 'postinspiral'
    cmap: str, optional
        The cmap to use when generating the IMRCT deviation triangle plot.
        Default 'YlOrBr'
    levels: tuple, optional
        The levels to plot in the IMRCT deviation triangle plot. Default
        [0.68, 0.95]
    level_kwargs: dict, optional
        Level kwargs to use in the IMRCT deviation triangle plot. Default
        {'colors': ['k', 'k']}
    xlabel: str, optional
        the xlabel to use in the IMRCT deviation triangle plot. Default
        r'$\Delta M_{\mathrm{f}} / \bar{M}_{\mathrm{f}}$'
    ylabel: str, optional
        the ylabel to use in the IMRCT deviation triangle plot. Default
        r'$\Delta a_{\mathrm{f}} / \bar{a}_{\mathrm{f}}$'
    plot_kwargs: dict, optional
        all additional kwargs passed to the IMRCT deviation triangle plot

    Returns
    -------
    figs: dict
        dictionary of figures. The IMRCT deviation plot has key
        'imrct_deviation' and diagnostic plots have keys
        'diagnostic_{}_{}'
    """
    figs = {}
    _plot_kwargs = _default_plot_kwargs.copy()
    _plot_kwargs.update(
        {
            "cmap": cmap,
            "levels": levels,
            "level_kwargs": level_kwargs,
            "xlabel": xlabel,
            "ylabel": ylabel,
        }
    )
    _plot_kwargs.update(plot_kwargs)
    fig, _ax1, ax_2d, _ax3 = imrct_deviations.plot(
        "final_mass_final_spin_deviations",
        **_plot_kwargs,
    )
    ax_2d.plot(0, 0, "k+", ms=12, mew=2)
    figs["imrct_deviation"] = [fig, _ax1, ax_2d, _ax3]
    if make_diagnostic_plots and samples is None:
        raise ValueError(
            "Please provide a MultiAnalysisSamplesDict object containing the "
            "posterior samples for the inspiral and postinspiral analysis"
        )
    elif make_diagnostic_plots:
        evolve_spins_string = ""
        if not evolve_spins:
            evolve_spins_string = "_non_evolved"

        samples_string = "final_{}" + evolve_spins_string
        plot_kwargs = _default_plot_kwargs.copy()
        plot_kwargs.update(
            {"fill_alpha": 0.2, "labels": [inspiral_string, postinspiral_string]}
        )
        parameters_to_plot = [
            [samples_string.format("mass"), samples_string.format("spin")],
            ["mass_1", "mass_2"],
            ["a_1", "a_2"],
        ]

        for parameters in parameters_to_plot:
            fig, ax1, ax2, ax3 = samples.plot(
                parameters,
                **plot_kwargs,
            )
            figs["diagnostic_{}_{}".format(*parameters)] = [fig, ax1, ax2, ax3]
    return figs


def make_and_save_imrct_plots(
    *args, webdir="./", plot_label=None, return_fig=False,
    save=True, **kwargs
):
    """Generate and save a triangle plot showing the IMR deviation parameters
    and diagnostic plots if specified.

    Parameters
    ----------
    *args: tuple
        all args passed to the imrct_plot function
    webdir: str, optional
        the directory to save the plots. Default "./"
    plot_label: str, optional
        label to prepend the
    return_fig: Bool, optional
        if True, return the figure and axes associated with the imrct_deviation
        plot
    save: Bool, optional
        if True, save the figures
    **kwargs: dict, optional
        all additional kwargs passed to the imrct_plot function
    """
    plotdir = os.path.join(webdir, "plots")
    if plot_label is not None:
        base_string = os.path.join(plotdir, "%s_imrct_{}.png" % (plot_label))
    else:
        base_string = os.path.join(plotdir, "imrct_{}.png")
    figs = imrct_plot(*args, **kwargs)
    if save:
        fig = figs["imrct_deviation"][0]
        fig.savefig(
            base_string.format("deviations_triangle_plot"), bbox_inches="tight"
        )
        fig.close()
        diagnostic_keys = [key for key in figs.keys() if "diagnostic" in key]
        for diag in diagnostic_keys:
            fig = figs[diag][0]
            save_string = "_".join(diag.split("_")[1:])
            fig.savefig(base_string.format(save_string))
            fig.close()
    if return_fig:
        return figs["imrct_deviation"]
