# Copyright (C) 2018  Charlie Hoy <charlie.hoy@ligo.org>
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

from pesummary.utils.utils import logger
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn
from pesummary.gw.plots import violin
from pesummary.gw.plots.bounded_2d_kde import Bounded_2d_kde, default_bounds
from pesummary.gw.plots.cmap import colormap_with_fixed_hue
from pesummary.gw.file.conversions import mchirp_from_m1_m2, q_from_m1_m2
import numpy as np

seaborn.set(style="whitegrid")


def chirp_mass_and_q_from_mass1_mass2(pts):
    """Transform the component masses to chirp mass and mass ratio

    Parameters
    ----------
    pts: numpy.array
        array containing the mass1 and mass2 samples
    """
    pts = np.atleast_2d(pts)

    m1 = pts[:, 0]
    m2 = pts[:, 1]
    mc = mchirp_from_m1_m2(m1, m2)
    q = q_from_m1_m2(m1, m2)
    return np.column_stack([mc, q])


def estimate_2d_posterior(samples, xlow=None, xhigh=None, ylow=None,
                          yhigh=None, transform=None, gridsize=100):
    """Estimate a 2 dimensional posterior distribution

    Parameters
    ----------
    samples: nd list
        2d list of samples
    xlow: float
        bound the KDE to not take x values below xlow
    xhigh: float
        bound the KDE to not take x values above xhigh
    ylow: float
        bound the KDE to not take y values below ylow
    yhigh: float
        bound the KDE to not take y values above ylow
    transform: func
        function to transform the parameters
    gridsize: int
        the number of points to use when estimating the KDE
    """
    x = np.array(samples[0])
    y = np.array(samples[1])

    if transform is None:
        transform = lambda x: x

    deltax = 0.1 * (x.max() - x.min())
    deltay = 0.1 * (y.max() - y.min())
    x_pts = np.linspace(x.min() - deltax, x.max() + deltax, gridsize)
    y_pts = np.linspace(y.min() - deltay, y.max() + deltay, gridsize)
    xx, yy = np.meshgrid(x_pts, y_pts)

    positions = np.column_stack([xx.ravel(), yy.ravel()])
    pts = np.array([x, y]).T
    selected_indices = np.random.choice(len(pts), len(pts) // 2, replace=False)
    kde_sel = np.zeros(len(pts), dtype=bool)
    kde_sel[selected_indices] = True
    kde_pts = transform(pts[kde_sel])
    untransformed_den_pts = pts[~kde_sel]
    den_pts = transform(untransformed_den_pts)
    Nden = den_pts.shape[0]

    post_kde = Bounded_2d_kde(
        kde_pts, xlow=xlow, xhigh=xhigh, ylow=ylow, yhigh=yhigh)
    den = post_kde(den_pts)
    inds = np.argsort(den)[::-1]
    den = den[inds]

    z = np.reshape(post_kde(transform(positions)), xx.shape)
    return {'xx': xx, 'yy': yy, 'z': z, 'kde': den, 'kde_sel': kde_sel}


def twod_contour_plots(parameters, samples, labels, latex_labels, colors=None):
    """Generate 2d contour plots for a set of samples for given parameters

    Parameters
    ----------
    parameters: list
        names of the parameters that you wish to plot
    samples: nd list
        list of samples for each parameter
    labels: list
        list of labels corresponding to each set of samples
    latex_labels: dict
        dictionary of latex labels
    """
    logger.debug("Generating 2d contour plots for %s" % ("_and_".join(parameters)))
    if colors is None:
        palette = seaborn.color_palette(palette="pastel", n_colors=len(samples))
    else:
        palette = colors
    fig, ax1 = plt.subplots(nrows=1, ncols=1)
    transform = xlow = xhigh = ylow = yhigh = None
    for num, i in enumerate(samples):
        if parameters[0] in list(default_bounds.keys()):
            if "low" in list(default_bounds[parameters[0]].keys()):
                xlow = default_bounds[parameters[0]]["low"]
            if "high" in list(default_bounds[parameters[0]].keys()):
                if isinstance(default_bounds[parameters[0]]["high"], str):
                    if "mass_1" in default_bounds[parameters[0]]["high"]:
                        transform = chirp_mass_and_q_from_mass1_mass2
                        xhigh = 1.
                else:
                    xhigh = default_bounds[parameters[0]]["high"]
        if parameters[1] in list(default_bounds.keys()):
            if "low" in list(default_bounds[parameters[1]].keys()):
                ylow = default_bounds[parameters[1]]["low"]
            if "high" in list(default_bounds[parameters[1]].keys()):
                if isinstance(default_bounds[parameters[1]]["high"], str):
                    if "mass_1" in default_bounds[parameters[1]]["high"]:
                        transform = chirp_mass_and_q_from_mass1_mass2
                        yhigh = 1.
                else:
                    yhigh = default_bounds[parameters[1]]["high"]

        contour_data = estimate_2d_posterior(
            i, transform=transform, xlow=xlow, xhigh=xhigh, ylow=ylow,
            yhigh=yhigh)
        xx = contour_data['xx']
        yy = contour_data['yy']
        z = contour_data['z']
        den = contour_data['kde']
        Nden = len(den)
        kde_sel = contour_data['kde_sel']

        pts = np.array([i[0], i[1]]).T
        levels = [0.9]
        zvalues = np.empty(len(levels))
        for i, level in enumerate(levels):
            ilevel = int(np.ceil(Nden * level))
            ilevel = min(ilevel, Nden - 1)
            zvalues[i] = den[ilevel]
        zvalues.sort()

        cs = ax1.contour(
            xx, yy, z, levels=zvalues, colors=[palette[num]], linewidths=1.5)
        cs.collections[0].set_label('%s' % (labels[num]))
    if all("mass_1" in i or "mass_2" in i for i in parameters):

        reg = plt.Polygon([[0, 0], [0, 1000], [1000, 1000]], color='gray', alpha=0.75)
        ax1.add_patch(reg)
    ax1.set_xlabel(latex_labels[parameters[0]])
    ax1.set_ylabel(latex_labels[parameters[1]])
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=4, mode="expand", borderaxespad=0.)
    plt.tight_layout()
    return fig


def violin_plots(parameter, samples, labels, latex_labels):
    """Generate violin plots for a set of parameters and samples

    Parameters
    ----------
    parameters: str
        the name of the parameter that you wish to plot
    samples: nd list
        list of samples for each parameter
    labels: list
        list of labels corresponding to each set of samples
    latex_labels: dict
        dictionary of latex labels
    """
    logger.debug("Generating violin plots for %s" % (parameter))
    fig, ax1 = plt.subplots(nrows=1, ncols=1)
    ax1 = violin.violinplot(data=samples, palette="pastel", inner="line", cut=0,
                            outer="percent: 90", scale="width")
    ax1.set_xticklabels(labels)
    for label in ax1.get_xmajorticklabels():
        label.set_rotation(30)
    ax1.set_ylabel(latex_labels[parameter])
    plt.tight_layout()
    return fig


def spin_distribution_plots(parameters, samples, label, color):
    """Generate spin distribution plots for a set of parameters and samples

    Parameters
    ----------
    parameters: list
        list of parameters
    samples: nd list
        list of samples for each spin component
    label: str
        the label corresponding to the set of samples
    color: tuple
        tuple describing the color to be used for plotting
    """
    logger.debug("Generating spin distribution plots for %s" % (label))
    from matplotlib.ticker import FixedLocator, Formatter
    from matplotlib.projections import PolarAxes
    from matplotlib.transforms import Affine2D
    from matplotlib.patches import Wedge
    from matplotlib import patheffects as PathEffects
    from matplotlib.collections import PatchCollection
    from matplotlib.transforms import ScaledTranslation

    from mpl_toolkits.axisartist.grid_finder import MaxNLocator
    import mpl_toolkits.axisartist.floating_axes as floating_axes
    import mpl_toolkits.axisartist.angle_helper as angle_helper
    from matplotlib.colors import LinearSegmentedColormap

    cmap = colormap_with_fixed_hue(color)

    spin1 = samples[parameters.index("a_1")]
    spin2 = samples[parameters.index("a_2")]
    theta1 = samples[parameters.index("tilt_1")]
    theta2 = samples[parameters.index("tilt_2")]

    pts = np.array([spin1, theta1]).T
    selected_indices = np.random.choice(len(pts), len(pts) // 2, replace=False)
    kde_sel = np.zeros(len(pts), dtype=bool)
    kde_sel[selected_indices] = True
    kde_pts = pts[kde_sel]
    spin1 = Bounded_2d_kde(kde_pts, xlow=0, xhigh=.99, ylow=-1, yhigh=1)
    pts = np.array([spin2, theta2]).T
    selected_indices = np.random.choice(len(pts), len(pts) // 2, replace=False)
    kde_sel = np.zeros(len(pts), dtype=bool)
    kde_sel[selected_indices] = True
    kde_pts = pts[kde_sel]
    spin2 = Bounded_2d_kde(kde_pts, xlow=0, xhigh=.99, ylow=-1, yhigh=1)

    rs = np.linspace(0, .99, 25)
    dr = np.abs(rs[1] - rs[0])
    costs = np.linspace(-1, 1, 25)
    dcost = np.abs(costs[1] - costs[0])
    COSTS, RS = np.meshgrid(costs[:-1], rs[:-1])
    X = np.arccos(COSTS) * 180 / np.pi + 90.
    Y = RS

    scale = np.exp(1.0)
    spin1_PDF = spin1(
        np.column_stack([RS.ravel() + dr / 2, COSTS.ravel() + dcost / 2]))
    spin2_PDF = spin2(
        np.column_stack([RS.ravel() + dr / 2, COSTS.ravel() + dcost / 2]))
    H1 = np.log(1.0 + scale * spin1_PDF)
    H2 = np.log(1.0 + scale * spin2_PDF)
    vmin = 0.0
    vmax = np.log(1.0 + scale * 3.024)

    rect = 121

    tr = Affine2D().translate(90, 0) + Affine2D().scale(np.pi / 180., 1.) + \
        PolarAxes.PolarTransform()

    grid_locator1 = angle_helper.LocatorD(7)
    tick_formatter1 = angle_helper.FormatterDMS()
    grid_locator2 = MaxNLocator(5)
    grid_helper = floating_axes.GridHelperCurveLinear(
        tr, extremes=(0, 180, 0, .99),
        grid_locator1=grid_locator1,
        grid_locator2=grid_locator2,
        tick_formatter1=tick_formatter1,
        tick_formatter2=None)

    fig = plt.figure(figsize=(6, 6))
    ax1 = floating_axes.FloatingSubplot(fig, rect, grid_helper=grid_helper)
    fig.add_subplot(ax1)

    ax1.axis["bottom"].toggle(all=False)
    ax1.axis["top"].toggle(all=True)
    ax1.axis["top"].major_ticks.set_tick_out(True)

    ax1.axis["top"].set_axis_direction("top")
    ax1.axis["top"].set_ticklabel_direction('+')

    ax1.axis["left"].major_ticks.set_tick_out(True)
    ax1.axis["left"].set_axis_direction('right')
    dx = 7.0 / 72.
    dy = 0 / 72.
    offset_transform = ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    ax1.axis["left"].major_ticklabels.set(figure=fig,
                                          transform=offset_transform)

    patches = []
    colors = []
    for x, y, h in zip(X.ravel(), Y.ravel(), H1.ravel()):
        cosx = np.cos((x - 90) * np.pi / 180)
        cosxp = cosx + dcost
        xp = np.arccos(cosxp)
        xp = xp * 180. / np.pi + 90.
        patches.append(Wedge((0., 0.), y + dr, xp, x, width=dr))
        colors.append(h)

    p = PatchCollection(patches, cmap=cmap, edgecolors='face', zorder=10)
    p.set_clim(vmin, vmax)
    p.set_array(np.array(colors))
    ax1.add_collection(p)

    # Spin 2
    rect = 122

    tr_rotate = Affine2D().translate(90, 0)
    tr_scale = Affine2D().scale(np.pi / 180., 1.)
    tr = tr_rotate + tr_scale + PolarAxes.PolarTransform()

    grid_locator1 = angle_helper.LocatorD(7)
    tick_formatter1 = angle_helper.FormatterDMS()

    grid_locator2 = MaxNLocator(5)

    grid_helper = floating_axes.GridHelperCurveLinear(
        tr, extremes=(0, 180, 0, .99),
        grid_locator1=grid_locator1,
        grid_locator2=grid_locator2,
        tick_formatter1=tick_formatter1,
        tick_formatter2=None)

    ax1 = floating_axes.FloatingSubplot(fig, rect, grid_helper=grid_helper)
    ax1.invert_xaxis()
    fig.add_subplot(ax1)

    # Label angles on the outside
    ax1.axis["bottom"].toggle(all=False)
    ax1.axis["top"].toggle(all=True)
    ax1.axis["top"].set_axis_direction("top")
    ax1.axis["top"].major_ticks.set_tick_out(True)

    # Remove radial labels
    ax1.axis["left"].major_ticks.set_tick_out(True)
    ax1.axis["left"].toggle(ticklabels=False)
    ax1.axis["left"].major_ticklabels.set_visible(False)
    # Also have radial ticks for the lower half of the right semidisk
    ax1.axis["right"].major_ticks.set_tick_out(True)

    patches = []
    colors = []
    for x, y, h in zip(X.ravel(), Y.ravel(), H2.ravel()):
        cosx = np.cos((x - 90) * np.pi / 180)
        cosxp = cosx + dcost
        xp = np.arccos(cosxp)
        xp = xp * 180. / np.pi + 90.
        patches.append(Wedge((0., 0.), y + dr, xp, x, width=dr))
        colors.append(h)

    p = PatchCollection(patches, cmap=cmap, edgecolors='face', zorder=10)
    p.set_clim(vmin, vmax)
    p.set_array(np.array(colors))
    ax1.add_collection(p)

    # Event name top, spin labels bottom
    title = plt.text(0.16, 1.25, label, fontsize=18, horizontalalignment='center')
    S1_label = plt.text(1.25, -1.15, r'${S}_{1}/(Gm_1^2)$', fontsize=14)
    S2_label = plt.text(-.5, -1.15, r'${S}_{2}/(Gm_2^2)$', fontsize=14)
    plt.subplots_adjust(wspace=0.295)
    bbox_extra_artists = [title, S1_label, S2_label]
    return fig
