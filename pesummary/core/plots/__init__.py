# Licensed under an MIT style license -- see LICENSE.md

import matplotlib
from pesummary.utils.utils import _check_latex_install

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]
_check_latex_install(force_tex=True)
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974', '#64B5CD']
for code, color in zip("bgrmyck", colors):
    rgb = matplotlib.colors.colorConverter.to_rgb(color)
    matplotlib.colors.colorConverter.colors[code] = rgb
    matplotlib.colors.colorConverter.cache[code] = rgb
