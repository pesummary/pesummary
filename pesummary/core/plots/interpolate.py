# Licensed under an MIT style license -- see LICENSE.md

from pesummary.utils.decorators import deprecation
from pesummary.utils.interpolate import BoundedInterp1d

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class Bounded_interp1d(BoundedInterp1d):
    @deprecation(
        "pesummary.core.plots.interpolate.Bounded_interp1d has changed to "
        "pesummary.utils.interpolate.BoundedInterp1d. This may not be supported "
        "in future releases. Please update."
    )
    def __init__(self, *args, **kwargs):
        return super(Bounded_interp1d, self).__init__(*args, **kwargs)
