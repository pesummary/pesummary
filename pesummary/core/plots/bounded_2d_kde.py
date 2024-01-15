# Licensed under an MIT style license -- see LICENSE.md

from pesummary.utils.bounded_2d_kde import Bounded_2d_kde as _Bounded_2d_kde
from pesummary.utils.decorators import deprecation


class Bounded_2d_kde(_Bounded_2d_kde):
    @deprecation(
        "pesummary.core.plots.bounded_2d_kde.Bounded_2d_kde has changed to "
        "pesummary.utils.bounded_2d_kde.Bounded_2d_kde. This may not be "
        "supported in future releases. Please update."
    )
    def __init__(self, *args, **kwargs):
        return super(Bounded_2d_kde, self).__init__(*args, **kwargs)
