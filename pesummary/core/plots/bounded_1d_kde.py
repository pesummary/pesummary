# Licensed under an MIT style license -- see LICENSE.md

from pesummary.utils.bounded_1d_kde import (
    BoundedKDE as _BoundedKDE, TransformBoundedKDE as _TransformBoundedKDE,
    ReflectionBoundedKDE as _ReflectionBoundedKDE
)
from pesummary.utils.decorators import deprecation

@deprecation(
    "pesummary.core.plots.bounded_1d_kde.transform_logit has changed to "
    "pesummary.utils.bounded_1d_kde.transform_logit. This may not be "
    "supported in future releases. Please update."
)
def transform_logit(*args, **kwargs):
    from pesummary.utils.bounded_1d_kde import transform_logit
    return transform_logit(*args, **kwargs)

@deprecation(
    "pesummary.core.plots.bounded_1d_kde.inverse_transform_logit has changed "
    "to pesummary.utils.bounded_1d_kde.inverse_transform_logit. This may not "
    "be supported in future releases. Please update."
)
def inverse_transform_logit(*args, **kwargs):
    from pesummary.utils.bounded_1d_kde import inverse_transform_logit
    return inverse_transform_logit(*args, **kwargs)

@deprecation(
    "pesummary.core.plots.bounded_1d_kde.dydx_logit has changed to "
    "pesummary.utils.bounded_1d_kde.dydx_logit. This may not be supported in "
    "future releases. Please update."
)
def dydx_logit(*args, **kwargs):
    from pesummary.utils.bounded_1d_kde import dydx_logit
    return dydx_logit(*args, **kwargs)

@deprecation(
    "pesummary.core.plots.bounded_1d_kde.bounded_1d_kde has changed to "
    "pesummary.utils.bounded_1d_kde.bounded_1d_kde. This may not be supported "
    "in future releases. Please update."
)
def bounded_1d_kde(*args, **kwargs):
    from pesummary.utils.bounded_1d_kde import bounded_1d_kde
    return bounded_1d_kde(*args, **kwargs)


class BoundedKDE(_BoundedKDE):
    @deprecation(
        "pesummary.core.plots.bounded_1d_kde.BoundedKDE has changed to "
        "pesummary.utils.bounded_1d_kde.BoundedKDE. This may not be supported "
        "in future releases. Please update."
    )
    def __init__(self, *args, **kwargs):
        return super(BoundedKDE, self).__init__(*args, **kwargs)


class TransformBoundedKDE(_TransformBoundedKDE):
    @deprecation(
        "pesummary.core.plots.bounded_1d_kde.TransformBoundedKDE has changed "
        "to pesummary.utils.bounded_1d_kde.TransformBoundedKDE. This may not "
        "be supported in future releases. Please update."
    )
    def __init__(self, *args, **kwargs):
        return super(TransformBoundedKDE, self).__init__(*args, **kwargs)


class ReflectionBoundedKDE(_ReflectionBoundedKDE):
    @deprecation(
        "pesummary.core.plots.bounded_1d_kde.ReflectionBoundedKDE has changed "
        "to pesummary.utils.bounded_1d_kde.ReflectionBoundedKDE. This may not "
        "be supported in future releases. Please update."
    )
    def __init__(self, *args, **kwargs):
        return super(ReflectionBoundedKDE, self).__init__(*args, **kwargs)
