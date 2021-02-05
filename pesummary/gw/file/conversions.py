# Licensed under an MIT style license -- see LICENSE.md

from pesummary.utils.decorators import deprecation

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class _Conversion(object):
    @deprecation(
        "The _Conversion class will be deprecated in future releases. Please "
        "use pesummary.gw.conversions.convert"
    )
    def __new__(cls, *args, **kwargs):
        from pesummary.gw.conversions import convert
        return convert(*args, **kwargs)
