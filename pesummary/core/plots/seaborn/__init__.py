# Licensed under an MIT style license -- see LICENSE.md

from pesummary.utils.utils import logger, import_error_msg
try:
    from packaging import version
    import seaborn
    if version.parse(seaborn.__version__) < version.parse("0.13.0"):
        logger.warning(
            "An old version of 'seaborn' has been found. This version of "
            "'pesummary' is only compatible with 'seaborn >= 0.13.0'. Please "
            "update"
        )
        SEABORN = False
    else:
        SEABORN = True
except ImportError:
    SEABORN = False
    logger.warning(import_error_msg.format("seaborn"))

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]
