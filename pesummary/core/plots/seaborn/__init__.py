# Licensed under an MIT style license -- see LICENSE.md

from pesummary.utils.utils import logger, import_error_msg
try:
    import seaborn
    SEABORN = True
except ImportError:
    SEABORN = False
    logger.warning(import_error_msg.format("seaborn"))

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]
