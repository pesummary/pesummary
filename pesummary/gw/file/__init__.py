# Licensed under an MIT style license -- see LICENSE.md

from pesummary.utils.utils import logger
from astropy.utils import iers

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


def check_IERS():
    """Check that the latest IERS data can be downloaded
    """
    try:
        iers.conf.auto_download = True
        iers_a = iers.IERS_Auto.open()
    except Exception:
        logger.warning("Unable to download latest IERS data. The bundled IERS-B "
                       "data which covers the time range from 1962 to just before "
                       "the astropy release dat will be used. Any transformations "
                       "outside of this range will not be allowed.")
        iers.conf.auto_download = False


check_IERS()
