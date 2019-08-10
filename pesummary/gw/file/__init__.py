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
from astropy.utils import iers


def check_IERS():
    """Check that the latest IERS data can be downloaded
    """
    try:
        iers.conf.auto_download = True
        iers_a = iers.IERS_Auto.open()
    except Exception:
        logger.warn("Unable to download latest IERS data. The bundled IERS-B "
                    "data which covers the time range from 1962 to just before "
                    "the astropy release dat will be used. Any transformations "
                    "outside of this range will not be allowed.")
        iers.conf.auto_download = False


check_IERS()
