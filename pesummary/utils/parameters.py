# Copyright (C) 2020 Charlie Hoy <charlie.hoy@ligo.org>
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

from .list import List


class Parameters(List):
    """Class to store the list of parameters

    Parameters
    ----------
    *args: tuple
        all arguments are passed to the list class
    **kwargs: dict
        all kwargs are passed to the list class

    Attributes
    ----------
    added: list
        list of parameters that have been appended to the original list
    """
    def __init__(self, *args, **kwargs):
        super(Parameters, self).__init__(*args, **kwargs)
