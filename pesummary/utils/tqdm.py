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

from tqdm import tqdm as _tqdm
from tqdm.utils import _unicode
import time


class tqdm(_tqdm):

    def __init__(self, *args, logger=None, logger_level="INFO", **kwargs):
        self.logger = logger
        self.logger_level = logger_level
        super(tqdm, self).__init__(*args, **kwargs)
        logger_prefix = '%(message)s'
        if self.logger is not None:
            logger_prefix = logger.handlers[0].formatter._fmt
        if not self.gui:
            self.sp = self.status_printer(
                self.fp, logger=self.logger, logger_prefix=logger_prefix,
                **self.format_dict
            )

    @staticmethod
    def status_printer(file, logger=None, logger_prefix='%(message)s', **kwargs):
        """
        Manage the printing and in-place updating of a line of characters.
        Note that if the string is longer than a line, then in-place
        updating may not work (it will print a new line at each refresh).
        """
        fp = file
        fp_flush = getattr(fp, 'flush', lambda: None)  # pragma: no cover

        def fp_write_log(s):
            logger.debug(_unicode(s))

        def fp_write(s):
            text = _unicode(s)
            fp.write(text)
            fp_flush()

        last_len = [0]

        def print_status(s, time=None):
            len_s = len(s)
            _message = s + (' ' * max(last_len[0] - len_s, 0))
            kwargs["message"] = _message
            if logger is not None:
                fp_write_log(_message)
            if time is not None:
                kwargs["asctime"] = time
            fp_write('\r' + logger_prefix % kwargs)
            last_len[0] = len_s

        return print_status

    @property
    def format_dict(self):
        """Public API for read-only member access."""

        base = dict(
            n=self.n, total=self.total,
            elapsed=self._time() - self.start_t
            if hasattr(self, 'start_t') else 0,
            asctime=time.strftime("%Y-%m-%d  %H:%M:%S"),
            ncols=self.dynamic_ncols(self.fp)
            if self.dynamic_ncols else self.ncols,
            prefix=self.desc, ascii=self.ascii, unit=self.unit,
            unit_scale=self.unit_scale,
            rate=1 / self.avg_time if self.avg_time else None,
            bar_format=self.bar_format, postfix=self.postfix,
            unit_divisor=self.unit_divisor)
        if self.logger is not None:
            base.update(
                {"levelname": self.logger_level, "name": self.logger.name}
            )
        return base

    def display(self, msg=None, pos=None):
        """
        Use `self.sp` to display `msg` in the specified `pos`.
        Consider overloading this function when inheriting to use e.g.:
        `self.some_frontend(**self.format_dict)` instead of `self.sp`.
        Parameters
        ----------
        msg  : str, optional. What to display (default: `repr(self)`).
        pos  : int, optional. Position to `moveto`
          (default: `abs(self.pos)`).
        """
        if pos is None:
            pos = abs(self.pos)

        nrows = self.nrows or 20
        if pos >= nrows - 1:
            if pos >= nrows:
                return False
            if msg or msg is None:  # override at `nrows - 1`
                msg = " ... (more hidden) ..."

        if pos:
            self.moveto(pos)
        _msg = self.__repr__() if msg is None else msg
        self.sp(_msg, time.strftime("%Y-%m-%d  %H:%M:%S"))
        if pos:
            self.moveto(-pos)
        return True


def trange(*args, **kwargs):
    """
    A shortcut for tqdm(range(*args), **kwargs).
    """
    return tqdm(range(*args), **kwargs)
