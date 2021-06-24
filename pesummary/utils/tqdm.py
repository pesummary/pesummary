# Licensed under an MIT style license -- see LICENSE.md

from tqdm import tqdm as _tqdm
from tqdm.utils import _unicode
import time

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


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
        """Extension of the tqdm.status_printer function to allow for tqdm
        to interact with logger
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
        """Extension of the tqdm.format_dict property to add extra quantities
        """
        base = super(tqdm, self).format_dict
        if self.logger is not None:
            base.update(
                {"levelname": self.logger_level, "name": self.logger.name}
            )
        base.update({"asctime": time.strftime("%Y-%m-%d  %H:%M:%S")})
        return base

    def __str__(self):
        """Hack of the tqdm.__str__ function to prevent duplicating the entirety
        of the tqdm.display function
        """
        if hasattr(self, "display_msg") and self.display_msg is not None:
            return self.display_msg
        return super(tqdm, self).__str__()

    def display(self, msg=None, pos=None):
        """Extension of the tqdm.display function to allow for the time to be
        passed to the status_printer function
        """
        self.display_msg = msg
        _original_sp = self.sp
        self.sp = lambda _msg: _original_sp(
            _msg, time.strftime("%Y-%m-%d  %H:%M:%S")
        )
        _ = super(tqdm, self).display(msg=None, pos=pos)
        self.sp = _original_sp
        return _


def trange(*args, **kwargs):
    """
    A shortcut for tqdm(range(*args), **kwargs).
    """
    return tqdm(range(*args), **kwargs)
