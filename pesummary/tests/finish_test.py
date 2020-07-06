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

import shutil
import os
import pytest

from .base import make_argparse
from pesummary.core.finish import FinishingTouches


class TestFinishingTouches(object):
    """Class to test pesummary.core.finish.FinishingTouches
    """
    def setup(self):
        """Setup the pesummary.core.finish.FinishingTouches class
        """
        if not os.path.isdir(".outdir"):
            os.mkdir(".outdir")
        opts, inputs = make_argparse()
        self.finish = FinishingTouches(inputs)

    def teardown(self):
        """Remove the any files generated
        """
        if os.path.isdir(".outdir"):
            shutil.rmtree(".outdir")

    def test_default_message(self):
        """Test the default email message
        """
        message = self.finish._email_message()
        assert message is not None

    def test_custom_message(self):
        """Test a custom email message
        """
        custom_message = "This is a test message"
        message = self.finish._email_message(message=custom_message)
        assert message == custom_message
