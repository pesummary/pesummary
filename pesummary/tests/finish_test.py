# Licensed under an MIT style license -- see LICENSE.md

import shutil
import os
import pytest

from .base import make_argparse
from pesummary.core.finish import FinishingTouches

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


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
