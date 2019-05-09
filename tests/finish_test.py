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

import os
import shutil

from pesummary.core.command_line import command_line
from pesummary.gw.command_line import insert_gwspecific_option_group
from pesummary.gw.inputs import GWInput
from pesummary.core.finish import FinishingTouches

import pytest


class TestFinishingTouches(object):

    def setup(self):
        if os.path.isdir("./.outdir"):
            shutil.rmtree("./.outdir")
        os.makedirs("./.outdir")
        self.parser = command_line()
        insert_gwspecific_option_group(self.parser)
        self.default_arguments = [
            "--webdir", "./.outdir",
            "--samples", "./tests/files/bilby_example.h5",
            "--email", "albert.einstein@ligo.org"]
        self.opts = self.parser.parse_args(self.default_arguments)
        self.inputs = GWInput(self.opts)
        self.finish = FinishingTouches(self.inputs)

    def test_email_message(self):
        assert "Your output page is ready on" in str(self.finish._email_message()) 
