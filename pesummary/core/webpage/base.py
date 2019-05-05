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

import numpy as np


class Base():
    """Meta class containing helper functions for generating webpages
    """
    def close(self):
        """Close the opened html file.
        """
        self.html_file.close()

    def _check_content(self, content):
        """Make sure that the content has new line in string

        Parameters
        ----------
        content: str
            string that you want to check
        """
        if content[-1] != "\n":
            content += "\n"
        return content

    def add_content(self, content, indent=0):
        """Add content to the html page

        Parameters
        ----------
        content: str/list, optional
            either a single string or a list of string that you want to add to
            your html page
        indent: int, optional
            the indent of the line
        """
        if type(content) == list:
            for i in np.arange(len(content)):
                content[i] == self._check_content(content[i])
                self.html_file.write(" " * indent + content)
        else:
            content = self._check_content(content)
            self.html_file.write(" " * indent + content)

    def make_div(self, indent=0, _class=None, _style=None, _id=None):
        """Make a div of your choice

        indent: int, optional
            the indent of the line
        _class: str, optional
            the class name of your div
        _style: str, optional
            the style of your div
        _id: str, optional
            the id of your div
        """
        string = "<div"
        if _class:
            string += " class='%s'" % (_class)
        if _style:
            string += " style='%s'" % (_style)
        if _id:
            string += " id='%s'" % (_id)
        string += ">\n"
        self.add_content(string, indent=indent)

    def end_div(self, indent=0):
        """End a div of your choice

        Parameters
        ----------
        indent: int, optional
            the indent of the new line
        """
        self.add_content("</div>", indent)

    def make_container(self, style=None, indent=0):
        """Make a container for your webpage

        Parameters
        ----------
        indent: int, optional
            the indent of the new line
        """
        if not style:
            style = "margin-top:7.5em; margin-bottom:5em; background-color:#FFFFFF; " + \
                    "box-shadow: 0 0 5px grey;"
        self.make_div(indent, _class="container", _style=style)

    def end_container(self, indent=0):
        """End a container

         Parameters
         ----------
         indent: int, optional
             the indent of the new line
         """
        self.end_div(indent)
