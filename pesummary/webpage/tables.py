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

from pesummary.webpage.base import Base


class table_of_images(Base):

    def __init__(self, content, rows, columns, html_file, code):
        """

        Parameters
        ----------
        content: nd list
            nd list containing the image paths that you want to include in your
            table. Sorted by columns [[column1], [column2]]
        """
        self.content = content
        self.rows = rows
        self.columns = columns
        self.html_file = html_file
        self.code = code
        self._add_scripts()

    def _add_scripts(self):
        self.add_content("<link rel='stylesheet' href='../css/image_styles.css'>\n")

    def _insert_image(self, path, width, indent, _id, justify="center"):
        string = "<img src='{}' alt='No image available' ".format(path) + \
                 "style='align-items:center; width:{}px;'".format(width) + \
                 "id={} onclick='{}(\"{}\")'>\n".format(_id, self.code, _id)
        self.add_content(string, indent=indent)

    def make(self):
        self.make_container()
        self.make_div(2, _class="mx-auto d-block", _style=None)
        if self.rows == 1:
            _id = self.content[0][0].split("/")[-1][:-4]
            self._insert_image(self.content[0][0], 750, 4, _id, justify="left")
            self.make_div(2, _class=None, _style="float: right;")
            for i in self.content[1]:
                _id = i.split("/")[-1][:-4]
                self.make_div(2, _class="row")
                self.make_div(4, _class="column")
                self.add_content("<a>", 6)
                self._insert_image(i, 375, 8, _id, justify=None)
                self.add_content("</a>", 6)
                self.end_div(4)
                self.end_div(2)
            self.end_div(2)
        else:
            ind = 0
            width = "280"
            _class = "row justify-content-center"
            self.make_div(4, _class=_class, _style=None)
            self.make_div(6, _class="row", _style=None)
            for i in self.content:
                self.make_div(8, _class="column", _style=None)
                for num, j in enumerate(i):
                    _id = j.split("/")[-1][:-4]
                    self.add_content(
                        "<a href='#demo' data-slide-to='%s'>\n" % (ind), indent=6)
                    self._insert_image(j, width, 8, _id, justify=None)
                    self.add_content("</a>\n", indent=6)
                    ind += 1
                self.end_div(8)
            self.end_div(6)
            self.end_div(4)
        self.end_div(2)
        self.end_container()
