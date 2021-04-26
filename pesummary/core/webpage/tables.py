# Licensed under an MIT style license -- see LICENSE.md

from pesummary.core.webpage.base import Base

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class table_of_images(Base):

    def __init__(self, content, rows, columns, html_file, code, cli,
                 autoscale=False, unique_id=None, captions=None, extra_div=False,
                 mcmc_samples=False, margin_left=None, display=None,
                 container_id=None, close_container=True,
                 add_to_open_container=False):
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
        self.cli = cli
        self.captions = captions
        self.autoscale = autoscale
        self.unique_id = unique_id
        self.extra_div = extra_div
        self.mcmc_samples = mcmc_samples
        self.margin_left = margin_left
        self.display = display
        self.container_id = container_id
        self.close_container = close_container
        self.add_to_open_container = add_to_open_container
        if self.unique_id is not None:
            self.modal_id = "Modal_{}".format(self.unique_id)
            self.demo_id = "demo_{}".format(self.unique_id)
        else:
            self.modal_id = "MyModal"
            self.demo_id = "demo"
        self._add_scripts()

    def _add_scripts(self):
        self.add_content("<link rel='stylesheet' href='../css/image_styles.css'>\n")

    def _insert_image(self, path, width, indent, _id, justify="center"):
        string = "<img src='{}' alt='No image available' ".format(path) + \
                 "style='align-items:center; width:{}px;'".format(width)
        string += "id={} onclick='{}(\"{}\"".format(_id, self.code, _id)
        if self.code != "changeimage":
            string += ", \"{}\"".format(self.modal_id)
        if self.mcmc_samples:
            string += ", mcmc_samples=\"{}\"".format(self.mcmc_samples)
        string += ")'>\n"
        self.add_content(string, indent=indent)

    def make(self):
        if not self.add_to_open_container:
            self.add_content("<script>")
            self.add_content("$(document).ready(function(){", indent=2)
            self.add_content("$('[data-toggle=\"popover\"]').popover();", indent=4)
            self.add_content("});", indent=2)
            self.add_content("</script>")
            self.add_content("<style>")
            self.add_content(".popover {", indent=2)
            self.add_content("max-width: 550px;", indent=4)
            self.add_content("width: 550px;", indent=4)
            self.add_content("}", indent=2)
            self.add_content("</style>")
            self.make_container(
                display=self.display, container_id=self.container_id
            )
        self.make_div(2, _class="mx-auto d-block", _style=None)
        if self.rows == 1:
            _id = self.content[0][0].split("/")[-1][:-4]
            self._insert_image(self.content[0][0], 900, 4, _id, justify="left")
            self.make_div(2, _class=None, _style="float: right;")
            for num, i in enumerate(self.content[1]):
                _id = i.split("/")[-1][:-4]
                if self.margin_left is not None:
                    _style = "margin-left: {}".format(self.margin_left)
                else:
                    _style = None
                self.make_div(2, _class="row", _style=_style)
                self.make_div(4, _class="column")
                self.add_content("<a>", 6)
                self._insert_image(i, 415, 8, _id, justify=None)
                self.add_content("</a>", 6)
                if self.captions:
                    self.add_content("<div class='row justify-content-center'>", indent=6)
                    self.make_div(6, _class="col-sm-4", _style=None)
                    self.add_content("<button value='test' class='btn btn-info "
                                     "btn-xs' style='cursor: pointer' "
                                     "data-toggle='popover' "
                                     "data-placement='top' data-content='%s'>"
                                     "Caption</button>" % (self.captions[1][num]),
                                     indent=8)
                    self.end_div(6)
                    self.end_div(6)
                self.end_div(4)
                self.end_div(2)
            self.end_div(2)
        else:
            ind = 0
            width = "450"
            captions_margin_left = "-70"
            _class = "row justify-content-center"
            self.make_div(4, _class=_class, _style=None)
            if self.margin_left is not None:
                _style = "margin-left: {}".format(self.margin_left)
            else:
                _style = None
            self.make_div(6, _class="row", _style=_style)
            for idx, i in enumerate(self.content):
                if self.autoscale:
                    width = str(1350 / len(i))
                    captions_margin_left = str(-1350. / (9.5 * len(i)))
                self.make_div(8, _class="column", _style="padding-left: 1em;")
                for num, j in enumerate(i):
                    _id = j.split("/")[-1][:-4]
                    self.make_div(10, _class='container',
                                  _style=("display: inline-block; width: auto; "
                                          "padding: 0;"))
                    self.add_content(
                        "<a href='#%s' data-slide-to='%s'>\n" % (
                            self.demo_id, ind
                        ), indent=6)
                    self._insert_image(j, width, 8, _id, justify=None)
                    self.add_content("</a>\n", indent=6)
                    if self.cli or self.captions:
                        self.add_content("<div class='row justify-content-center'>", indent=6)
                        self.add_content(
                            "<div class='col-sm-{}'>".format(len(i) + 1), indent=6
                        )
                    if self.cli:
                        self.add_content("<button value='test' class='btn "
                                         "btn-info btn-xs' style='cursor: pointer' "
                                         "data-toggle='popover' data-placement='top' "
                                         "data-content='%s'>"
                                         "Command Line</button>" % (self.cli[idx][num]),
                                         indent=10)
                        self.end_div(10)
                    if self.captions:
                        if self.cli:
                            _style = "margin-left: {}px;".format(
                                captions_margin_left
                            )
                        else:
                            _style = None
                        self.make_div(
                            6, _class="col-sm-{}".format(len(i) + 1),
                            _style=_style
                        )
                        self.add_content("<button value='test' class='btn "
                                         "btn-info btn-xs' style='cursor: pointer' "
                                         "data-toggle='popover' "
                                         "data-placement='top' data-content='%s'>"
                                         "Caption</button>" % (self.captions[idx][num]),
                                         indent=10)
                        self.end_div(10)
                    self.end_div(10)
                    if self.extra_div:
                        self.end_div(10)
                    if self.cli or self.captions:
                        self.end_div(10)
                    ind += 1
                self.end_div(8)
            self.end_div(6)
            self.end_div(4)
        self.end_div(2)
        if self.close_container:
            self.end_container()
