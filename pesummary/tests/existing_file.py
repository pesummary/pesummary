# Licensed under an MIT style license -- see LICENSE.md

import argparse
import pesummary

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


def command_line():
    """Generate an Argument Parser object to control the command line options
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-f", "--file", help="Result file you wish to test")
    parser.add_argument(
        "-t", "--type", help="The class you expect to be used to load the file",
        default="pesummary.gw.file.formats.pesummary.PESummary"
    )
    return parser


def test_load_pesummary(file):
    """Load a file using PESummary and check that we can access the properties
    """
    from pesummary.gw.file.read import read

    f = read(file)
    assert isinstance(f, pesummary.gw.file.formats.pesummary.PESummary)
    samples = f.samples_dict
    assert isinstance(samples, dict)
    labels = f.labels
    assert isinstance(labels, list)
    config = f.config
    assert isinstance(config, dict)
    calibration = f.calibration
    assert isinstance(calibration, dict)
    psd = f.psd
    assert isinstance(psd, dict)


def test_load(file, _type):
    """Load a file using PESummary and check that we can access the properties
    """
    from pesummary.gw.file.read import read

    f = read(file)
    assert isinstance(f, eval(_type))
    samples = f.samples_dict
    assert isinstance(samples, dict)


parser = command_line()
opts = parser.parse_args()
if opts.type == "pesummary.gw.file.formats.pesummary.PESummary":
    test_load_pesummary(opts.file)
else:
    test_load(opts.file, _type=opts.type)
