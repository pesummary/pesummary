import argparse
import pesummary


def command_line():
    """Generate an Argument Parser object to control the command line options
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-f", "--file", help="Result file you wish to test")
    return parser


def test_load(file):
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


parser = command_line()
opts = parser.parse_args()
test_load(opts.file)
