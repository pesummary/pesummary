#! /usr/bin/env python

# Licensed under an MIT style license -- see LICENSE.md

from pesummary.core.cli.parser import ArgumentParser as _ArgumentParser

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class ArgumentParser(_ArgumentParser):
    def _pesummary_options(self):
        options = super(ArgumentParser, self)._pesummary_options()
        options.update(
            {
                "--id": {
                    "required": True,
                    "help": "The GraceDB id of the event you are interested in",
                },
                "--info": {
                    "nargs": "+",
                    "help": "Specific information you wish to retrieve",
                },
                "--output": {
                    "short": "-o",
                    "help": "Output json file you wish to save the data to",
                },
            }
        )
        return options


def main(args=None):
    """
    """
    from pesummary.gw.gracedb import get_gracedb_data

    _parser = ArgumentParser(description=__doc__)
    _parser.add_known_options_to_parser(["--id", "--info", "--output"])
    opts, unknown = _parser.parse_known_args(args=args)
    data = get_gracedb_data(opts.id, info=opts.info)
    if opts.output is not None:
        import json

        with open(opts.output, "w") as f:
            json.dump(data, f)
        return
    print(data)


if __name__ == "__main__":
    main()
