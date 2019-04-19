from pesummary.core import command_line
from pesummary.gw import insert_gwspecific_option_group
from pesummary.utils import functions
from .summaryplots import PlotGeneration

if __name__ == "__main__":
    main()


def main():
    parser = command_line()
    insert_gwspecific_option_group(parser)
    opts = parser.parse_args()
    func = functions()
    args = func["input"](opts)
    func["PlotGeneration"](args)
