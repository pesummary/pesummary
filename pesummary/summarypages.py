from pesummary.core import command_line
from pesummary.gw import insert_gwspecific_option_group

if __name__ == "__main__":
    parser = command_line()
    insert_gwspecific_option_group(parser)
    opts = parser.parse_args()
    print(opts.webdir, opts.psd)
