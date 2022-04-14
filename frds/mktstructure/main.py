import argparse


from . import __description__, __version__


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION]...",
        description=__description__,
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"{parser.prog} version {__version__}",
    )
    # subparsers
    subparsers = parser.add_subparsers(
        title="Sub-commands",
        dest="command",
        description="Choose one from the following. Use `frds-mktstructure subcommand -h` to see help for each sub-command.",
    )
    parser_download = subparsers.add_parser(
        "download",
        description="Download data from Refinitiv Tick History",
        help="Download data from Refinitiv Tick History",
    )
    parser_clean = subparsers.add_parser(
        "clean",
        description="Clean the data in the data directory",
        help="Clean downloaded data",
    )
    parser_classify = subparsers.add_parser(
        "classify",
        description="Classify ticks into buy and sell orders using Lee and Ready (1991) algorithm",
        help="Classify ticks into buy and sell orders",
    )
    parser_compute = subparsers.add_parser(
        "compute",
        description="Compute specified measures",
        help="Compute market microstructure measures",
    )

    # subparser for `download` subcommand
    parser_download.add_argument(
        "-u",
        metavar="user",
        help="DataScope username",
    )
    parser_download.add_argument(
        "-p",
        metavar="password",
        help="DataScope password",
    )
    parser_download.add_argument(
        "-b",
        metavar="begin",
        default="2021-02-15",
        help="begin UTC date (YYYY-MM-DD)",
    )
    parser_download.add_argument(
        "-e",
        metavar="end",
        default="2021-02-28",
        help="end UTC date (YYYY-MM-DD)",
    )
    parser_download.add_argument(
        "-o",
        metavar="out",
        default="out.csv.gz",
        help="output file path",
    )
    parser_download.add_argument(
        "--ric",
        nargs="*",
        default=[],
        help="RIC of securities to process",
    )
    parser_download.add_argument(
        "--sp500",
        default=False,
        const=True,
        action="store_const",
        help="if set, process all S&P500 components (extending RIC list, if any)",
    )
    parser_download.add_argument(
        "--parse",
        default=False,
        const=True,
        action="store_const",
        help="if set, parse the downloaded raw data to output data directory",
    )
    parser_download.add_argument(
        "--data_dir",
        metavar="dir",
        default="./data",
        help="output data directory (used when --parse is set)",
    )

    # parser for `clean` subcommand
    parser_clean.add_argument(
        "--ric",
        nargs="*",
        default=[],
        help="RIC of securities to clean",
    )
    parser_clean.add_argument(
        "-b",
        metavar="begin",
        default="2021-02-15",
        help="begin UTC date (YYYY-MM-DD)",
    )
    parser_clean.add_argument(
        "-e",
        metavar="end",
        default="2021-02-28",
        help="end UTC date (YYYY-MM-DD)",
    )
    parser_clean.add_argument(
        "--data_dir",
        metavar="dir",
        help="data directory",
        required=True,
    )
    parser_clean.add_argument(
        "--all",
        default=False,
        const=True,
        action="store_const",
        help="if set, clean all data in the data director",
    )
    parser_clean.add_argument(
        "--replace",
        default=False,
        const=True,
        action="store_const",
        help="if set, replace raw data with cleaned data",
    )

    # parser for `classify` subcommand
    parser_classify.add_argument(
        "--ric",
        nargs="*",
        default=[],
        help="RIC of securities to clean",
    )
    parser_classify.add_argument(
        "-b",
        metavar="begin",
        default="2021-02-15",
        help="begin UTC date (YYYY-MM-DD)",
    )
    parser_classify.add_argument(
        "-e",
        metavar="end",
        default="2021-02-28",
        help="end UTC date (YYYY-MM-DD)",
    )
    parser_classify.add_argument(
        "--data_dir",
        metavar="dir",
        help="data directory",
        required=True,
    )
    parser_classify.add_argument(
        "--all",
        default=False,
        const=True,
        action="store_const",
        help="if set, classify all data in the data director",
    )

    # parser for `compute` subcommand
    parser_compute.add_argument(
        "--ric",
        nargs="*",
        default=[],
        help="RIC of securities to clean",
    )
    parser_compute.add_argument(
        "-b",
        metavar="begin",
        default="2021-02-15",
        help="begin UTC date (YYYY-MM-DD)",
    )
    parser_compute.add_argument(
        "-e",
        metavar="end",
        default="2021-02-28",
        help="end UTC date (YYYY-MM-DD)",
    )
    parser_compute.add_argument(
        "--data_dir",
        metavar="dir",
        help="data directory",
        required=True,
    )
    parser_compute.add_argument(
        "--all",
        default=False,
        const=True,
        action="store_const",
        help="if set, compute metrics for all data in the data director",
    )
    parser_compute.add_argument(
        "--out",
        metavar="out",
        help="file to save output results",
        required=True,
    )
    parser_compute.add_argument(
        "--bid_ask_spread",
        default=False,
        const=True,
        action="store_const",
        help="if set, compute the bid-ask spread",
    )
    parser_compute.add_argument(
        "--effective_spread",
        default=False,
        const=True,
        action="store_const",
        help="if set, compute the effective spread",
    )
    parser_compute.add_argument(
        "--realized_spread",
        default=False,
        const=True,
        action="store_const",
        help="if set, compute the realized spread",
    )
    parser_compute.add_argument(
        "--price_impact",
        default=False,
        const=True,
        action="store_const",
        help="if set, compute the price impact",
    )

    return parser


def main():
    parser = init_argparse()
    args = parser.parse_args()

    if args.command == "download":
        from .cmd_download import cmd_download

        cmd_download(args)

    if args.command == "clean":
        from .cmd_clean import cmd_clean

        cmd_clean(args)

    if args.command == "classify":
        from .cmd_classify import cmd_classify

        cmd_classify(args)

    if args.command == "compute":
        from .cmd_compute import cmd_compute

        cmd_compute(args)


if __name__ == "__main__":
    main()
