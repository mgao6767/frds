import argparse
import gzip
import os
from shutil import copyfileobj

from .utils import extract_index_components_ric
from .utils import SP500_RIC, NASDAQ_RIC, NYSE_RIC
from .trth import Connection
from .trth_parser import parse_to_data_dir


def cmd_download(args: argparse.Namespace):

    start_date = f"{args.b}T00:00:00.000Z"
    end_date = f"{args.e}T00:00:00.000Z"

    print("Connecting to TRTH...")
    trth = Connection(args.u, args.p, progress_callback=print)

    if args.sp500:
        args.ric.extend(
            extract_index_components_ric(
                trth.get_index_components([SP500_RIC], start_date, end_date),
                mkt_index=[SP500_RIC],
            )
        )

    data = trth.get_table(args.ric, start_date, end_date)

    print(f"Saving data to {args.o}...")

    trth.save_results(data, args.o)

    print("Downloading finished.")

    if args.parse:

        print("Decompressing downloaded data.")
        with gzip.open(args.o, "rb") as fin, open("__tmp.csv", "wb") as fout:
            copyfileobj(fin, fout)

        print("Parsing downloaded raw data.")
        parse_to_data_dir("__tmp.csv", args.data_dir, "1")

        os.remove("__tmp.csv")
