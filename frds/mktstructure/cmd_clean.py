import argparse
import os
from datetime import datetime as dt

from .utils import _sort_and_rm_duplicates


def cmd_clean(args: argparse.Namespace):

    # sort by time and remove duplicates
    if args.all:
        for root, _, files in os.walk(args.data_dir):
            for f in files:
                path = os.path.join(root, f)
                if os.path.isfile(path):
                    print(f"Cleaning {path}")
                    _sort_and_rm_duplicates(path, replace=args.replace)
    else:
        for root, _, files in os.walk(args.data_dir):
            for f in files:
                path = os.path.join(root, f)
                ric, date = os.path.normpath(path).split(os.sep)[-2:]

                if not ric in args.ric:
                    continue
                date = dt.fromisoformat(
                    date.removesuffix(".csv")
                    .removesuffix(".csv.gz")
                    .removesuffix(".sorted")
                )
                if not (dt.fromisoformat(args.b) <= date <= dt.fromisoformat(args.e)):
                    continue

                if os.path.isfile(path):
                    print(f"Cleaning {path}")
                    _sort_and_rm_duplicates(path, replace=args.replace)
