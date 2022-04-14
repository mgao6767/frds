import argparse
import os
from datetime import datetime as dt

import pandas as pd

from .utils import lee_and_ready


def cmd_classify(args: argparse.Namespace):

    for root, _, files in os.walk(args.data_dir):
        for f in files:
            # skip those signed ones
            # TODO: .csv vs .csv.gz
            if "signed" in f:
                continue

            path = os.path.join(root, f)
            ric, date = os.path.normpath(path).split(os.sep)[-2:]

            # if `--all` flag is set
            if not args.all:
                if ric not in args.ric:
                    continue
                date = dt.fromisoformat(
                    date.removesuffix(".csv")
                    .removesuffix(".csv.gz")
                    .removesuffix(".sorted")
                )
                if not (dt.fromisoformat(args.b) <= date <= dt.fromisoformat(args.e)):
                    continue

            if os.path.isfile(path):
                print(f"Classify trades for {path}")
                df = pd.read_csv(path)
                df_signed = lee_and_ready(df)
                df_signed.to_csv(path.replace(".csv", ".signed.csv"))
                del df
