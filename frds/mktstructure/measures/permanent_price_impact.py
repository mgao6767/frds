import numpy as np
import pandas as pd

# from .exceptions import *

name = "PermanentPriceImpact"
description = """
"""
vars_needed = {"Price", "Volume", "Mid Point", "Direction"}


def estimate(data: pd.DataFrame) -> np.ndarray:
    if not vars_needed.issubset(data.columns):
        raise MissingVariableError(name, vars_needed.difference(data.columns))

    price = data["Price"].to_numpy()
    volume = data["Volume"].to_numpy()
    direction = data["Direction"].to_numpy()
    dollar_volume = np.multiply(price, volume)
    signed_dollar_volume = np.multiply(direction, dollar_volume)

    # Find the total signed dollar volume and return per 1 second.
    timestamps = np.array(data.index, dtype="datetime64")
    last_ts, last_price = timestamps[0], price[0]
    bracket_sdv = 0
    bracket = last_ts + np.timedelta64(1, "s")
    rets, sdvs = [], []
    for idx, ts in enumerate(timestamps):
        if ts <= bracket:
            bracket_sdv += signed_dollar_volume[idx]
        else:
            ret = np.log(price[idx - 1]) - np.log(last_price)
            if not np.isnan(ret) and not np.isnan(bracket_sdv):
                rets.append(ret)
                sdvs.append(bracket_sdv)
            # Reset bracket
            bracket = ts + np.timedelta64(1, "s")
            last_price = price[idx]
            bracket_sdv = signed_dollar_volume[idx]

    # rets and sdvs store the returns and signed dollar volume per 1 second
    return rets, sdvs


if __name__ == "__main__":
    import pandas as pd
    from statsmodels.tsa.api import VAR

    df = pd.read_csv("./data/ABT.N/2021-02-16.signed.csv")

    rets, sdvs = estimate(df)

    df = pd.DataFrame({"returns": rets, "sdv": sdvs})
    model = VAR(df)
    results = model.fit(60)
    irf = results.irf(60)

    print(irf.cum_effects[-1][0][1] * 10_000)
    f = irf.plot_cum_effects(impulse="sdv", response="returns")

    from matplotlib import pyplot as plt

    plt.savefig("a.png")

    f = irf.plot_cum_effects(impulse="sdv", response="returns", orth=True)
    plt.savefig("b.png")
    print(irf.cum_effects[-1][0][1] * 10_000)
