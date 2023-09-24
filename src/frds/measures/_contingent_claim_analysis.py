from math import log, sqrt, exp
from typing import Tuple
from scipy.optimize import fsolve
from scipy.stats import norm


class ContingentClaimAnalysis:
    """:doc:`/measures/contingent_claim_analysis`"""

    def __init__(self) -> None:
        pass

    @staticmethod
    def estimate(
        equity: float,
        volatility: float,
        risk_free_rate: float,
        default_barrier: float,
        time_to_maturity: float,
        cds_spread: float,
    ) -> Tuple[float, float]:
        r"""Systemic risk based on contingent claim analysis (CCA).

        Args:
            equity (float): the market value of the equity of the firm.
            volatility (float): the volatility of equity.
            risk_free_rate (float): the risk-free rate in annualized terms.
            default_barrier (float): the face value of the outstandind debt at maturity.
            time_to_maturity (float): the time to maturity of the debt.
            cds_spread (float): the CDS spread for the firm.

        Returns:
            Tuple[float, float]: A tuple of put price and the firm's contribution to the systemic risk indicator (put price - CDS put price).
        """

        def cca_func(x, e, vol, rf, d, t):
            init_e, init_vol = x
            d1 = (log(pow(init_e, 2) / d) + (rf + (pow(init_vol, 4)) / 2) * t) / (
                pow(init_vol, 2) * sqrt(t)
            )
            d2 = d1 - pow(init_vol, 2) * sqrt(t)

            eqty = e - init_e**2 * norm.cdf(d1) + d * exp(-rf * t) * norm.cdf(d2)
            sigm = e * vol - init_e**2 * init_vol**2 * norm.cdf(d1)

            return eqty, sigm

        # We need to solve a system of non-linear equations for asset price and asset volatility
        # x = [equity, volatility]
        x = fsolve(
            cca_func,
            (equity, volatility),  # initial values set to equity and its volatility
            args=(
                equity,
                volatility,
                risk_free_rate,
                default_barrier,
                time_to_maturity,
            ),
        )

        # We solved for (asset price)^1/2 and (asset volatility)^1/2 to ensure the
        # values are positive. We recover asset price and asset volatility here.
        x = x**2

        #  Solve for implied price of put
        d1 = (
            log(x[0] / default_barrier)
            + (risk_free_rate + (x[1] ** 2) / 2) * time_to_maturity
        ) / (x[1] * sqrt(time_to_maturity))
        d2 = d1 - x[1] * sqrt(time_to_maturity)

        # The price of the put
        put_price = default_barrier * exp(
            -risk_free_rate * time_to_maturity
        ) * norm.cdf(-d2) - x[0] * norm.cdf(-d1)

        # Solve for price of CDS implied put
        # Risky debt
        debt = default_barrier * exp(-risk_free_rate * time_to_maturity) - put_price

        # The price of the CDS put option
        cds_put = (
            (
                1
                - exp(
                    -(cds_spread / 10000)
                    * (default_barrier / debt - 1)
                    * time_to_maturity
                )
            )
            * default_barrier
            * exp(-risk_free_rate * time_to_maturity)
        )

        return put_price, put_price - cds_put
