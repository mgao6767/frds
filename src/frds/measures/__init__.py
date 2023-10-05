"""
The :mod:`frds.measures` module includes the collection of measures.
For a complete documentation, see :doc:`/measures/index`.
"""

from ._absorption_ratio import AbsorptionRatio
from ._contingent_claim_analysis import ContingentClaimAnalysis
from ._distress_insurance_premium import DistressInsurancePremium
from ._long_run_mes import LongRunMarginalExpectedShortfall
from ._marginal_expected_shortfall import MarginalExpectedShortfall
from ._srisk import SRISK
from ._systemic_expected_shortfall import SystemicExpectedShortfall

from ._option_price import blsprice
from ._z_score import z_score

LRMES = LongRunMarginalExpectedShortfall

__all__ = [
    # classes
    "AbsorptionRatio",
    "ContingentClaimAnalysis",
    "DistressInsurancePremium",
    "LongRunMarginalExpectedShortfall",
    "LRMES",
    "MarginalExpectedShortfall",
    "SRISK",
    "SystemicExpectedShortfall",
    # functions
    "blsprice",
    "z_score",
]
