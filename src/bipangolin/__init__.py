"""biPangolin: per-base donor/acceptor splice site predictions from Pangolin.

The base Pangolin model collapses splice donor and acceptor predictions into
a single "splice usage" track. biPangolin trains small probes on Pangolin's
internal representations to recover the donor/acceptor distinction.

Quick start:
    from bipangolin import BiPangolinRunner
    runner = BiPangolinRunner()                    # auto-downloads weights
    result = runner.score_sequence("ACGT" * 500)
    result.probe_donor    # (L,) tensor of P(donor) per position
    result.probe_acceptor # (L,) tensor of P(acceptor) per position
"""
from .runner import (
    BiPangolinRunner,
    BiPangolinResult,
    one_hot_encode,
    selftest,
    CALIBRATION_SEQ,
    TISSUE_NAMES,
    NONE_CLASS, ACC_CLASS, DON_CLASS,
)
from ._variants import VariantScore

__version__ = "0.4.0"
__all__ = [
    "BiPangolinRunner", "BiPangolinResult", "VariantScore",
    "selftest", "CALIBRATION_SEQ",
    "one_hot_encode", "TISSUE_NAMES",
    "NONE_CLASS", "ACC_CLASS", "DON_CLASS",
]
