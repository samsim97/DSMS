"""
Reconstruction Module
=====================

Contains filters for converting the 1-bit bitstream back to analog. 

IMPORTANT: Match filter order to modulator order!
============================================================
- 1st order DSM → 1 RC stage (20 dB/decade roll-off)
- 2nd order DSM → 2 RC stages (40 dB/decade roll-off)
- 3rd order DSM → 3 RC stages (60 dB/decade roll-off)
- Nth order DSM → N RC stages (N×20 dB/decade roll-off)

The first-order RC filter is insufficient for higher-order modulators
because it only attenuates at 20 dB/decade, while higher-order DSMs
push noise up at 40, 60, 80 dB/decade respectively. 

Use CascadedRCLowPassFilter or create_filter_for_modulator_order()
for proper reconstruction with higher-order modulators.
"""

from .ideal_low_pass_filter import IdealLowPassFilter
from .cascaded_rc_low_pass_filter import (
    CascadedRCLowPassFilter,
    create_filter_for_modulator_order
)

__all__ = [
    "FirstOrderRCLowPassFilter",
    "IdealLowPassFilter", 
    "CascadedRCLowPassFilter",
    "create_filter_for_modulator_order"
]