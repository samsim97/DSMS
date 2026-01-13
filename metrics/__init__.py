"""
Metrics Module
==============

This module contains functions for calculating performance metrics: 
- SNR (Signal-to-Noise Ratio)
- ENOB (Effective Number of Bits)
- FPGA-relevant metrics
"""

from .signal_to_noise_ratio import (
    compute_signal_to_noise_ratio_time_domain,
    compute_signal_to_noise_ratio_frequency_domain,
    compute_in_band_snr
)
from .effective_number_of_bits import (
    compute_effective_number_of_bits,
    compute_theoretical_enob_for_delta_sigma
)
from .fpga_metrics import FPGAMetricsCalculator

__all__ = [
    "compute_signal_to_noise_ratio_time_domain",
    "compute_signal_to_noise_ratio_frequency_domain",
    "compute_in_band_snr",
    "compute_effective_number_of_bits",
    "compute_theoretical_enob_for_delta_sigma",
    "FPGAMetricsCalculator"
]