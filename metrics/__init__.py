"""
Metrics Module
==============

This module contains functions for calculating performance metrics: 
- SNR (Signal-to-Noise Ratio)
- ENOB (Effective Number of Bits)
- FPGA-relevant metrics
- PSD computation and spectrum-based metrics
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

# PSD and spectrum-based metrics
from .psd_utils import (
    compute_welch_psd,
    psd_to_db,
    compute_psd_with_options,
    compute_band_power_from_psd
)
from .spectrum_metrics import (
    compute_snr_from_psd,
    compute_sndr_from_psd,
    compute_thd_from_psd,
    compute_sfdr_from_psd,
    compute_all_metrics_from_psd
)

__all__ = [
    "compute_signal_to_noise_ratio_time_domain",
    "compute_signal_to_noise_ratio_frequency_domain",
    "compute_in_band_snr",
    "compute_effective_number_of_bits",
    "compute_theoretical_enob_for_delta_sigma",
    "FPGAMetricsCalculator",
    # PSD utilities
    "compute_welch_psd",
    "psd_to_db",
    "compute_psd_with_options",
    "compute_band_power_from_psd",
    # Spectrum metrics
    "compute_snr_from_psd",
    "compute_sndr_from_psd",
    "compute_thd_from_psd",
    "compute_sfdr_from_psd",
    "compute_all_metrics_from_psd"
]