"""
Spectrum-Based Metrics for Delta-Sigma Analysis
===============================================

This module provides reliable metrics for comparing delta-sigma configurations:
- In-band SNR and SNDR from PSD or FFT
- Signal, noise, and distortion power computation
- THD, SFDR (optional)
- ENOB derived from SNDR

These metrics are designed for fair comparison of configurations with different
input word lengths, OSR values, or modulator orders.
"""

import numpy as np
from typing import Tuple, Dict, Optional, Set


def locate_fundamental_bin(
    frequencies: np.ndarray,
    psd_or_spectrum: np.ndarray,
    fundamental_frequency_hz: float,
    search_window_bins: int = 5
) -> int:
    """
    Locate the bin containing the fundamental frequency.
    
    Searches near the expected bin and returns the one with maximum power.
    This handles small frequency mismatches due to non-coherent sampling.
    
    Args:
        frequencies: Frequency array
        psd_or_spectrum: PSD or power spectrum
        fundamental_frequency_hz: Expected fundamental frequency
        search_window_bins: Number of bins to search around expected location
    
    Returns:
        Index of the fundamental bin
    """
    freq_resolution = frequencies[1] - frequencies[0] if len(frequencies) > 1 else 1.0
    expected_bin = int(round(fundamental_frequency_hz / freq_resolution))
    
    # Search window
    start_bin = max(1, expected_bin - search_window_bins)
    end_bin = min(len(psd_or_spectrum) - 1, expected_bin + search_window_bins + 1)
    
    # Find peak in window
    window_slice = psd_or_spectrum[start_bin:end_bin]
    max_idx = np.argmax(window_slice)
    fundamental_bin = start_bin + max_idx
    
    return fundamental_bin


def get_signal_bins(
    fundamental_bin: int,
    num_bins_around: int = 2,
    spectrum_length: int = None
) -> Set[int]:
    """
    Get indices of bins containing the fundamental signal.
    
    Args:
        fundamental_bin: Index of fundamental frequency
        num_bins_around: Number of bins on each side to include (accounts for leakage)
        spectrum_length: Length of spectrum (for bounds checking)
    
    Returns:
        Set of bin indices
    """
    bins = set()
    for offset in range(-num_bins_around, num_bins_around + 1):
        bin_idx = fundamental_bin + offset
        if spectrum_length is None or (0 <= bin_idx < spectrum_length):
            bins.add(bin_idx)
    return bins


def get_harmonic_bins(
    fundamental_bin: int,
    num_harmonics: int = 6,
    num_bins_around: int = 2,
    spectrum_length: int = None
) -> Set[int]:
    """
    Get indices of bins containing harmonics.
    
    Args:
        fundamental_bin: Index of fundamental frequency
        num_harmonics: Number of harmonics to include (2nd, 3rd, ..., nth)
        num_bins_around: Number of bins around each harmonic to include
        spectrum_length: Length of spectrum (for bounds checking)
    
    Returns:
        Set of bin indices for all harmonics
    """
    bins = set()
    for harmonic in range(2, num_harmonics + 2):  # 2nd through (num_harmonics+1)th
        harmonic_bin = fundamental_bin * harmonic
        for offset in range(-num_bins_around, num_bins_around + 1):
            bin_idx = harmonic_bin + offset
            if spectrum_length is None or (0 <= bin_idx < spectrum_length):
                bins.add(bin_idx)
    return bins


def compute_snr_from_psd(
    frequencies: np.ndarray,
    psd_linear: np.ndarray,
    fundamental_frequency_hz: float,
    band_limit_hz: float,
    num_bins_around_fundamental: int = 2,
    num_harmonics_to_exclude: int = 0
) -> Tuple[float, Dict[str, float]]:
    """
    Compute SNR from Power Spectral Density.
    
    SNR excludes harmonic distortion from noise calculation.
    
    Args:
        frequencies: Frequency bins in Hz
        psd_linear: PSD in V²/Hz (linear units, NOT dB)
        fundamental_frequency_hz: Frequency of the signal
        band_limit_hz: Upper frequency limit for in-band analysis
        num_bins_around_fundamental: Bins to include around fundamental (±N)
        num_harmonics_to_exclude: Number of harmonics to exclude from noise
    
    Returns:
        snr_db: SNR in decibels
        powers: Dict with 'signal_power', 'noise_power', 'total_inband_power'
    """
    # Find in-band region
    band_mask = (frequencies > 0) & (frequencies <= band_limit_hz)
    
    if not np.any(band_mask):
        return 0.0, {'signal_power': 0.0, 'noise_power': 0.0, 'total_inband_power': 0.0}
    
    # Locate fundamental
    fundamental_bin = locate_fundamental_bin(frequencies, psd_linear, fundamental_frequency_hz)
    
    # Get signal bins
    signal_bins = get_signal_bins(fundamental_bin, num_bins_around_fundamental, len(psd_linear))
    
    # Get harmonic bins to exclude
    if num_harmonics_to_exclude > 0:
        harmonic_bins = get_harmonic_bins(fundamental_bin, num_harmonics_to_exclude, 
                                         num_bins_around_fundamental, len(psd_linear))
    else:
        harmonic_bins = set()
    
    # Compute powers by integrating PSD
    signal_power = 0.0
    noise_power = 0.0
    total_inband_power = 0.0
    
    for i in range(len(frequencies) - 1):
        if not band_mask[i]:
            continue
        
        # Trapezoidal integration
        df = frequencies[i + 1] - frequencies[i]
        psd_avg = (psd_linear[i] + psd_linear[i + 1]) / 2.0
        bin_power = psd_avg * df
        
        total_inband_power += bin_power
        
        if i in signal_bins:
            signal_power += bin_power
        elif i not in harmonic_bins and i != 0:  # Exclude DC
            noise_power += bin_power
    
    # Compute SNR
    if noise_power < 1e-30 or signal_power <= 0.0:
        snr_db = 200.0
    else:
        snr_db = 10.0 * np.log10(signal_power / noise_power)
    
    powers = {
        'signal_power': float(signal_power),
        'noise_power': float(noise_power),
        'total_inband_power': float(total_inband_power)
    }
    
    return snr_db, powers


def compute_sndr_from_psd(
    frequencies: np.ndarray,
    psd_linear: np.ndarray,
    fundamental_frequency_hz: float,
    band_limit_hz: float,
    num_bins_around_fundamental: int = 2,
    num_harmonics: int = 6
) -> Tuple[float, float, Dict[str, float]]:
    """
    Compute SNDR (Signal-to-Noise-and-Distortion Ratio) and ENOB from PSD.
    
    SNDR includes harmonic distortion in the denominator, providing a more
    realistic measure of usable dynamic range.
    
    Args:
        frequencies: Frequency bins in Hz
        psd_linear: PSD in V²/Hz (linear units)
        fundamental_frequency_hz: Frequency of the signal
        band_limit_hz: Upper frequency limit for in-band analysis
        num_bins_around_fundamental: Bins around fundamental and harmonics
        num_harmonics: Number of harmonics to include in distortion
    
    Returns:
        sndr_db: SNDR in decibels
        enob: Effective number of bits derived from SNDR
        powers: Dict with signal/noise/distortion powers
    """
    # Find in-band region
    band_mask = (frequencies > 0) & (frequencies <= band_limit_hz)
    
    if not np.any(band_mask):
        return 0.0, 0.0, {'signal_power': 0.0, 'noise_power': 0.0, 
                         'distortion_power': 0.0, 'total_inband_power': 0.0}
    
    # Locate fundamental
    fundamental_bin = locate_fundamental_bin(frequencies, psd_linear, fundamental_frequency_hz)
    
    # Get signal bins
    signal_bins = get_signal_bins(fundamental_bin, num_bins_around_fundamental, len(psd_linear))
    
    # Get harmonic bins
    harmonic_bins = get_harmonic_bins(fundamental_bin, num_harmonics, 
                                     num_bins_around_fundamental, len(psd_linear))
    
    # Compute powers
    signal_power = 0.0
    distortion_power = 0.0
    noise_power = 0.0
    total_inband_power = 0.0
    
    for i in range(len(frequencies) - 1):
        if not band_mask[i]:
            continue
        
        # Trapezoidal integration
        df = frequencies[i + 1] - frequencies[i]
        psd_avg = (psd_linear[i] + psd_linear[i + 1]) / 2.0
        bin_power = psd_avg * df
        
        total_inband_power += bin_power
        
        if i in signal_bins:
            signal_power += bin_power
        elif i in harmonic_bins:
            distortion_power += bin_power
        elif i != 0:  # Exclude DC
            noise_power += bin_power
    
    # Compute SNDR
    denominator = noise_power + distortion_power
    if denominator < 1e-30 or signal_power <= 0.0:
        sndr_db = 200.0
        enob = float('inf')
    else:
        sndr_db = 10.0 * np.log10(signal_power / denominator)
        # ENOB formula: ENOB = (SNDR - 1.76) / 6.02
        enob = (sndr_db - 1.76) / 6.02
    
    powers = {
        'signal_power': float(signal_power),
        'noise_power': float(noise_power),
        'distortion_power': float(distortion_power),
        'total_inband_power': float(total_inband_power)
    }
    
    return sndr_db, enob, powers


def compute_thd_from_psd(
    frequencies: np.ndarray,
    psd_linear: np.ndarray,
    fundamental_frequency_hz: float,
    band_limit_hz: float,
    num_harmonics: int = 6,
    num_bins_around: int = 2
) -> Tuple[float, Dict[str, float]]:
    """
    Compute Total Harmonic Distortion (THD) from PSD.
    
    THD = sqrt(sum of harmonic powers) / fundamental power
    
    Args:
        frequencies: Frequency bins
        psd_linear: PSD in V²/Hz
        fundamental_frequency_hz: Fundamental frequency
        band_limit_hz: Band limit
        num_harmonics: Number of harmonics to include
        num_bins_around: Bins around each component
    
    Returns:
        thd_db: THD in dB (negative value, e.g., -60 dB is good)
        powers: Dict with fundamental and harmonic powers
    """
    # Locate fundamental
    fundamental_bin = locate_fundamental_bin(frequencies, psd_linear, fundamental_frequency_hz)
    
    # Get signal bins
    signal_bins = get_signal_bins(fundamental_bin, num_bins_around, len(psd_linear))
    
    # Get harmonic bins
    harmonic_bins = get_harmonic_bins(fundamental_bin, num_harmonics, 
                                     num_bins_around, len(psd_linear))
    
    # Band mask
    band_mask = (frequencies > 0) & (frequencies <= band_limit_hz)
    
    # Compute powers
    fundamental_power = 0.0
    harmonic_power = 0.0
    
    for i in range(len(frequencies) - 1):
        if not band_mask[i]:
            continue
        
        df = frequencies[i + 1] - frequencies[i]
        psd_avg = (psd_linear[i] + psd_linear[i + 1]) / 2.0
        bin_power = psd_avg * df
        
        if i in signal_bins:
            fundamental_power += bin_power
        elif i in harmonic_bins:
            harmonic_power += bin_power
    
    # Compute THD
    if fundamental_power < 1e-30:
        thd_db = -200.0
    else:
        thd_ratio = np.sqrt(harmonic_power) / np.sqrt(fundamental_power)
        thd_db = 20.0 * np.log10(thd_ratio + 1e-20)
    
    powers = {
        'fundamental_power': float(fundamental_power),
        'harmonic_power': float(harmonic_power)
    }
    
    return thd_db, powers


def compute_sfdr_from_psd(
    frequencies: np.ndarray,
    psd_linear: np.ndarray,
    fundamental_frequency_hz: float,
    band_limit_hz: float,
    num_bins_around_fundamental: int = 2
) -> Tuple[float, float]:
    """
    Compute Spurious-Free Dynamic Range (SFDR) from PSD.
    
    SFDR is the ratio of the fundamental to the largest spurious tone.
    
    Args:
        frequencies: Frequency bins
        psd_linear: PSD in V²/Hz
        fundamental_frequency_hz: Fundamental frequency
        band_limit_hz: Band limit
        num_bins_around_fundamental: Bins to exclude around fundamental
    
    Returns:
        sfdr_db: SFDR in dB
        spur_frequency_hz: Frequency of the largest spur
    """
    # Locate fundamental
    fundamental_bin = locate_fundamental_bin(frequencies, psd_linear, fundamental_frequency_hz)
    
    # Get signal bins to exclude
    signal_bins = get_signal_bins(fundamental_bin, num_bins_around_fundamental, len(psd_linear))
    
    # Band mask
    band_mask = (frequencies > 0) & (frequencies <= band_limit_hz)
    
    # Find fundamental power
    fundamental_power = np.sum([psd_linear[i] for i in signal_bins if i < len(psd_linear)])
    
    # Find largest spur excluding fundamental and DC
    max_spur_power = 0.0
    max_spur_idx = 0
    
    for i in range(len(psd_linear)):
        if band_mask[i] and i not in signal_bins and i != 0:
            if psd_linear[i] > max_spur_power:
                max_spur_power = psd_linear[i]
                max_spur_idx = i
    
    # Compute SFDR
    if max_spur_power < 1e-30 or fundamental_power <= 0.0:
        sfdr_db = 200.0
    else:
        sfdr_db = 10.0 * np.log10(fundamental_power / max_spur_power)
    
    spur_frequency_hz = frequencies[max_spur_idx] if max_spur_idx < len(frequencies) else 0.0
    
    return sfdr_db, float(spur_frequency_hz)


def compute_all_metrics_from_psd(
    frequencies: np.ndarray,
    psd_linear: np.ndarray,
    fundamental_frequency_hz: float,
    band_limit_hz: float,
    num_harmonics: int = 6
) -> Dict[str, float]:
    """
    Compute all common metrics from PSD in one call.
    
    Args:
        frequencies: Frequency bins in Hz
        psd_linear: PSD in V²/Hz (linear units)
        fundamental_frequency_hz: Signal frequency
        band_limit_hz: In-band frequency limit
        num_harmonics: Number of harmonics for distortion analysis
    
    Returns:
        Dict containing SNR, SNDR, ENOB, THD, SFDR and power components
    """
    # Compute SNDR and ENOB
    sndr_db, enob, sndr_powers = compute_sndr_from_psd(
        frequencies, psd_linear, fundamental_frequency_hz, 
        band_limit_hz, num_harmonics=num_harmonics
    )
    
    # Compute SNR
    snr_db, snr_powers = compute_snr_from_psd(
        frequencies, psd_linear, fundamental_frequency_hz,
        band_limit_hz, num_harmonics_to_exclude=num_harmonics
    )
    
    # Compute THD
    thd_db, thd_powers = compute_thd_from_psd(
        frequencies, psd_linear, fundamental_frequency_hz,
        band_limit_hz, num_harmonics=num_harmonics
    )
    
    # Compute SFDR
    sfdr_db, spur_freq = compute_sfdr_from_psd(
        frequencies, psd_linear, fundamental_frequency_hz, band_limit_hz
    )
    
    # Compile all metrics
    metrics = {
        'snr_db': snr_db,
        'sndr_db': sndr_db,
        'enob': enob,
        'thd_db': thd_db,
        'sfdr_db': sfdr_db,
        'spur_frequency_hz': spur_freq,
        'signal_power': sndr_powers['signal_power'],
        'noise_power': sndr_powers['noise_power'],
        'distortion_power': sndr_powers['distortion_power'],
        'total_inband_power': sndr_powers['total_inband_power']
    }
    
    return metrics
