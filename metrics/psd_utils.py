"""
Power Spectral Density (PSD) Utilities
=======================================

This module provides correct PSD computation using Welch's method
with proper normalization for delta-sigma modulator analysis.

The key difference from naive FFT-based spectrum plots:
- Welch's method averages multiple overlapping segments, reducing variance
- scaling='density' provides true power per Hz (V²/Hz)
- Proper one-sided PSD handling for real signals
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any

try:
    from scipy import signal as scipy_signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def compute_welch_psd(
    signal_data: np.ndarray,
    sampling_frequency_hz: float,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    window: str = 'hann',
    detrend: str = 'constant',
    remove_dc: bool = True,
    return_onesided: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Power Spectral Density using Welch's method.
    
    This provides a true PSD estimate in V²/Hz (linear units) by averaging
    periodograms of overlapping segments. This is the correct way to analyze
    noise and compare configurations.
    
    Args:
        signal_data: Input signal array
        sampling_frequency_hz: Sampling rate in Hz
        nperseg: Length of each segment for FFT. If None, uses len(signal)//8
        noverlap: Number of points to overlap between segments. If None, uses nperseg//2
        window: Window function name ('hann', 'hamming', 'blackman', etc.)
        detrend: Detrend type ('constant' removes mean, 'linear' removes trend, False for none)
        remove_dc: If True, remove mean before analysis (recommended)
        return_onesided: Return one-sided PSD for real signals (True recommended)
    
    Returns:
        frequencies: Frequency bins in Hz
        psd: Power spectral density in V²/Hz (linear units)
        
    Raises:
        ImportError: If scipy is not available
    """
    if not SCIPY_AVAILABLE:
        raise ImportError(
            "scipy is required for Welch PSD computation. "
            "Install it with: pip install scipy>=1.7.0"
        )
    
    # Remove DC if requested
    if remove_dc:
        signal_data = signal_data - np.mean(signal_data)
    
    # Default segment length: ~1/8 of signal for good frequency resolution
    if nperseg is None:
        nperseg = max(256, len(signal_data) // 8)
    
    # Default overlap: 50%
    if noverlap is None:
        noverlap = nperseg // 2
    
    # Compute Welch PSD
    frequencies, psd = scipy_signal.welch(
        signal_data,
        fs=sampling_frequency_hz,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend,
        return_onesided=return_onesided,
        scaling='density',  # This is the key: power per Hz
        average='mean'
    )
    
    return frequencies, psd


def psd_to_db(psd_linear: np.ndarray, ref: Optional[float] = None) -> np.ndarray:
    """
    Convert PSD from linear units (V²/Hz) to dB scale.
    
    Args:
        psd_linear: PSD in V²/Hz
        ref: Reference value for dBFS conversion. If None, uses dB/Hz (10*log10(PSD))
             If provided, uses dBFS/Hz (10*log10(PSD/ref²))
    
    Returns:
        PSD in dB/Hz or dBFS/Hz
    """
    if ref is not None:
        # dBFS/Hz relative to full-scale reference
        psd_db = 10.0 * np.log10(psd_linear / (ref ** 2) + 1e-20)
    else:
        # dB/Hz (absolute)
        psd_db = 10.0 * np.log10(psd_linear + 1e-20)
    
    return psd_db


def compute_psd_with_options(
    signal_data: np.ndarray,
    sampling_frequency_hz: float,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    window: str = 'hann',
    detrend: str = 'constant',
    remove_dc: bool = True,
    return_db: bool = True,
    db_reference: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Compute PSD with flexible output options and metadata.
    
    Args:
        signal_data: Input signal
        sampling_frequency_hz: Sampling rate
        nperseg: Segment length for Welch's method
        noverlap: Overlap length
        window: Window function
        detrend: Detrending method
        remove_dc: Whether to remove DC component
        return_db: If True, return PSD in dB; if False, return linear V²/Hz
        db_reference: Full-scale reference for dBFS conversion (only used if return_db=True)
    
    Returns:
        frequencies: Frequency bins in Hz
        psd: PSD in requested units
        metadata: Dict with computation parameters
    """
    frequencies, psd_linear = compute_welch_psd(
        signal_data=signal_data,
        sampling_frequency_hz=sampling_frequency_hz,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
        detrend=detrend,
        remove_dc=remove_dc,
        return_onesided=True
    )
    
    if return_db:
        psd_output = psd_to_db(psd_linear, ref=db_reference)
        units = 'dBFS/Hz' if db_reference is not None else 'dB/Hz'
    else:
        psd_output = psd_linear
        units = 'V²/Hz'
    
    # Collect metadata
    metadata = {
        'nperseg': nperseg if nperseg is not None else len(signal_data) // 8,
        'noverlap': noverlap if noverlap is not None else (nperseg if nperseg is not None else len(signal_data) // 8) // 2,
        'window': window,
        'detrend': detrend,
        'remove_dc': remove_dc,
        'units': units,
        'frequency_resolution_hz': frequencies[1] - frequencies[0] if len(frequencies) > 1 else 0,
        'db_reference': db_reference
    }
    
    return frequencies, psd_output, metadata


def compute_band_power_from_psd(
    frequencies: np.ndarray,
    psd_linear: np.ndarray,
    freq_min: float = 0.0,
    freq_max: Optional[float] = None
) -> float:
    """
    Compute total power in a frequency band from PSD.
    
    Integrates PSD over specified frequency range using trapezoidal rule.
    
    Args:
        frequencies: Frequency bins in Hz
        psd_linear: PSD in V²/Hz (linear units, not dB)
        freq_min: Lower frequency bound
        freq_max: Upper frequency bound (if None, uses max frequency)
    
    Returns:
        Total power in the band (V²)
    """
    if freq_max is None:
        freq_max = frequencies[-1]
    
    # Find indices for the band
    band_mask = (frequencies >= freq_min) & (frequencies <= freq_max)
    
    if not np.any(band_mask):
        return 0.0
    
    # Integrate using trapezoidal rule
    band_freqs = frequencies[band_mask]
    band_psd = psd_linear[band_mask]
    
    # Power = integral of PSD over frequency
    power = np.trapz(band_psd, band_freqs)
    
    return float(power)
