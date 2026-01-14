"""
Signal-to-Noise Ratio (SNR) Computation
=======================================

SNR measures the ratio of signal power to noise power, expressed in decibels (dB).

IMPORTANT: This module handles phase delay compensation for causal filters.
"""

import numpy as np
from typing import Tuple


def find_signal_delay_subsample(
    reference_signal: np.ndarray,
    delayed_signal: np.ndarray,
    signal_frequency_hz: float,
    sampling_frequency_hz: float
) -> int:
    """
    Find the delay between two signals, constrained to less than one period.
    
    This function uses cross-correlation but limits the search to less than
    one signal period to avoid matching to the wrong cycle of a periodic signal.
    
    Args:
        reference_signal: The original (non-delayed) signal.
        delayed_signal: The signal that may be delayed.
        signal_frequency_hz:  Frequency of the signal (to calculate period).
        sampling_frequency_hz: Sampling rate. 
    
    Returns:
        int: The delay in samples (positive = delayed_signal is behind).
    """
    # Calculate one period in samples
    samples_per_period:  int = int(sampling_frequency_hz / signal_frequency_hz)
    
    # Limit search to 80% of one period to avoid ambiguity
    max_delay:  int = int(0.8 * samples_per_period)
    
    # Also limit to 10% of signal length as a safety
    max_delay = min(max_delay, len(reference_signal) // 10)
    
    # Ensure minimum search range
    max_delay = max(max_delay, 100)
    
    # Normalize signals
    ref_normalized = reference_signal - np.mean(reference_signal)
    del_normalized = delayed_signal - np.mean(delayed_signal)
    
    ref_std = np.std(ref_normalized)
    del_std = np.std(del_normalized)
    
    if ref_std < 1e-10 or del_std < 1e-10:
        return 0
    
    ref_normalized = ref_normalized / ref_std
    del_normalized = del_normalized / del_std
    
    # Find best correlation within allowed delay range
    best_delay:  int = 0
    best_correlation: float = -np.inf
    
    for delay in range(0, max_delay):
        if delay == 0:
            correlation = np.mean(ref_normalized * del_normalized)
        else:
            ref_segment = ref_normalized[delay:]
            del_segment = del_normalized[:-delay]
            
            min_len = min(len(ref_segment), len(del_segment))
            if min_len < 100:
                break
            
            correlation = np.mean(ref_segment[: min_len] * del_segment[:min_len])
        
        if correlation > best_correlation: 
            best_correlation = correlation
            best_delay = delay
    
    return best_delay


def estimate_filter_group_delay(
    number_of_stages: int,
    cutoff_frequency_hz:  float,
    sampling_frequency_hz: float,
    signal_frequency_hz: float
) -> int:
    """
    Estimate the group delay of a cascaded RC filter at the signal frequency.
    
    For a first-order RC filter, the group delay is: 
        τ_g(f) = τ / (1 + (f/f_c)²)
    
    where τ = 1/(2πf_c) is the time constant. 
    
    For N cascaded stages, the total group delay is approximately N × τ_g.
    
    Args:
        number_of_stages:  Number of RC stages.
        cutoff_frequency_hz: Cutoff frequency of each stage.
        sampling_frequency_hz: Sampling rate.
        signal_frequency_hz:  Frequency at which to calculate delay.
    
    Returns:
        int:  Estimated delay in samples.
    """
    # Time constant
    tau:  float = 1.0 / (2.0 * np.pi * cutoff_frequency_hz)
    
    # Frequency ratio
    freq_ratio: float = signal_frequency_hz / cutoff_frequency_hz
    
    # Group delay per stage
    group_delay_per_stage: float = tau / (1.0 + freq_ratio ** 2)
    
    # Total delay for N stages
    total_delay_seconds: float = number_of_stages * group_delay_per_stage
    
    # Convert to samples
    delay_samples: int = int(round(total_delay_seconds * sampling_frequency_hz))
    
    return delay_samples


def compute_signal_to_noise_ratio_time_domain(
    reconstructed_signal: np.ndarray,
    reference_signal: np.ndarray,
    signal_frequency_hz: float = None,
    sampling_frequency_hz: float = None,
    compensate_delay: bool = True,
    normalize_amplitude: bool = True
) -> float:
    """
    Compute SNR by comparing reconstructed signal to reference. 
    
    Args:
        reconstructed_signal: The output after reconstruction filter.
        reference_signal: The original input signal (ground truth).
        signal_frequency_hz: Signal frequency (needed for delay compensation).
        sampling_frequency_hz: Sampling rate (needed for delay compensation).
        compensate_delay: If True, find and compensate for phase delay.
        normalize_amplitude: If True, scale reconstructed to match reference.
    
    Returns:
        float: SNR in decibels (dB).
    """
    # ===== STEP 1: FIND AND COMPENSATE FOR DELAY =====
    if compensate_delay and signal_frequency_hz is not None and sampling_frequency_hz is not None:
        delay_samples:  int = find_signal_delay_subsample(
            reference_signal=reference_signal,
            delayed_signal=reconstructed_signal,
            signal_frequency_hz=signal_frequency_hz,
            sampling_frequency_hz=sampling_frequency_hz
        )
        
        if delay_samples > 0:
            reference_aligned = reference_signal[delay_samples:]
            reconstructed_aligned = reconstructed_signal[:-delay_samples]
        else:
            reference_aligned = reference_signal
            reconstructed_aligned = reconstructed_signal
    else:
        reference_aligned = reference_signal
        reconstructed_aligned = reconstructed_signal
    
    # Ensure same length
    min_length:  int = min(len(reference_aligned), len(reconstructed_aligned))
    reference_aligned = reference_aligned[:min_length]
    reconstructed_aligned = reconstructed_aligned[:min_length]
    
    # ===== STEP 2: NORMALIZE AMPLITUDE =====
    if normalize_amplitude:
        dot_product: float = np.dot(reference_aligned, reconstructed_aligned)
        reconstructed_power: float = np.dot(reconstructed_aligned, reconstructed_aligned)
        
        if reconstructed_power > 1e-20:
            scale_factor: float = dot_product / reconstructed_power
            reconstructed_scaled = reconstructed_aligned * scale_factor
        else:
            reconstructed_scaled = reconstructed_aligned
    else:
        reconstructed_scaled = reconstructed_aligned
    
    # ===== STEP 3: CALCULATE SNR =====
    error_signal = reference_aligned - reconstructed_scaled
    signal_power:  float = float(np.mean(reference_aligned ** 2))
    noise_power: float = float(np.mean(error_signal ** 2))
    
    if noise_power < 1e-20:
        return 200.0
    
    snr_db: float = 10.0 * np.log10(signal_power / noise_power)
    
    return snr_db


def compute_snr_with_diagnostics(
    reconstructed_signal: np.ndarray,
    reference_signal: np.ndarray,
    signal_frequency_hz: float = None,
    sampling_frequency_hz:  float = None,
    verbose: bool = True
) -> Tuple[float, dict]:
    """
    Compute SNR with detailed diagnostics.
    """
    # Use subsample-aware delay finding if signal info provided
    if signal_frequency_hz is not None and sampling_frequency_hz is not None:
        delay_samples = find_signal_delay_subsample(
            reference_signal, reconstructed_signal,
            signal_frequency_hz, sampling_frequency_hz
        )
    else:
        # Fallback to simple correlation (may find wrong peak)
        delay_samples = 0
    
    # Align signals
    if delay_samples > 0:
        ref_aligned = reference_signal[delay_samples:]
        rec_aligned = reconstructed_signal[:-delay_samples]
    else: 
        ref_aligned = reference_signal
        rec_aligned = reconstructed_signal
    
    min_len = min(len(ref_aligned), len(rec_aligned))
    ref_aligned = ref_aligned[: min_len]
    rec_aligned = rec_aligned[:min_len]
    
    # Calculate scale factor
    dot_product = np.dot(ref_aligned, rec_aligned)
    rec_power = np.dot(rec_aligned, rec_aligned)
    scale_factor = dot_product / rec_power if rec_power > 1e-20 else 1.0
    
    rec_scaled = rec_aligned * scale_factor
    
    # Calculate SNR
    error = ref_aligned - rec_scaled
    signal_power = np.mean(ref_aligned ** 2)
    noise_power = np.mean(error ** 2)
    snr_db = 10.0 * np.log10(signal_power / noise_power) if noise_power > 1e-20 else 200.0
    
    # Correlation
    correlation = np.corrcoef(ref_aligned, rec_scaled)[0, 1] if len(ref_aligned) > 1 else 0.0
    
    diagnostics = {
        'delay_samples': delay_samples,
        'scale_factor': scale_factor,
        'signal_power': signal_power,
        'noise_power': noise_power,
        'ref_std': np.std(ref_aligned),
        'rec_std_before_scale': np.std(rec_aligned),
        'rec_std_after_scale': np.std(rec_scaled),
        'correlation': correlation
    }
    
    if verbose:
        print(f"\n--- SNR Diagnostics ---")
        print(f"  Detected Delay:       {delay_samples} samples")
        print(f"  Scale Factor:        {scale_factor:.4f}")
        print(f"  Correlation:         {correlation:.4f}")
        print(f"  Reference Std:       {diagnostics['ref_std']:.4f}")
        print(f"  Reconstructed Std:    {diagnostics['rec_std_after_scale']:.4f}")
        print(f"  SNR:                 {snr_db:.1f} dB")
    
    return snr_db, diagnostics


def compute_signal_to_noise_ratio_frequency_domain(
    signal: np.ndarray,
    sampling_frequency_hz: float,
    signal_frequency_hz: float,
    number_of_harmonics_to_exclude: int = 5
) -> float:
    """
    Compute SNR in frequency domain (phase-independent).
    """
    number_of_samples: int = len(signal)
    
    window = np.hanning(number_of_samples)
    windowed_signal = signal * window
    
    spectrum = np.fft.fft(windowed_signal)
    power_spectrum = np.abs(spectrum) ** 2
    
    positive_spectrum = power_spectrum[: number_of_samples // 2]
    
    frequency_resolution_hz = sampling_frequency_hz / number_of_samples
    
    signal_bins: set = set()
    for harmonic in range(1, number_of_harmonics_to_exclude + 1):
        harmonic_frequency = harmonic * signal_frequency_hz
        bin_index = int(round(harmonic_frequency / frequency_resolution_hz))
        if bin_index >= len(positive_spectrum):
            continue
        for offset in range(-2, 3):
            adjacent_bin = bin_index + offset
            if 0 <= adjacent_bin < len(positive_spectrum):
                signal_bins.add(adjacent_bin)
    
    signal_power = sum(positive_spectrum[b] for b in signal_bins)
    
    noise_power = 0.0
    for bin_index in range(1, len(positive_spectrum)):
        if bin_index not in signal_bins:
            noise_power += positive_spectrum[bin_index]
    
    if noise_power < 1e-20:
        return 200.0
    
    return 10.0 * np.log10(signal_power / noise_power)


def compute_sinad_in_band(
    signal: np.ndarray,
    sampling_frequency_hz: float,
    signal_frequency_hz: float,
    signal_bandwidth_hz: float,
    num_harmonics: int = 6
) -> tuple[float, float]:
    """
    Compute SINAD (signal-to-noise-and-distortion) restricted to a reconstruction band
    and return (SINAD_dB, ENOB_bits).

    The function:
    - Applies a Hann window and computes the FFT power spectrum.
    - Identifies the fundamental (±2 bins) as the signal.
    - Identifies harmonics (2..num_harmonics) within the reconstruction band as distortion.
    - Treats remaining in-band bins (excluding DC) as noise.
    - Returns SINAD in dB and ENOB computed from SINAD using ENOB = (SINAD - 1.76) / 6.02.

    This is the recommended metric for ENOB in delta-sigma systems because it
    measures usable resolution inside the reconstruction band and accounts for
    harmonic distortion.
    """
    N = len(signal)
    if N < 4:
        return 0.0, 0.0

    window = np.hanning(N)
    windowed = signal * window
    spectrum = np.fft.fft(windowed)
    power_spectrum = np.abs(spectrum) ** 2

    half = N // 2
    freq_axis = np.fft.fftfreq(N, d=1.0 / sampling_frequency_hz)[:half]

    # Frequency resolution and fundamental bin
    freq_res = sampling_frequency_hz / N
    fund_bin = int(round(signal_frequency_hz / freq_res))

    # Collect bins for fundamental (±2 bins) if inside band
    signal_bins = set()
    for offset in range(-2, 3):
        b = fund_bin + offset
        if 0 <= b < half and freq_axis[b] <= signal_bandwidth_hz:
            signal_bins.add(b)

    # Collect harmonic bins (2..num_harmonics)
    harmonic_bins = set()
    for h in range(2, num_harmonics + 1):
        hb = fund_bin * h
        for offset in range(-2, 3):
            b = hb + offset
            if 0 <= b < half and freq_axis[b] <= signal_bandwidth_hz:
                harmonic_bins.add(b)

    # Compute powers
    signal_power = float(np.sum([power_spectrum[b] for b in signal_bins])) if signal_bins else 0.0
    distortion_power = float(np.sum([power_spectrum[b] for b in harmonic_bins])) if harmonic_bins else 0.0

    # Noise = in-band bins excluding DC (bin 0), signal bins and harmonic bins
    noise_power = 0.0
    for b in range(1, half):
        if freq_axis[b] <= signal_bandwidth_hz and b not in signal_bins and b not in harmonic_bins:
            noise_power += float(power_spectrum[b])

    denom = noise_power + distortion_power
    if denom < 1e-20 or signal_power <= 0.0:
        sinad_db = 200.0
        enob = float('inf')
    else:
        sinad_db = 10.0 * np.log10(signal_power / denom)
        enob = (sinad_db - 1.76) / 6.02

    return sinad_db, enob


def compute_in_band_snr(
    signal: np.ndarray,
    sampling_frequency_hz: float,
    signal_frequency_hz: float,
    signal_bandwidth_hz: float
) -> float:
    """
    Compute SNR considering only in-band noise (phase-independent).
    """
    number_of_samples: int = len(signal)
    
    window = np.hanning(number_of_samples)
    windowed_signal = signal * window
    
    spectrum = np.fft.fft(windowed_signal)
    power_spectrum = np.abs(spectrum) ** 2
    
    frequency_axis = np.fft.fftfreq(number_of_samples, d=1.0 / sampling_frequency_hz)
    
    in_band_mask = (frequency_axis >= 0) & (frequency_axis <= signal_bandwidth_hz)
    
    frequency_resolution_hz = sampling_frequency_hz / number_of_samples
    signal_bin = int(round(signal_frequency_hz / frequency_resolution_hz))
    
    signal_bins:  set = set()
    for offset in range(-3, 4):
        bin_idx = signal_bin + offset
        if 0 <= bin_idx < number_of_samples:
            signal_bins.add(bin_idx)
    
    signal_power = 0.0
    for bin_idx in signal_bins:
        if in_band_mask[bin_idx]:
            signal_power += power_spectrum[bin_idx]
    
    noise_power = 0.0
    for bin_idx in range(number_of_samples):
        if in_band_mask[bin_idx] and bin_idx not in signal_bins and bin_idx != 0:
            noise_power += power_spectrum[bin_idx]
    
    if noise_power < 1e-20:
        return 200.0
    
    return 10.0 * np.log10(signal_power / noise_power)