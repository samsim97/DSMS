"""
Effective Number of Bits (ENOB) Computation
===========================================

ENOB is a measure of the dynamic performance of an ADC or DAC. 
It represents the number of bits of an ideal converter that would
have the same SNR as the device under test.

The relationship between SNR and ENOB is: 

    SNR (dB) = 6.02 * ENOB + 1.76

Therefore: 

    ENOB = (SNR - 1.76) / 6.02

Where:
- 6.02 dB comes from the fact that each bit doubles the number of
  quantization levels, improving SNR by 20*log10(2) ≈ 6.02 dB
- 1.76 dB is a correction factor for quantization noise in an ideal
  converter (derived from the quantization noise power of a full-scale
  sine wave)

For delta-sigma modulators, the theoretical ENOB can be much higher
than the number of bits in the quantizer due to noise shaping and
oversampling. 

THEORETICAL ENOB FOR DELTA-SIGMA MODULATORS: 
============================================

For an N-th order modulator with oversampling ratio OSR:

    SNR ≈ 6.02 + 1.76 + 10*log10(2N+1) + (2N+1)*10*log10(OSR) - 10*log10(π^(2N)/(2N+1))

Simplified approximation:
    
    ENOB ≈ (N + 0.5) * log2(OSR) - log2(π^N / sqrt(2N+1))

Example (2nd order, OSR=256):
    ENOB ≈ 25 * 8 - 2.7 ≈ 17.3 bits

This is why delta-sigma modulators can achieve high resolution
with just a 1-bit quantizer! 
"""

import numpy as np


def compute_effective_number_of_bits(signal_to_noise_ratio_db: float) -> float:
    """
    Compute ENOB from measured SNR.

    Formula:  ENOB = (SNR - 1.76) / 6.02

    Args:
        signal_to_noise_ratio_db: The measured SNR in decibels.

    Returns:
        float:  Effective number of bits. 

    Example:
        SNR = 98 dB → ENOB = (98 - 1.76) / 6.02 ≈ 16 bits
    """
    # The standard formula relating SNR to ENOB
    # Derived from:  SNR = 6.02*N + 1.76 for an ideal N-bit converter
    enob: float = (signal_to_noise_ratio_db - 1.76) / 6.02

    return enob


def compute_theoretical_enob_for_delta_sigma(
    modulator_order: int,
    oversampling_ratio: int,
    quantizer_bits: int = 1
) -> float:
    """
    Compute theoretical ENOB for an ideal delta-sigma modulator.

    This gives the best-case ENOB assuming: 
    - Ideal components (no thermal noise, mismatch, etc.)
    - Stable operation
    - Perfect reconstruction filter

    The formula is based on the noise transfer function NTF = (1 - z^-1)^N
    and accounts for oversampling and noise shaping.

    For an N-th order modulator with L-bit quantizer and OSR: 

    In-band quantization noise power:
        σ²_inband = (Δ²/12) * (π^(2N) / (2N+1)) * (1/OSR)^(2N+1)

    Where Δ is the quantizer step size.

    SNR = 10*log10(signal_power / noise_power)

    For a full-scale sine wave with 1-bit quantizer:
        SNR ≈ -7.2 + 30*log10(OSR) dB  (for 1st order)
        SNR ≈ -12.9 + 50*log10(OSR) dB (for 2nd order)

    General formula:
        SNR ≈ 6.02*L + 1.76 - 5.17 + (2N+1)*10*log10(OSR) - 10*log10(π^(2N)/(2N+1))

    Args:
        modulator_order:  Order of the modulator (N).
        oversampling_ratio: The OSR value.
        quantizer_bits: Number of bits in quantizer (L). Default 1.

    Returns:
        float: Theoretical ENOB in bits. 

    References:
        - Schreier & Temes, "Understanding Delta-Sigma Data Converters"
        - Analog Devices MT-022 Tutorial
    """
    # Validate inputs
    if modulator_order < 1:
        raise ValueError("Modulator order must be at least 1")
    if oversampling_ratio < 1:
        raise ValueError("Oversampling ratio must be at least 1")
    if quantizer_bits < 1:
        raise ValueError("Quantizer must have at least 1 bit")

    # Shorthand variables for readability
    N:  int = modulator_order
    OSR: int = oversampling_ratio
    L: int = quantizer_bits

    # ===== THEORETICAL SNR CALCULATION =====
    # 
    # The in-band noise power for an Nth-order modulator with NTF = (1-z^-1)^N is:
    #
    # P_noise_inband = (Δ²/12) * (π^(2N) / (2N+1)) * (1/OSR)^(2N+1)
    #
    # For a 1-bit quantizer with output ±1, Δ = 2, so Δ²/12 = 4/12 = 1/3
    # For an L-bit quantizer, Δ = 2/(2^L - 1)
    #
    # Signal power for full-scale sine:  P_signal = 0.5 (amplitude = 1, RMS = 1/√2)
    # But we use reduced amplitude for stability, so actual SNR will be lower. 

    # Quantizer step size for [-1, 1] range
    number_of_levels: int = 2 ** L
    quantizer_step_delta: float = 2.0 / (number_of_levels - 1) if number_of_levels > 1 else 2.0

    # Quantization noise power (for uniform quantizer)
    quantization_noise_power: float = (quantizer_step_delta ** 2) / 12.0

    # Noise shaping factor:  π^(2N) / (2N + 1)
    noise_shaping_numerator: float = np.pi ** (2 * N)
    noise_shaping_denominator: float = 2 * N + 1
    noise_shaping_factor:  float = noise_shaping_numerator / noise_shaping_denominator

    # Oversampling benefit: (1/OSR)^(2N+1)
    oversampling_factor: float = (1.0 / OSR) ** (2 * N + 1)

    # In-band noise power
    in_band_noise_power: float = (
        quantization_noise_power * noise_shaping_factor * oversampling_factor
    )

    # Signal power (assuming amplitude = 0.5 for stability margin)
    # For a sine wave: P = A²/2
    assumed_amplitude: float = 0.5
    signal_power: float = (assumed_amplitude ** 2) / 2.0

    # SNR calculation
    if in_band_noise_power < 1e-30:
        snr_db: float = 200.0  # Effectively infinite
    else:
        snr_db = 10.0 * np.log10(signal_power / in_band_noise_power)

    # Convert SNR to ENOB
    theoretical_enob: float = compute_effective_number_of_bits(snr_db)

    return theoretical_enob


def compute_snr_from_enob(enob: float) -> float:
    """
    Compute SNR from ENOB (inverse of compute_effective_number_of_bits).

    Formula: SNR = 6.02 * ENOB + 1.76

    Args:
        enob: Effective number of bits. 

    Returns:
        float:  SNR in decibels.
    """
    snr_db: float = 6.02 * enob + 1.76
    return snr_db


def print_enob_table(
    max_order: int = 5,
    osr_values: list | None = None
) -> None:
    """
    Print a table of theoretical ENOB values for different configurations.

    Useful for planning your FPGA implementation. 

    Args:
        max_order: Maximum modulator order to show.
        osr_values: List of OSR values to include. 
    """
    if osr_values is None:
        osr_values = [32, 64, 128, 256, 512, 1024, 2048]

    print("\n" + "=" * 80)
    print("THEORETICAL ENOB (bits) FOR 1-BIT DELTA-SIGMA MODULATOR")
    print("=" * 80)
    print(f"{'Order':<8}", end="")
    for osr in osr_values: 
        print(f"OSR={osr:<6}", end="")
    print()
    print("-" * 80)

    for order in range(1, max_order + 1):
        print(f"{order:<8}", end="")
        for osr in osr_values:
            enob = compute_theoretical_enob_for_delta_sigma(order, osr, 1)
            print(f"{enob:<10.1f}", end="")
        print()

    print("=" * 80)
    print("Note:  Actual ENOB will be lower due to non-idealities and stability limits.")
    print("      Higher orders (3+) require reduced input amplitude for stability.")
    print()