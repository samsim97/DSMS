"""
Ideal Low-Pass Filter
=====================

This module implements an ideal "brick-wall" low-pass filter using FFT.

An ideal filter has: 
- Unity gain (0 dB) below cutoff frequency
- Zero gain below cutoff frequency
- Infinitely sharp transition

This is NOT physically realizable but useful for: 
1. Theoretical performance analysis
2. Best-case SNR calculations
3. Comparison with practical filters
"""

import numpy as np


class IdealLowPassFilter: 
    """
    Ideal brick-wall low-pass filter using FFT.

    This filter passes all frequencies below cutoff with unity gain
    and completely blocks all frequencies above cutoff.

    WARNING: This filter is NOT causal and introduces artifacts at
    signal boundaries. It's for analysis only, not for real-time use.

    Attributes:
        cutoff_frequency_hz (float): The cutoff frequency. 
        sampling_frequency_hz (float): The sampling rate.
    """

    def __init__(
        self,
        cutoff_frequency_hz: float,
        sampling_frequency_hz: float
    ) -> None:
        """
        Initialize the ideal low-pass filter.

        Args:
            cutoff_frequency_hz:  Frequencies above this are blocked.
            sampling_frequency_hz: The sampling rate. 
        """
        if cutoff_frequency_hz <= 0:
            raise ValueError("Cutoff frequency must be positive.")

        if cutoff_frequency_hz >= sampling_frequency_hz / 2:
            raise ValueError("Cutoff must be below Nyquist frequency.")

        self.cutoff_frequency_hz: float = cutoff_frequency_hz
        self.sampling_frequency_hz: float = sampling_frequency_hz

    def filter_signal(self, input_signal: np.ndarray) -> np.ndarray:
        """
        Apply ideal low-pass filter using FFT.

        Process: 
        1. Compute FFT of input signal
        2. Zero out all frequency components above cutoff
        3. Compute inverse FFT

        Args:
            input_signal: Array of input samples.

        Returns:
            np.ndarray: Filtered signal.
        """
        number_of_samples: int = len(input_signal)

        # Compute frequency axis
        frequency_axis: np.ndarray = np.fft.fftfreq(
            number_of_samples,
            d=1.0 / self.sampling_frequency_hz
        )

        # Compute FFT
        spectrum: np.ndarray = np.fft.fft(input_signal)
        # Create brick-wall filter mask
        # Pass frequencies where |f| <= cutoff, block others
        filter_mask: np.ndarray = (
            np.abs(frequency_axis) <= self.cutoff_frequency_hz
        )

        # Apply filter (zero out blocked frequencies)
        filtered_spectrum:  np.ndarray = spectrum * filter_mask

        # Inverse FFT to get time-domain signal
        filtered_signal: np.ndarray = np.real(np.fft.ifft(filtered_spectrum))

        return filtered_signal