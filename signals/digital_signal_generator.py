"""
Digital Signal Generator
========================

This module provides a class for generating digital PCM (Pulse Code Modulation)
signals that simulate the input to a Delta-Sigma DAC. 

PCM = Pulse Code Modulation:  A method to digitally represent analog signals
      where the amplitude is sampled at regular intervals and quantized to
      the nearest value within a range of digital steps. 

In the context of a DAC: 
- Input: High-resolution digital samples (e.g., 16-bit, 24-bit)
- Output: These samples will be converted to analog via the delta-sigma process
"""

import numpy as np
from typing import Dict


class DigitalSignalGenerator: 
    """
    Generates digital PCM signals for delta-sigma DAC simulation. 

    This class creates test signals (primarily sinusoidal) that represent
    the digital input to a DAC system. The signals are normalized to the
    range [-1, 1] to match typical DAC full-scale conventions.

    Attributes:
        sampling_frequency_hz (float): The rate at which samples are generated. 
            This represents the OVERSAMPLED rate (FPGA clock frequency),
            not the original signal's Nyquist rate. 

        number_of_samples (int): Total number of samples to generate. 

        word_length_bits (int): The bit depth of the PCM signal. 
            Common values:  16 bits (CD quality), 24 bits (professional audio).
            This determines the quantization resolution of the INPUT signal.

    FPGA Relevance:
        - word_length_bits determines the width of your input data bus
        - sampling_frequency_hz corresponds to your FPGA clock rate
        - The generated signal can be stored in FPGA block RAM for testing
    """

    def __init__(
        self,
        sampling_frequency_hz: float,
        number_of_samples:  int,
        word_length_bits: int = 16
    ) -> None:
        """
        Initialize the digital signal generator. 

        Args:
            sampling_frequency_hz:  The sampling rate in Hertz (Hz).
                For FPGA simulation, this is typically the oversampled rate. 
                Example: 200 MHz = 200_000_000 Hz

            number_of_samples: How many samples to generate.
                Should be large enough to capture several periods of the
                lowest frequency signal you want to analyze. 
                Recommendation: At least 2^13 = 8192 for good FFT resolution.

            word_length_bits:  Bit depth of the digital signal.
                Determines the quantization levels:  2^word_length_bits levels.
                Default is 16 bits = 65,536 levels. 

        Raises:
            ValueError:  If any parameter is invalid (non-positive or too small).
        """
        # ===== INPUT VALIDATION =====
        if sampling_frequency_hz <= 0:
            raise ValueError(
                f"Sampling frequency must be positive. "
                f"Received: {sampling_frequency_hz} Hz"
            )

        if number_of_samples < 2: 
            raise ValueError(
                f"Number of samples must be at least 2. "
                f"Received: {number_of_samples}"
            )

        if word_length_bits < 1 or word_length_bits > 32:
            raise ValueError(
                f"Word length must be between 1 and 32 bits.  "
                f"Received:  {word_length_bits} bits"
            )

        # ===== STORE PARAMETERS =====
        self.sampling_frequency_hz:  float = sampling_frequency_hz
        self.number_of_samples: int = number_of_samples
        self.word_length_bits: int = word_length_bits

        # ===== DERIVED QUANTITIES =====
        # Calculate the number of quantization levels
        # For N bits, we have 2^N possible values
        self.number_of_quantization_levels:  int = 2 ** word_length_bits

        # Calculate the quantization step size for the [-1, 1] range
        # Step size = full range / (number of levels - 1)
        # We use (levels - 1) because we want to include both -1 and +1
        self.quantization_step_size: float = 2.0 / (self.number_of_quantization_levels - 1)

        # Generate the time axis (in seconds)
        # This array represents the time instant of each sample
        self.time_axis_seconds: np.ndarray = np.arange(
            self.number_of_samples
        ) / self.sampling_frequency_hz

    def generate_sinusoidal_signal(
        self,
        signal_frequency_hz: float,
        amplitude: float = 0.5,
        phase_radians: float = 0.0,
        direct_current_offset: float = 0.0
    ) -> np.ndarray:
        """
        Generate a quantized sinusoidal signal. 

        This method creates a sine wave and quantizes it to simulate
        a real digital signal with finite word length.

        The mathematical formula is:
            signal[n] = amplitude * sin(2 * π * frequency * t[n] + phase) + dc_offset

        Then the signal is quantized to the specified word length.

        Args:
            signal_frequency_hz:  Frequency of the sine wave in Hertz. 
                Must be less than sampling_frequency / 2 (Nyquist criterion).
                For your application:  1 kHz to 10 kHz. 

            amplitude: Peak amplitude of the sine wave (0 to 1).
                Default is 0.5 to leave headroom and avoid clipping. 
                IMPORTANT: For delta-sigma modulators, keep amplitude <= 0.7
                to maintain stability, especially for higher orders. 
                Higher orders require lower input amplitude for stability.

            phase_radians: Initial phase of the sine wave in radians. 
                Default is 0 (starts at zero crossing, going positive).

            direct_current_offset: DC offset added to the signal.
                Default is 0 (centered around zero).
                The final signal must stay within [-1, 1]. 

        Returns:
            np.ndarray: The quantized sinusoidal signal as a 1D array.
                Values are in the range [-1, 1]. 
                Length equals self.number_of_samples. 

        Raises:
            ValueError:  If parameters would result in invalid signal. 

        FPGA Relevance:
            - The quantized signal represents what you'd store in a lookup table
              (LUT) or receive from an external digital source
            - The quantization models the finite precision of FPGA arithmetic
        """
        # ===== INPUT VALIDATION =====
        # Check Nyquist criterion:  signal frequency must be less than half
        # the sampling frequency to avoid aliasing
        nyquist_frequency_hz:  float = self.sampling_frequency_hz / 2.0
        if signal_frequency_hz >= nyquist_frequency_hz: 
            raise ValueError(
                f"Signal frequency ({signal_frequency_hz} Hz) must be less than "
                f"Nyquist frequency ({nyquist_frequency_hz} Hz). "
                f"This is required to prevent aliasing."
            )

        if signal_frequency_hz <= 0:
            raise ValueError(
                f"Signal frequency must be positive. "
                f"Received: {signal_frequency_hz} Hz"
            )

        # Check amplitude is valid
        if amplitude <= 0 or amplitude > 1.0:
            raise ValueError(
                f"Amplitude must be in range (0, 1].  "
                f"Received:  {amplitude}"
            )

        # Check that signal will stay within [-1, 1]
        if abs(direct_current_offset) + amplitude > 1.0:
            raise ValueError(
                f"Signal would exceed [-1, 1] range. "
                f"Amplitude ({amplitude}) + |DC offset| ({abs(direct_current_offset)}) > 1.0"
            )

        # ===== GENERATE IDEAL SINUSOID =====
        # Calculate angular frequency:  ω = 2πf
        angular_frequency_radians_per_second: float = 2.0 * np.pi * signal_frequency_hz

        # Generate the ideal (infinite precision) sinusoidal signal
        # Formula: A * sin(ωt + φ) + DC
        ideal_signal: np.ndarray = (
            amplitude * np.sin(
                angular_frequency_radians_per_second * self.time_axis_seconds
                + phase_radians
            )
            + direct_current_offset
        )

        # ===== QUANTIZE THE SIGNAL =====
        # Quantization simulates the finite word length of digital systems
        quantized_signal: np.ndarray = self._quantize_signal(ideal_signal)

        return quantized_signal

    def _quantize_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Quantize a signal to the specified word length.

        This is a uniform mid-tread quantizer that maps continuous values
        to discrete levels. "Mid-tread" means that zero maps exactly to zero. 

        Quantization Process:
        1. Clip signal to [-1, 1] range (safety)
        2. Map [-1, 1] to [0, N-1] where N = number of levels
        3. Round to nearest integer (quantization)
        4. Map back to [-1, 1] range

        Args:
            signal: Input signal array with values ideally in [-1, 1]. 

        Returns:
            np.ndarray: Quantized signal with discrete levels.
        """
        # Step 1: Clip to valid range (in case of numerical errors)
        clipped_signal: np.ndarray = np.clip(signal, -1.0, 1.0)

        # Step 2: Map from [-1, 1] to [0, N-1]
        normalized_indices: np.ndarray = (
            (clipped_signal + 1.0) / self.quantization_step_size
        )

        # Step 3: Round to nearest integer (actual quantization)
        quantized_indices: np.ndarray = np.round(normalized_indices)

        # Step 4: Map back to [-1, 1]
        quantized_signal: np.ndarray = (
            -1.0 + quantized_indices * self.quantization_step_size
        )

        return quantized_signal

    def get_time_axis(self) -> np.ndarray:
        """Return the time axis for plotting."""
        return self.time_axis_seconds.copy()

    def get_signal_parameters_summary(
        self,
        signal_frequency_hz: float
    ) -> Dict[str, float]:
        """
        Calculate and return useful signal parameters for analysis.

        Args:
            signal_frequency_hz: The frequency of the signal being analyzed. 

        Returns:
            dict: Dictionary containing signal parameters. 
        """
        samples_per_period: float = self.sampling_frequency_hz / signal_frequency_hz
        signal_duration_seconds: float = self.number_of_samples / self.sampling_frequency_hz
        number_of_complete_periods: float = signal_frequency_hz * signal_duration_seconds
        frequency_resolution_hz: float = self.sampling_frequency_hz / self.number_of_samples

        return {
            "samples_per_period": samples_per_period,
            "number_of_complete_periods": number_of_complete_periods,
            "actual_frequency_resolution_hz": frequency_resolution_hz,
            "nyquist_frequency_hz": self.sampling_frequency_hz / 2.0,
            "total_duration_seconds": signal_duration_seconds,
            "quantization_levels": self.number_of_quantization_levels,
            "quantization_step_size": self. quantization_step_size
        }