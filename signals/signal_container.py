"""
Signal Container
================

This module provides a data class for storing and organizing signals
throughout the delta-sigma DAC simulation pipeline.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class SignalContainer:
    """
    Container for storing signals at various stages of the DSM-DAC pipeline.

    The signal flow in a Delta-Sigma DAC is: 
    [Digital PCM Input] → [Delta-Sigma Modulator] → [1-bit Bitstream] →
    [Reconstruction Filter] → [Analog Output]

    Attributes:
        time_axis_seconds: Time values for each sample point.
        input_signal_digital_pcm:  The original high-resolution digital input.
        modulator_output_bitstream: The 1-bit output of the delta-sigma modulator. 
        reconstructed_analog_signal: The signal after the reconstruction filter.
        integrator_state_history: Internal modulator states for stability analysis.
    """
    # Required attributes
    time_axis_seconds: np.ndarray
    input_signal_digital_pcm:  np.ndarray

    # Optional attributes (filled during simulation)
    modulator_output_bitstream: Optional[np.ndarray] = None
    reconstructed_analog_signal: Optional[np.ndarray] = None
    integrator_state_history:  Optional[np.ndarray] = None

    # Metadata
    sampling_frequency_hz: float = 0.0
    signal_frequency_hz: float = 0.0
    oversampling_ratio: int = 0
    modulator_order: int = 0

    def validate(self) -> bool:
        """Validate that all arrays have consistent lengths."""
        expected_length: int = len(self.time_axis_seconds)

        if len(self.input_signal_digital_pcm) != expected_length:
            raise ValueError("Input signal length mismatch")

        if self.modulator_output_bitstream is not None:
            if len(self.modulator_output_bitstream) != expected_length: 
                raise ValueError("Modulator output length mismatch")

        if self.reconstructed_analog_signal is not None:
            if len(self.reconstructed_analog_signal) != expected_length:
                raise ValueError("Reconstructed signal length mismatch")

        return True

    def get_number_of_samples(self) -> int:
        """Return the number of samples in the container."""
        return len(self.time_axis_seconds)