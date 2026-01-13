"""
Cascaded RC Low-Pass Filter
===========================

This module implements a cascaded (multi-stage) RC low-pass filter. 

WHY CASCADED STAGES ARE NEEDED:
===============================
A single first-order RC filter attenuates at only 20 dB/decade. 
However, delta-sigma modulators push noise up at: 
- 1st order DSM: 20 dB/decade  → 1 RC stage needed
- 2nd order DSM: 40 dB/decade  → 2 RC stages needed  
- 3rd order DSM: 60 dB/decade  → 3 RC stages needed
- Nth order DSM: N×20 dB/decade → N RC stages needed

By cascading N identical RC stages, we get approximately N×20 dB/decade
roll-off, which matches the noise shaping of an Nth order modulator.

PHYSICAL IMPLEMENTATION:
========================
On your PCB, this would look like: 

    FPGA Pin ──┬── R1 ──┬── R2 ──┬── R3 ──┬── Output
               │        │        │        │
              C1       C2       C3      (Load)
               │        │        │
              GND      GND      GND

Each stage is an independent RC filter.  The stages are buffered
in this simulation (no loading effects), which is a simplification. 
In real hardware, you might need buffer amplifiers between stages
or account for loading in your component selection.

CUTOFF FREQUENCY NOTE:
=====================
When you cascade N identical RC stages, the overall -3dB point shifts. 
The combined -3dB frequency is approximately:
    f_combined ≈ f_stage × sqrt(2^(1/N) - 1)

For example, with 3 stages of 15 kHz each:
    f_combined ≈ 15 kHz × 0.51 ≈ 7.6 kHz

To compensate, we can either:
1. Accept the lower combined cutoff (simpler)
2. Increase each stage's cutoff to achieve desired combined cutoff

This implementation offers both options.
"""

import numpy as np
from typing import List, Optional


class CascadedRCLowPassFilter:
    """
    Multi-stage cascaded RC low-pass filter for delta-sigma reconstruction. 

    This filter consists of N identical first-order RC stages in series. 
    Each stage is modeled as: 
        y[n] = α × x[n] + (1 - α) × y[n-1]

    Where α = 1 - exp(-T/τ) and τ = RC = 1/(2πf_c)

    The cascaded stages provide steeper roll-off: 
    - 1 stage:   -20 dB/decade
    - 2 stages: -40 dB/decade
    - 3 stages: -60 dB/decade
    - N stages:  -N×20 dB/decade

    Attributes:
        number_of_stages (int): Number of cascaded RC stages. 
        stage_cutoff_frequency_hz (float): Cutoff of each individual stage. 
        sampling_frequency_hz (float): Digital sampling rate.
        alpha (float): Filter coefficient for each stage.
        stage_states (List[float]): Output state of each stage. 

    FPGA/Hardware Relevance:
        This directly models cascaded RC stages on your PCB.
        Component values for each stage: 
            R = chosen resistance (e.g., 10 kΩ)
            C = 1 / (2π × f_c × R)
    """

    def __init__(
        self,
        number_of_stages: int,
        cutoff_frequency_hz: float,
        sampling_frequency_hz:  float,
        compensate_for_cascade: bool = True
    ) -> None:
        """
        Initialize the cascaded RC low-pass filter.

        Args:
            number_of_stages:  Number of RC stages to cascade.
                Rule of thumb: Use at least as many stages as DSM order.
                - DSM Order 1 → 1-2 stages
                - DSM Order 2 → 2-3 stages
                - DSM Order 3 → 3-4 stages

            cutoff_frequency_hz:  Desired overall -3dB cutoff frequency. 
                Should be above signal frequency but below noise band. 
                Typical:  1.2× to 2× the maximum signal frequency.

            sampling_frequency_hz: The sampling rate in Hz. 

            compensate_for_cascade:  If True, adjusts each stage's cutoff
                so the combined response has -3dB at cutoff_frequency_hz.
                If False, each stage uses cutoff_frequency_hz directly
                (combined -3dB will be lower).

        Raises:
            ValueError:  If parameters are invalid.
        """
        # ===== INPUT VALIDATION =====
        if number_of_stages < 1:
            raise ValueError(
                f"Number of stages must be at least 1. "
                f"Received:  {number_of_stages}"
            )

        if cutoff_frequency_hz <= 0:
            raise ValueError(
                f"Cutoff frequency must be positive.  "
                f"Received:  {cutoff_frequency_hz} Hz"
            )

        nyquist_frequency_hz:  float = sampling_frequency_hz / 2.0
        if cutoff_frequency_hz >= nyquist_frequency_hz:
            raise ValueError(
                f"Cutoff frequency ({cutoff_frequency_hz} Hz) must be less than "
                f"Nyquist frequency ({nyquist_frequency_hz} Hz)"
            )

        # ===== STORE PARAMETERS =====
        self.number_of_stages:  int = number_of_stages
        self.desired_cutoff_frequency_hz:  float = cutoff_frequency_hz
        self.sampling_frequency_hz: float = sampling_frequency_hz
        self.compensate_for_cascade: bool = compensate_for_cascade

        # ===== CALCULATE STAGE CUTOFF FREQUENCY =====
        # When cascading N identical stages, the -3dB point shifts down. 
        # To compensate, we increase each stage's cutoff frequency.
        #
        # For N cascaded identical RC filters:
        # |H(f)|² = 1 / (1 + (f/f_c)²)^N
        #
        # At combined -3dB point (|H|² = 0.5):
        # (1 + (f_3dB/f_stage)²)^N = 2
        # f_3dB/f_stage = sqrt(2^(1/N) - 1)
        #
        # Therefore: f_stage = f_3dB / sqrt(2^(1/N) - 1)

        if compensate_for_cascade and number_of_stages > 1:
            # Calculate compensation factor
            compensation_factor:  float = np.sqrt(2 ** (1.0 / number_of_stages) - 1)
            self.stage_cutoff_frequency_hz:  float = (
                cutoff_frequency_hz / compensation_factor
            )
        else:
            # No compensation - each stage uses the specified cutoff
            self.stage_cutoff_frequency_hz = cutoff_frequency_hz

        # Ensure stage cutoff doesn't exceed Nyquist
        if self.stage_cutoff_frequency_hz >= nyquist_frequency_hz: 
            self.stage_cutoff_frequency_hz = nyquist_frequency_hz * 0.9
            print(
                f"WARNING: Stage cutoff clamped to {self.stage_cutoff_frequency_hz:.1f} Hz "
                f"to stay below Nyquist.  Consider reducing desired cutoff or stages."
            )

        # ===== CALCULATE FILTER COEFFICIENT =====
        # Time constant:  τ = 1 / (2π × f_c)
        self.time_constant_seconds: float = (
            1.0 / (2.0 * np.pi * self.stage_cutoff_frequency_hz)
        )

        # Sample period
        sample_period_seconds: float = 1.0 / sampling_frequency_hz

        # Filter coefficient: α = 1 - exp(-T/τ)
        # This comes from the exact solution of the RC differential equation
        self.alpha: float = 1.0 - np.exp(
            -sample_period_seconds / self.time_constant_seconds
        )

        # ===== INITIALIZE STAGE STATES =====
        # Each stage maintains its own output state (the capacitor voltage)
        self.stage_states: List[float] = [0.0] * number_of_stages

    def filter_sample(self, input_sample: float) -> float:
        """
        Filter a single sample through all cascaded stages.

        Each stage processes the output of the previous stage: 
        Stage 1 input = filter input
        Stage 2 input = Stage 1 output
        Stage 3 input = Stage 2 output
        ...and so on.

        The difference equation for each stage is:
            y[n] = α × x[n] + (1 - α) × y[n-1]

        This is equivalent to:  output approaches input with time constant τ.

        Args:
            input_sample: The input value to filter.

        Returns:
            float: The filtered output value (from last stage).

        VHDL Implementation Note:
            Each stage is a simple first-order IIR filter: 

            -- For each stage (assuming fixed-point):
            stage_output <= alpha * stage_input + (1-alpha) * stage_output_prev;

            With careful fixed-point scaling to avoid overflow.
        """
        # Start with the input sample
        current_value: float = input_sample

        # Process through each stage sequentially
        for stage_index in range(self.number_of_stages):
            # First-order RC filter difference equation: 
            # y[n] = α × x[n] + (1 - α) × y[n-1]
            #
            # Where: 
            # - x[n] is current_value (input to this stage)
            # - y[n-1] is self.stage_states[stage_index] (previous output)
            # - y[n] is the new output

            new_output: float = (
                self.alpha * current_value
                + (1.0 - self.alpha) * self.stage_states[stage_index]
            )

            # Store the new state for this stage
            self.stage_states[stage_index] = new_output

            # The output of this stage becomes the input to the next stage
            current_value = new_output

        # Return the output of the final stage
        return current_value

    def filter_signal(self, input_signal: np.ndarray) -> np.ndarray:
        """
        Filter an entire signal through all cascaded stages. 

        This processes each sample sequentially, maintaining state
        between samples (like real hardware would).

        Args:
            input_signal: Array of input samples to filter.

        Returns:
            np.ndarray: Filtered signal (same length as input).
        """
        number_of_samples: int = len(input_signal)
        output_signal: np.ndarray = np.zeros(number_of_samples)

        # Process each sample
        for sample_index in range(number_of_samples):
            output_signal[sample_index] = self.filter_sample(
                input_signal[sample_index]
            )

        return output_signal

    def reset(self) -> None:
        """
        Reset all stage states to zero.

        Call this before processing a new signal to ensure
        clean initial conditions. 

        FPGA Equivalent:  Assert reset signal to clear all registers.
        """
        self.stage_states = [0.0] * self.number_of_stages

    def get_stage_states(self) -> List[float]:
        """Return current states of all stages (for debugging)."""
        return self.stage_states.copy()

    def get_component_values(
        self,
        target_resistance_ohms: float = 10000.0
    ) -> dict:
        """
        Calculate suggested R and C values for physical implementation.

        Args:
            target_resistance_ohms:  Desired resistance value for each stage.
                Common values: 1kΩ, 4.7kΩ, 10kΩ, 47kΩ. 
                Consider FPGA output impedance and load requirements.

        Returns:
            dict: Component values and filter characteristics.
        """
        # C = τ / R = 1 / (2π × f_c × R)
        capacitance_farads: float = (
            self.time_constant_seconds / target_resistance_ohms
        )

        # Calculate actual combined -3dB frequency
        if self.number_of_stages > 1:
            cascade_factor: float = np.sqrt(
                2 ** (1.0 / self.number_of_stages) - 1
            )
            actual_combined_cutoff_hz: float = (
                self.stage_cutoff_frequency_hz * cascade_factor
            )
        else:
            actual_combined_cutoff_hz = self.stage_cutoff_frequency_hz

        return {
            "number_of_stages": self.number_of_stages,
            "resistance_ohms_per_stage": target_resistance_ohms,
            "resistance_kohms_per_stage": target_resistance_ohms / 1000,
            "capacitance_farads_per_stage": capacitance_farads,
            "capacitance_nanofarads_per_stage":  capacitance_farads * 1e9,
            "capacitance_picofarads_per_stage": capacitance_farads * 1e12,
            "time_constant_microseconds":  self.time_constant_seconds * 1e6,
            "stage_cutoff_frequency_hz":  self.stage_cutoff_frequency_hz,
            "combined_cutoff_frequency_hz": actual_combined_cutoff_hz,
            "rolloff_db_per_decade": 20 * self.number_of_stages,
            "alpha_coefficient": self.alpha
        }

    def get_frequency_response(
        self,
        frequencies_hz: Optional[np.ndarray] = None,
        num_points: int = 1000
    ) -> tuple:
        """
        Calculate the filter's frequency response.

        For N cascaded identical RC stages:
        |H(f)|² = 1 / (1 + (f/f_c)²)^N

        Args:
            frequencies_hz:  Specific frequencies to evaluate.
                If None, generates logarithmically spaced points.
            num_points: Number of frequency points if generating. 

        Returns:
            Tuple of (frequencies_hz, magnitude_db, phase_degrees)
        """
        if frequencies_hz is None:
            # Generate frequency points from 1 Hz to Nyquist
            frequencies_hz = np.logspace(
                0,  # 10^0 = 1 Hz
                np.log10(self.sampling_frequency_hz / 2),
                num_points
            )

        # Normalized frequency ratio
        frequency_ratio: np.ndarray = frequencies_hz / self.stage_cutoff_frequency_hz

        # Magnitude of single stage:  |H| = 1 / sqrt(1 + (f/f_c)²)
        # Magnitude of N stages: |H| = 1 / sqrt(1 + (f/f_c)²)^N
        magnitude:  np.ndarray = 1.0 / np.sqrt(
            (1.0 + frequency_ratio ** 2) ** self.number_of_stages
        )

        # Convert to dB
        magnitude_db: np.ndarray = 20.0 * np.log10(magnitude + 1e-20)

        # Phase of single stage: φ = -arctan(f/f_c)
        # Phase of N stages: φ = -N × arctan(f/f_c)
        phase_radians: np.ndarray = (
            -self.number_of_stages * np.arctan(frequency_ratio)
        )
        phase_degrees: np.ndarray = np.degrees(phase_radians)

        return frequencies_hz, magnitude_db, phase_degrees

    def print_info(self) -> None:
        """Print detailed filter information."""
        print("\n" + "=" * 60)
        print("CASCADED RC LOW-PASS FILTER INFORMATION")
        print("=" * 60)

        print(f"\n--- Configuration ---")
        print(f"  Number of Stages:         {self.number_of_stages}")
        print(f"  Desired Cutoff:          {self.desired_cutoff_frequency_hz / 1000:.2f} kHz")
        print(f"  Stage Cutoff:            {self.stage_cutoff_frequency_hz / 1000:.2f} kHz")
        print(f"  Sampling Frequency:      {self.sampling_frequency_hz / 1e6:.3f} MHz")
        print(f"  Cascade Compensation:    {'Enabled' if self.compensate_for_cascade else 'Disabled'}")

        print(f"\n--- Filter Characteristics ---")
        print(f"  Roll-off:                 {20 * self.number_of_stages} dB/decade")
        print(f"  Time Constant (τ):       {self.time_constant_seconds * 1e6:.2f} µs")
        print(f"  Alpha Coefficient:       {self.alpha:.6f}")

        # Get component values
        components = self.get_component_values(target_resistance_ohms=10000)
        print(f"\n--- Suggested Components (per stage) ---")
        print(f"  Resistance:              {components['resistance_kohms_per_stage']:.1f} kΩ")
        print(f"  Capacitance:              {components['capacitance_nanofarads_per_stage']:.2f} nF")

        # Show attenuation at key frequencies
        print(f"\n--- Attenuation at Key Frequencies ---")
        test_frequencies = [
            self.desired_cutoff_frequency_hz,
            self.desired_cutoff_frequency_hz * 2,
            self.desired_cutoff_frequency_hz * 5,
            self.desired_cutoff_frequency_hz * 10,
        ]

        for freq in test_frequencies: 
            if freq < self.sampling_frequency_hz / 2:
                _, mag_db, _ = self.get_frequency_response(
                    frequencies_hz=np.array([freq])
                )
                print(f"  @ {freq/1000:6.1f} kHz: {mag_db[0]: 7.1f} dB")

        print("\n" + "=" * 60)


def get_recommended_filter_stages(dsm_order: int, target_quality:  str = "good") -> int:
    """
    Get recommended number of RC filter stages for a given DSM order.
    
    Based on empirical testing, filter stages should exceed DSM order
    to achieve good noise suppression. 
    
    Args:
        dsm_order: Order of the delta-sigma modulator (1-5).
        target_quality: 
            - "minimum": DSM order + 1 (functional but not optimal)
            - "good":  DSM order + 2 (recommended for most applications)
            - "excellent":  DSM order + 3 (near-ideal, more hardware)
    
    Returns:
        int:  Recommended number of filter stages.
    
    Example:
        For DSM Order 2:
        - "minimum" → 3 stages (~45 dB SNR)
        - "good" → 4 stages (~53 dB SNR)  
        - "excellent" → 5 stages (~56 dB SNR)
    """
    quality_map = {
        "minimum": 1,
        "good": 2,
        "excellent": 3
    }
    
    extra_stages = quality_map.get(target_quality.lower(), 2)
    
    return dsm_order + extra_stages


def create_filter_for_modulator_order(
    modulator_order: int,
    cutoff_frequency_hz: float,
    sampling_frequency_hz:  float,
    quality: str = "good"
) -> 'CascadedRCLowPassFilter':
    """
    Factory function to create a properly-sized filter for a DSM. 
    
    This function automatically selects the number of filter stages
    based on the modulator order and desired quality level.
    
    Args:
        modulator_order: Order of the delta-sigma modulator. 
        cutoff_frequency_hz: Desired cutoff frequency. 
        sampling_frequency_hz:  Sampling rate. 
        quality: "minimum", "good", or "excellent".
    
    Returns:
        CascadedRCLowPassFilter: Configured filter instance.
    """
    num_stages = get_recommended_filter_stages(modulator_order, quality)
    
    print(f"Creating {num_stages}-stage RC filter for Order {modulator_order} DSM "
          f"(quality='{quality}')")
    
    return CascadedRCLowPassFilter(
        number_of_stages=num_stages,
        cutoff_frequency_hz=cutoff_frequency_hz,
        sampling_frequency_hz=sampling_frequency_hz,
        compensate_for_cascade=True
    )