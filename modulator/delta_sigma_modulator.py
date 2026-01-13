"""
Delta-Sigma Modulator Module
============================

This is the CORE module of the delta-sigma DAC simulation. 

A Delta-Sigma Modulator (DSM) converts a high-resolution, low-frequency
digital signal into a low-resolution (typically 1-bit), high-frequency
bitstream. The key innovation is NOISE SHAPING. 

================================================================================
ARCHITECTURE:  CIFB (Cascade of Integrators with Distributed FeedBack)
================================================================================

This implementation uses the CIFB topology, which is standard for
delta-sigma modulators.  The key characteristic is that the feedback
from the quantizer is distributed to ALL integrator stages. 

BLOCK DIAGRAM (Second Order Example):
=====================================

                    ┌─────────────────────────────────────────────────────┐
                    │                     FEEDBACK PATH                    │
                    │                                                      │
    Input ──────────┼──(+)────►[Integrator 1]────(+)────►[Integrator 2]───►[Quantizer]──┬──► Output
           x[n]     │   │                         │                                     │
                    │   │ (-)                     │ (-)                                 │
                    │   │                         │                                     │
                    │   └─────────────────────────┴─────────[Feedback DAC]◄─────────────┘
                    │              c1=1                  c2=1              v[n-1]
                    └─────────────────────────────────────────────────────────────────────

    Where:
    - x[n] is the input sample at time n
    - v[n] is the quantizer output at time n
    - c1, c2 are feedback coefficients (typically 1.0 for basic topology)

DIFFERENCE EQUATIONS (Verified against academic sources):
=========================================================

FIRST ORDER (N=1):
    integrator_1[n] = integrator_1[n-1] + (x[n] - v[n-1])
    v[n] = quantize(integrator_1[n])

SECOND ORDER (N=2):
    integrator_1[n] = integrator_1[n-1] + (x[n] - v[n-1])
    integrator_2[n] = integrator_2[n-1] + (integrator_1[n] - v[n-1])
    v[n] = quantize(integrator_2[n])

THIRD ORDER (N=3):
    integrator_1[n] = integrator_1[n-1] + (x[n] - v[n-1])
    integrator_2[n] = integrator_2[n-1] + (integrator_1[n] - v[n-1])
    integrator_3[n] = integrator_3[n-1] + (integrator_2[n] - v[n-1])
    v[n] = quantize(integrator_3[n])

GENERAL N-TH ORDER: 
    integrator_1[n] = integrator_1[n-1] + (x[n] - c1 * v[n-1])
    For k = 2 to N:
        integrator_k[n] = integrator_k[n-1] + (integrator_{k-1}[n] - c_k * v[n-1])
    v[n] = quantize(integrator_N[n])

    Where c_k are feedback coefficients (all = 1.0 for standard topology)

NOISE TRANSFER FUNCTION (NTF):
==============================
For an N-th order modulator with all feedback coefficients = 1:
    NTF(z) = (1 - z^-1)^N

This means quantization noise is shaped with N zeros at DC (z=1), pushing
noise energy to higher frequencies where it can be filtered out.

- 1st order: NTF = (1 - z^-1)       → 20 dB/decade noise attenuation
- 2nd order: NTF = (1 - z^-1)^2     → 40 dB/decade noise attenuation
- 3rd order: NTF = (1 - z^-1)^3     → 60 dB/decade noise attenuation

STABILITY CONSIDERATIONS:
=========================
Higher order modulators (N >= 3) can become unstable if:
- Input amplitude is too high (keep below 0.7 for order 3+)
- Integrators overflow

For your FPGA implementation: 
- Order 1-2: Very stable, recommended starting point
- Order 3: Generally stable with amplitude < 0.6
- Order 4-5: May require amplitude < 0.5 and careful design

REFERENCES:
===========
1. Oregon State University:  First-order Delta-Sigma Modulator
   https://web.engr.oregonstate.edu/~temes/ece627/Lecture_Notes/First_Order_DS_ADC.pdf
2. Wikibooks: Digital Circuits/Sigma-Delta Modulators
3. Analog Devices MT-022: ADC Architectures
4. R. Schreier, "Understanding Delta-Sigma Data Converters"
"""

import numpy as np
from typing import List, Tuple, Optional

from .quantizer import AbstractQuantizer, BinaryQuantizer
from .feedback_digital_to_analog_converter import FeedbackDigitalToAnalogConverter
from .integrator import Integrator


class DeltaSigmaModulator: 
    """
    N-th order delta-sigma modulator using CIFB topology.

    This class implements a single-loop delta-sigma modulator with
    distributed feedback.  Each integrator receives feedback from the
    quantizer output, weighted by a coefficient.

    Attributes:
        modulator_order (int): Number of integrator stages (1-5 recommended).
        quantizer (AbstractQuantizer): The quantizer instance.
        feedback_dac (FeedbackDigitalToAnalogConverter): Feedback path model.
        integrators (List[Integrator]): List of integrator instances.
        feedback_coefficients (List[float]): Coefficients for each feedback path.
        previous_output (float): Stored output for feedback delay (z^-1).

    FPGA Resource Estimation (for 1-bit quantizer):
        Order 1: ~1 adder, ~1 comparator, ~24-32 bit accumulator
        Order 2: ~2 adders, ~1 comparator, ~2×(24-32 bit) accumulators
        Order N: ~N adders, ~1 comparator, ~N accumulators
    """

    def __init__(
        self,
        modulator_order: int,
        quantizer: AbstractQuantizer,
        feedback_dac:  FeedbackDigitalToAnalogConverter,
        feedback_coefficients: Optional[List[float]] = None,
        integrator_saturation_limit: Optional[float] = None
    ) -> None:
        """
        Initialize the delta-sigma modulator.

        Args:
            modulator_order: Number of integrator stages (1 to 5 recommended).
                Higher orders give better noise shaping but risk instability. 

            quantizer: The quantizer to use (typically BinaryQuantizer).

            feedback_dac: The feedback path model. 

            feedback_coefficients:  Coefficients for feedback to each integrator.
                Length must equal modulator_order. 
                Default is [1.0, 1.0, ..., 1.0] (unity feedback to all stages).

            integrator_saturation_limit:  Limits integrator magnitude.
                Recommended for orders >= 3 to prevent instability.
                Typical value: 2.0 to 4.0

        Raises:
            ValueError:  If modulator_order < 1 or coefficients mismatch.
        """
        # ===== INPUT VALIDATION =====
        if modulator_order < 1:
            raise ValueError(
                f"Modulator order must be at least 1. Received:  {modulator_order}"
            )

        if modulator_order > 5:
            print(
                f"WARNING: Order {modulator_order} modulators can be unstable. "
                f"Consider using order 1-3 for reliable operation."
            )

        # ===== STORE BASIC PARAMETERS =====
        self.modulator_order: int = modulator_order
        self.quantizer: AbstractQuantizer = quantizer
        self.feedback_dac: FeedbackDigitalToAnalogConverter = feedback_dac

        # ===== SET UP FEEDBACK COEFFICIENTS =====
        # Default:  unity feedback to all integrators (standard CIFB topology)
        if feedback_coefficients is None:
            # All coefficients = 1.0 gives NTF = (1 - z^-1)^N
            self.feedback_coefficients: List[float] = [1.0] * modulator_order
        else: 
            if len(feedback_coefficients) != modulator_order:
                raise ValueError(
                    f"Number of feedback coefficients ({len(feedback_coefficients)}) "
                    f"must equal modulator order ({modulator_order})"
                )
            self.feedback_coefficients = list(feedback_coefficients)

        # ===== CREATE INTEGRATORS =====
        self.integrators: List[Integrator] = [
            Integrator(
                initial_state=0.0,
                saturation_limit=integrator_saturation_limit
            )
            for _ in range(modulator_order)
        ]

        # ===== INITIALIZE FEEDBACK STATE =====
        # The previous output is needed for feedback (z^-1 delay element)
        # Initialize to 0 (no feedback on first sample)
        self.previous_output: float = 0.0

    def process_sample(self, input_sample: float) -> float:
        """
        Process a single input sample through the modulator.

        This implements the CIFB difference equations: 

        For first-order: 
            integrator_1[n] = integrator_1[n-1] + (x[n] - v[n-1])
            v[n] = quantize(integrator_1[n])

        For N-th order:
            integrator_1[n] = integrator_1[n-1] + (x[n] - c1*v[n-1])
            integrator_k[n] = integrator_k[n-1] + (integrator_{k-1}[n] - c_k*v[n-1])
            v[n] = quantize(integrator_N[n])

        Args:
            input_sample: The input value (typically in range [-1, 1]).

        Returns:
            float: The quantizer output (+1 or -1 for binary quantizer).

        VHDL Implementation Sketch:
        ---------------------------
        -- Feedback register (z^-1 delay)
        previous_output_reg <= current_output;

        -- First integrator: error = input - feedback
        error_1 <= input_sample - previous_output_reg;
        integrator_1 <= integrator_1 + error_1;

        -- Second integrator (if order >= 2)
        error_2 <= integrator_1 - previous_output_reg;
        integrator_2 <= integrator_2 + error_2;

        -- Quantizer (1-bit = sign bit extraction)
        if integrator_N(MSB) = '0' then  -- positive
            current_output <= +1;
        else  -- negative
            current_output <= -1;
        end if;
        """
        # =================================================================
        # STEP 1: GET FEEDBACK VALUE
        # =================================================================
        # The feedback is the previous quantizer output passed through the DAC
        # For ideal 1-bit DAC, this is just +1 or -1
        # The z^-1 delay is implemented by using self.previous_output
        feedback_value:  float = self.feedback_dac.convert(self.previous_output)

        # =================================================================
        # STEP 2: UPDATE FIRST INTEGRATOR
        # =================================================================
        # First integrator receives:  input - (c1 * feedback)
        # This computes the error signal (delta)
        first_stage_error: float = (
            input_sample - self.feedback_coefficients[0] * feedback_value
        )
        # Integrate the error:  accumulator += error
        self.integrators[0].integrate(first_stage_error)

        # =================================================================
        # STEP 3: UPDATE REMAINING INTEGRATORS (for order > 1)
        # =================================================================
        # Each subsequent integrator receives: 
        # (output of previous integrator) - (c_k * feedback)
        for integrator_index in range(1, self.modulator_order):
            # Get the CURRENT output of the previous integrator
            # (computed in this same time step - no delay between stages)
            previous_integrator_output: float = self.integrators[
                integrator_index - 1
            ].get_state()

            # Compute input to this integrator stage
            # Subtract weighted feedback from previous stage output
            current_stage_input: float = (
                previous_integrator_output
                - self.feedback_coefficients[integrator_index] * feedback_value
            )

            # Integrate:  accumulator += input
            self.integrators[integrator_index].integrate(current_stage_input)

        # =================================================================
        # STEP 4: QUANTIZE THE FINAL INTEGRATOR OUTPUT
        # =================================================================
        # The quantizer input is the output of the LAST integrator
        final_integrator_output: float = self.integrators[-1].get_state()

        # For binary quantizer:  +1 if >= 0, else -1
        output_sample: float = self.quantizer.quantize(final_integrator_output)

        # =================================================================
        # STEP 5: STORE OUTPUT FOR NEXT SAMPLE'S FEEDBACK
        # =================================================================
        # This implements the z^-1 delay in the feedback path
        # On the next call, this value will be used as feedback
        self.previous_output = output_sample

        return output_sample

    def process_signal(
        self,
        input_signal: np.ndarray,
        store_integrator_history: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Process an entire input signal through the modulator.

        This method processes each sample sequentially, matching
        how the FPGA will operate (sample-by-sample at clock rate).

        Args:
            input_signal: Array of input samples.
            store_integrator_history: If True, stores all integrator states.

        Returns:
            Tuple containing:
            - output_signal: Array of quantizer outputs
            - integrator_history: Array of shape (samples, order) or None
        """
        number_of_samples: int = len(input_signal)

        # Allocate output array
        output_signal: np.ndarray = np.zeros(number_of_samples)

        # Optionally allocate integrator history storage
        integrator_history: Optional[np.ndarray] = None
        if store_integrator_history: 
            integrator_history = np.zeros(
                (number_of_samples, self.modulator_order)
            )

        # Process each sample sequentially
        for sample_index in range(number_of_samples):
            # Process this sample through the modulator
            output_signal[sample_index] = self.process_sample(
                input_signal[sample_index]
            )

            # Store integrator states if requested (for stability analysis)
            if store_integrator_history:
                for integrator_index in range(self.modulator_order):
                    integrator_history[sample_index, integrator_index] = (
                        self.integrators[integrator_index].get_state()
                    )

        return output_signal, integrator_history

    def reset(self) -> None:
        """
        Reset the modulator to initial state.

        Clears all integrators and the feedback register.
        Call before processing a new signal. 

        FPGA Equivalent:  Assert reset signal
        """
        for integrator in self.integrators:
            integrator.reset()
        self.previous_output = 0.0

    def get_integrator_states(self) -> List[float]:
        """Get current states of all integrators."""
        return [integrator.get_state() for integrator in self.integrators]

    def check_stability(self, maximum_allowed_magnitude: float = 10.0) -> bool:
        """
        Check if the modulator is currently stable.

        Args:
            maximum_allowed_magnitude:  Threshold for instability detection. 

        Returns:
            bool: True if stable, False if any integrator exceeds limit.
        """
        for integrator in self.integrators:
            if abs(integrator.get_state()) > maximum_allowed_magnitude: 
                return False
        return True

    def get_noise_transfer_function_order(self) -> int:
        """
        Return the order of the noise transfer function. 

        For standard CIFB with all coefficients = 1:
        NTF(z) = (1 - z^-1)^N where N = modulator_order
        """
        return self.modulator_order