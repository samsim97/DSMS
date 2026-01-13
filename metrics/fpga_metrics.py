"""
FPGA-Relevant Metrics Calculator
================================

This module provides calculations for metrics that are specifically
relevant to FPGA implementation of delta-sigma modulators. 

These metrics help you make informed decisions about: 
1. Resource usage (how many LUTs, registers, etc.)
2. Power consumption estimates
3. Clock frequency requirements
4. Bit width requirements for fixed-point implementation

For your cryogenic, low-power application, minimizing: 
- Switching activity (reduces dynamic power)
- Register count (reduces static power)
- Clock frequency (reduces both)

is critical.
"""

import numpy as np
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class FPGAResourceEstimate:
    """
    Estimated FPGA resource usage for a delta-sigma modulator.

    Attributes:
        accumulator_count: Number of accumulator registers needed. 
        accumulator_bit_width:  Bit width of each accumulator.
        total_register_bits: Total flip-flops needed.
        adder_count: Number of adders needed. 
        comparator_count: Number of comparators (for quantizer).
        estimated_lut_count:  Rough LUT estimate.
        estimated_slice_count: Rough slice estimate (for Xilinx).
    """
    accumulator_count:  int
    accumulator_bit_width: int
    total_register_bits: int
    adder_count: int
    comparator_count: int
    estimated_lut_count: int
    estimated_slice_count: int


class FPGAMetricsCalculator:
    """
    Calculator for FPGA implementation metrics. 

    This class helps estimate resource usage and performance metrics
    for implementing a delta-sigma modulator on an FPGA. 

    Usage:
        calculator = FPGAMetricsCalculator(
            modulator_order=2,
            oversampling_ratio=256,
            signal_bandwidth_hz=10000,
            fpga_clock_hz=200_000_000
        )
        resources = calculator.estimate_resources()
        power = calculator.estimate_switching_activity(bitstream)
    """

    def __init__(
        self,
        modulator_order: int,
        oversampling_ratio: int,
        signal_bandwidth_hz: float,
        fpga_clock_hz: float = 200_000_000
    ) -> None:
        """
        Initialize the FPGA metrics calculator.

        Args:
            modulator_order: Order of the delta-sigma modulator.
            oversampling_ratio: The OSR to be used.
            signal_bandwidth_hz: Maximum signal frequency (Hz).
            fpga_clock_hz:  FPGA clock frequency (Hz). Default 200 MHz.
        """
        self.modulator_order: int = modulator_order
        self.oversampling_ratio: int = oversampling_ratio
        self.signal_bandwidth_hz: float = signal_bandwidth_hz
        self.fpga_clock_hz: float = fpga_clock_hz

        # Calculate derived parameters
        self.nyquist_frequency_hz: float = 2.0 * signal_bandwidth_hz
        self.required_sampling_frequency_hz: float = (
            self.nyquist_frequency_hz * oversampling_ratio
        )

    def estimate_resources(
        self,
        input_word_length_bits: int = 16,
        accumulator_guard_bits: int = 8
    ) -> FPGAResourceEstimate:
        """
        Estimate FPGA resource usage for the modulator.

        The main resources needed are:
        1. Accumulators (one per integrator)
        2. Adders/subtractors (for error computation)
        3. Comparator (for 1-bit quantizer)
        4. Output register

        Args:
            input_word_length_bits:  Bit width of input signal.
            accumulator_guard_bits: Extra bits to prevent overflow. 
                Rule of thumb: log2(OSR) + a few bits for safety.

        Returns:
            FPGAResourceEstimate: Estimated resource usage. 

        FPGA Implementation Notes:
            - Each integrator needs one accumulator register
            - Accumulator width = input_bits + guard_bits + log2(expected_gain)
            - For order N, the last integrator can grow by factor ~OSR^N
            - Safe accumulator width ≈ input_bits + N*log2(OSR) + guard_bits
        """
        # Calculate required accumulator bit width
        # Each integrator can accumulate values, so we need extra bits
        # The growth factor depends on OSR and order
        log2_osr: float = np.log2(self.oversampling_ratio)

        # Conservative estimate: each order adds log2(OSR) bits of growth
        # Plus guard bits for safety margin
        accumulator_bit_width: int = (
            input_word_length_bits
            + int(np.ceil(self.modulator_order * log2_osr))
            + accumulator_guard_bits
        )

        # Ensure reasonable width (not excessive)
        accumulator_bit_width = min(accumulator_bit_width, 48)

        # Number of accumulators = number of integrators = modulator order
        accumulator_count: int = self.modulator_order

        # Total register bits for accumulators
        total_accumulator_bits: int = accumulator_count * accumulator_bit_width

        # Additional registers: 
        # - Output register (1 bit for binary quantizer)
        # - Input register (input_word_length_bits)
        # - Feedback register (1 bit)
        additional_register_bits: int = input_word_length_bits + 2

        total_register_bits: int = total_accumulator_bits + additional_register_bits

        # Adder count: 
        # - One subtractor for each integrator input (computing error)
        # - One adder for each integrator (accumulation)
        # Total: 2 * modulator_order
        adder_count: int = 2 * self.modulator_order

        # Comparator:  1 for binary quantizer
        comparator_count: int = 1

        # LUT estimate (rough):
        # - Each adder bit needs ~1 LUT (for Xilinx 6-input LUTs)
        # - Each comparator bit needs ~0.5 LUT
        # - Control logic adds ~10-20 LUTs
        luts_for_adders: int = adder_count * accumulator_bit_width
        luts_for_comparator: int = accumulator_bit_width // 2
        luts_for_control: int = 20
        estimated_lut_count: int = (
            luts_for_adders + luts_for_comparator + luts_for_control
        )

        # Slice estimate (for Xilinx 7-series):
        # - Each slice has 8 FFs and 4 LUTs
        # - Packing efficiency ~60-80%
        slices_for_registers: int = int(np.ceil(total_register_bits / 8 / 0.7))
        slices_for_luts: int = int(np.ceil(estimated_lut_count / 4 / 0.7))
        estimated_slice_count: int = max(slices_for_registers, slices_for_luts)

        return FPGAResourceEstimate(
            accumulator_count=accumulator_count,
            accumulator_bit_width=accumulator_bit_width,
            total_register_bits=total_register_bits,
            adder_count=adder_count,
            comparator_count=comparator_count,
            estimated_lut_count=estimated_lut_count,
            estimated_slice_count=estimated_slice_count
        )

    def calculate_switching_activity(
        self,
        output_bitstream:  np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate switching activity metrics from a bitstream.

        Switching activity directly correlates with dynamic power consumption: 
            P_dynamic ∝ C * V² * f * α

        Where:
            - C = capacitance
            - V = voltage
            - f = frequency
            - α = switching activity (0 to 1)

        Lower switching activity = lower power consumption.

        For a delta-sigma modulator output:
        - Random bitstream: α ≈ 0.5 (worst case)
        - DC input: α approaches 0.5 for small DC, varies for larger DC
        - Sine input: α depends on frequency and amplitude

        Args:
            output_bitstream: The 1-bit modulator output array.

        Returns:
            dict: Switching activity metrics including:
                - switching_rate: Fraction of samples where output changes
                - transitions_per_second:  Absolute transition rate
                - average_run_length: Average consecutive same-value samples
                - estimated_relative_power: Relative power indicator (0-1)
        """
        # Count transitions (where consecutive samples differ)
        transitions:  np.ndarray = np.diff(output_bitstream) != 0
        number_of_transitions: int = int(np.sum(transitions))
        total_samples: int = len(output_bitstream)

        # Switching rate (0 to 1)
        # 0 = never switches, 0.5 = maximum for binary signal
        switching_rate: float = number_of_transitions / (total_samples - 1)

        # Transitions per second
        transitions_per_second: float = (
            switching_rate * self.required_sampling_frequency_hz
        )

        # Average run length (consecutive same values)
        # Longer runs = fewer transitions = lower power
        if number_of_transitions > 0:
            average_run_length: float = total_samples / (number_of_transitions + 1)
        else:
            average_run_length = float(total_samples)

        # Estimated relative power (normalized to worst case)
        # Worst case is switching_rate = 0.5
        # Best case is switching_rate = 0
        estimated_relative_power:  float = switching_rate / 0.5

        return {
            "switching_rate": switching_rate,
            "transitions_per_second": transitions_per_second,
            "average_run_length": average_run_length,
            "estimated_relative_power": estimated_relative_power,
            "total_transitions": number_of_transitions,
            "total_samples": total_samples
        }

    def calculate_timing_requirements(self) -> Dict[str, float]: 
        """
        Calculate timing requirements for FPGA implementation.

        Returns:
            dict: Timing metrics including:
                - required_clock_period_ns:  Minimum clock period
                - samples_per_signal_period:  Samples in one signal cycle
                - clock_cycles_available: Cycles between signal samples
                - pipeline_stages_possible: How many pipeline stages fit
        """
        # Required clock period
        required_clock_period_ns: float = (
            1e9 / self.required_sampling_frequency_hz
        )

        # Actual FPGA clock period
        fpga_clock_period_ns:  float = 1e9 / self.fpga_clock_hz

        # Samples per signal period (at max signal frequency)
        samples_per_signal_period: float = (
            self.required_sampling_frequency_hz / self.signal_bandwidth_hz
        )

        # Clock cycles available per sample
        # (if FPGA clock is faster than required sampling rate)
        if self.fpga_clock_hz > self.required_sampling_frequency_hz:
            clock_cycles_per_sample: float = (
                self.fpga_clock_hz / self.required_sampling_frequency_hz
            )
        else:
            clock_cycles_per_sample = 1.0

        # How many pipeline stages could fit
        # (relevant if you need to meet timing with deep pipelines)
        pipeline_stages_possible: int = int(clock_cycles_per_sample)

        # Check if design is feasible
        is_feasible: bool = self.fpga_clock_hz >= self.required_sampling_frequency_hz

        return {
            "required_sampling_frequency_hz": self.required_sampling_frequency_hz,
            "required_clock_period_ns": required_clock_period_ns,
            "fpga_clock_period_ns":  fpga_clock_period_ns,
            "samples_per_signal_period": samples_per_signal_period,
            "clock_cycles_per_sample": clock_cycles_per_sample,
            "pipeline_stages_possible": pipeline_stages_possible,
            "is_timing_feasible": is_feasible
        }

    def print_summary(self, output_bitstream: np.ndarray | None = None) -> None:
        """
        Print a comprehensive summary of FPGA metrics. 

        Args:
            output_bitstream: Optional bitstream for switching analysis.
        """
        print("\n" + "=" * 70)
        print("FPGA IMPLEMENTATION METRICS SUMMARY")
        print("=" * 70)

        # Configuration
        print("\n--- Configuration ---")
        print(f"  Modulator Order:         {self.modulator_order}")
        print(f"  Oversampling Ratio:     {self.oversampling_ratio}")
        print(f"  Signal Bandwidth:       {self.signal_bandwidth_hz / 1000:.1f} kHz")
        print(f"  FPGA Clock:              {self.fpga_clock_hz / 1e6:.1f} MHz")

        # Timing
        print("\n--- Timing Requirements ---")
        timing = self.calculate_timing_requirements()
        print(f"  Required Sampling Rate: {timing['required_sampling_frequency_hz'] / 1e6:.3f} MHz")
        print(f"  Clock Cycles/Sample:    {timing['clock_cycles_per_sample']:.1f}")
        print(f"  Timing Feasible:        {'Yes' if timing['is_timing_feasible'] else 'NO - Increase FPGA clock!'}")

        # Resources
        print("\n--- Resource Estimates ---")
        resources = self.estimate_resources()
        print(f"  Accumulators:           {resources.accumulator_count}")
        print(f"  Accumulator Width:      {resources.accumulator_bit_width} bits")
        print(f"  Total Registers:        {resources.total_register_bits} bits")
        print(f"  Adders:                 {resources.adder_count}")
        print(f"  Estimated LUTs:         ~{resources.estimated_lut_count}")
        print(f"  Estimated Slices:       ~{resources.estimated_slice_count}")

        # Switching activity (if bitstream provided)
        if output_bitstream is not None:
            print("\n--- Switching Activity (Power) ---")
            switching = self.calculate_switching_activity(output_bitstream)
            print(f"  Switching Rate:          {switching['switching_rate']:.3f} ({switching['switching_rate']*100:.1f}%)")
            print(f"  Transitions/Second:     {switching['transitions_per_second'] / 1e6:.2f} M/s")
            print(f"  Avg Run Length:         {switching['average_run_length']:.1f} samples")
            print(f"  Relative Power:         {switching['estimated_relative_power']:.2f} (1.0 = worst case)")

        print("\n" + "=" * 70)