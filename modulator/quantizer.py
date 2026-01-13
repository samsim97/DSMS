"""
Quantizer Module
================

This module provides quantizer implementations for the delta-sigma modulator. 

In a delta-sigma modulator, the quantizer converts the continuous-valued
output of the integrator chain into discrete levels. 

For a DAC application with minimal power consumption (cryogenic FPGA),
a 1-bit (binary) quantizer is ideal because:
1.  Simplest implementation (just a comparator)
2. Lowest switching activity
3. No linearity requirements (only two levels)

FPGA Implementation Note:
    A 1-bit quantizer in VHDL is simply: 
    output <= '1' when integrator_output >= 0 else '0';
"""

from abc import ABC, abstractmethod
from typing import Tuple


class AbstractQuantizer(ABC):
    """Abstract base class for quantizers."""

    @abstractmethod
    def quantize(self, input_value: float) -> float:
        """Quantize a single input value."""
        pass

    @abstractmethod
    def get_number_of_levels(self) -> int:
        """Return the number of quantization levels."""
        pass

    @abstractmethod
    def get_output_range(self) -> Tuple[float, float]:
        """Return the (minimum, maximum) output values."""
        pass


class BinaryQuantizer(AbstractQuantizer):
    """
    A 1-bit (binary) quantizer for delta-sigma modulation.

    This is the simplest and most common quantizer for delta-sigma DACs.
    It outputs only two values: +1 or -1.

    Operation:
        If input >= threshold: output = +1
        If input < threshold: output = -1
    """

    def __init__(
        self,
        threshold: float = 0.0,
        positive_output_level: float = 1.0,
        negative_output_level:  float = -1.0
    ) -> None:
        """
        Initialize the binary quantizer.

        Args:
            threshold: Decision threshold. Typically 0 for symmetric operation.
            positive_output_level: Output when input >= threshold. Default +1.
            negative_output_level: Output when input < threshold. Default -1.
        """
        self.threshold: float = threshold
        self.positive_output_level: float = positive_output_level
        self.negative_output_level: float = negative_output_level

    def quantize(self, input_value: float) -> float:
        """
        Quantize input to binary output.

        VHDL Equivalent:
            if integrator_output >= 0 then
                quantizer_output <= '1';
            else
                quantizer_output <= '0';
            end if;
        """
        if input_value >= self.threshold:
            return self.positive_output_level
        else:
            return self.negative_output_level

    def get_number_of_levels(self) -> int:
        """Return 2 for binary quantizer."""
        return 2

    def get_output_range(self) -> Tuple[float, float]:
        """Return the (min, max) output values."""
        return (self.negative_output_level, self.positive_output_level)


class MultiLevelQuantizer(AbstractQuantizer):
    """
    A multi-level (multi-bit) quantizer for delta-sigma modulation. 

    NOT RECOMMENDED for low-power cryogenic application, but included
    for completeness and comparison purposes.
    """

    def __init__(self, number_of_levels: int = 2) -> None:
        """
        Initialize the multi-level quantizer.

        Args:
            number_of_levels: Number of quantization levels (>= 2).
        """
        if number_of_levels < 2:
            raise ValueError(f"Quantizer must have at least 2 levels.")

        self.number_of_levels: int = number_of_levels
        self.step_size: float = 2.0 / (number_of_levels - 1)
        self.output_levels: list[float] = [
            -1.0 + i * self.step_size
            for i in range(number_of_levels)
        ]

    def quantize(self, input_value: float) -> float:
        """Quantize input to the nearest output level."""
        clipped_value: float = max(-1.0, min(1.0, input_value))
        level_index: int = round((clipped_value + 1.0) / self.step_size)
        level_index = max(0, min(self.number_of_levels - 1, level_index))
        return self.output_levels[level_index]

    def get_number_of_levels(self) -> int:
        """Return the number of quantization levels."""
        return self.number_of_levels

    def get_output_range(self) -> Tuple[float, float]:
        """Return the (min, max) output values."""
        return (-1.0, 1.0)