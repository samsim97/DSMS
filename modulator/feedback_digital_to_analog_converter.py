"""
Feedback Digital-to-Analog Converter Module
============================================

In a delta-sigma modulator, the quantizer output must be converted back
to a form that can be subtracted from the input. 

For a fully digital delta-sigma DAC (FPGA application), the feedback
path is purely digital, so this is essentially a pass-through.
"""


class FeedbackDigitalToAnalogConverter:
    """
    Models the feedback DAC in a delta-sigma modulator. 

    For a fully digital implementation, this is essentially a wire. 
    Kept as separate class for code clarity and future extensibility. 

    Attributes:
        gain (float): Feedback gain (ideally 1.0).
        offset (float): Feedback offset (ideally 0.0).
    """

    def __init__(
        self,
        gain: float = 1.0,
        offset: float = 0.0
    ) -> None:
        """
        Initialize the feedback DAC.

        Args:
            gain: Multiplicative gain. Default 1.0 (ideal).
            offset: Additive offset. Default 0.0 (ideal).
        """
        self.gain: float = gain
        self.offset: float = offset

    def convert(self, digital_value: float) -> float:
        """
        Convert the digital quantizer output for feedback. 

        For ideal DAC:  output = input
        """
        return self.gain * digital_value + self.offset

    def is_ideal(self) -> bool:
        """Check if the DAC is configured as ideal."""
        return self.gain == 1.0 and self.offset == 0.0