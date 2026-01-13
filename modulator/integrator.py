"""
Integrator Module
=================

An integrator is a fundamental building block of delta-sigma modulators.
It accumulates the difference between the input and feedback signals. 

Mathematical Description:
    In discrete time:  y[n] = y[n-1] + x[n]

FPGA Implementation:
    process(clk)
    begin
        if rising_edge(clk) then
            if reset = '1' then
                accumulator <= (others => '0');
            else
                accumulator <= accumulator + input_value;
            end if;
        end if;
    end process;
"""


class Integrator:
    """
    A discrete-time integrator (accumulator) for delta-sigma modulation.

    Transfer Function:
        H(z) = 1 / (1 - z^-1)
        Current output = previous output + current input

    Attributes:
        state (float): The current accumulated value. 
        initial_state (float): The value to reset to. 
        saturation_limit (float): Optional limit to prevent runaway.
    """

    def __init__(
        self,
        initial_state: float = 0.0,
        saturation_limit: float | None = None
    ) -> None:
        """
        Initialize the integrator.

        Args:
            initial_state: Starting value for the accumulator.
            saturation_limit: If provided, clips output to this magnitude.
        """
        self.initial_state: float = initial_state
        self.state: float = initial_state
        self.saturation_limit: float | None = saturation_limit

    def integrate(self, input_value: float) -> float:
        """
        Perform one integration step:  state[n] = state[n-1] + input[n]
        """
        self.state = self.state + input_value

        if self.saturation_limit is not None:
            self.state = max(
                -self.saturation_limit,
                min(self.saturation_limit, self.state)
            )

        return self.state

    def get_state(self) -> float:
        """Return the current integrator state."""
        return self.state

    def set_state(self, new_state: float) -> None:
        """Set the integrator state to a specific value."""
        self.state = new_state
        if self.saturation_limit is not None:
            self.state = max(
                -self.saturation_limit,
                min(self.saturation_limit, self.state)
            )

    def reset(self) -> None:
        """Reset the integrator to its initial state."""
        self.state = self.initial_state