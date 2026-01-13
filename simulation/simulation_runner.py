"""
Simulation Runner
=================

This module provides the main simulation orchestration class that
coordinates all components of the delta-sigma DAC simulation. 

The SimulationRunner class handles: 
1. Signal generation
2. Delta-sigma modulation
3. Signal reconstruction
4. Performance metric calculation
5. Results aggregation

This is the main entry point for running simulations.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any

# Import all components
from ..signals.digital_signal_generator import DigitalSignalGenerator
from ..signals.signal_container import SignalContainer
from ..modulator.quantizer import BinaryQuantizer
from ..modulator.feedback_digital_to_analog_converter import FeedbackDigitalToAnalogConverter
from ..modulator.delta_sigma_modulator import DeltaSigmaModulator
from ..reconstruction.cascaded_rc_low_pass_filter import FirstOrderRCLowPassFilter
from ..reconstruction.ideal_low_pass_filter import IdealLowPassFilter
from ..metrics.signal_to_noise_ratio import (
    compute_signal_to_noise_ratio_time_domain,
    compute_in_band_snr
)
from ..metrics.effective_number_of_bits import (
    compute_effective_number_of_bits,
    compute_theoretical_enob_for_delta_sigma
)
from ..metrics.fpga_metrics import FPGAMetricsCalculator


@dataclass
class SimulationConfiguration:
    """
    Configuration parameters for a delta-sigma DAC simulation.

    This dataclass holds all parameters needed to configure and run
    a complete simulation. 

    Attributes:
        modulator_order: Order of the delta-sigma modulator (1-5).
        oversampling_ratio:  OSR value (e.g., 64, 128, 256).
        signal_frequency_hz: Frequency of test signal in Hz.
        signal_amplitude: Amplitude of test signal (0 to 1).
            IMPORTANT: Use lower amplitudes for higher orders: 
            - Order 1-2: 0.5-0.7
            - Order 3: 0.4-0.5
            - Order 4-5: 0.3-0.4
        number_of_samples: Total samples to simulate. 
        input_word_length_bits:  Bit depth of input signal. 
        filter_cutoff_frequency_hz: Reconstruction filter cutoff.  
        fpga_clock_frequency_hz: FPGA clock for resource estimation.
        use_ideal_filter: If True, use ideal brick-wall filter.
        integrator_saturation_limit:  Limit for integrator values. 
        store_integrator_history:  Store states for analysis.
    """
    # Modulator parameters
    modulator_order: int = 2
    oversampling_ratio:  int = 256

    # Signal parameters
    signal_frequency_hz: float = 1000.0
    signal_amplitude: float = 0.5
    number_of_samples: int = 16384

    # Input resolution
    input_word_length_bits:  int = 16

    # Filter parameters
    filter_cutoff_frequency_hz: Optional[float] = None  # Auto-calculated if None

    # FPGA parameters
    fpga_clock_frequency_hz: float = 200_000_000  # 200 MHz

    # Advanced options
    use_ideal_filter:  bool = False
    integrator_saturation_limit: Optional[float] = 4.0
    store_integrator_history: bool = True

    # Derived parameters (calculated in __post_init__)
    nyquist_frequency_hz: float = 0.0
    sampling_frequency_hz:  float = 0.0

    def __post_init__(self) -> None:
        """Validate and auto-calculate parameters after initialization."""
        # Calculate sampling frequency
        # Nyquist theorem: sampling rate must be at least 2x the signal frequency
        self.nyquist_frequency_hz = 2.0 * self.signal_frequency_hz
        
        # Actual sampling frequency = Nyquist * OSR
        # This is the rate at which the delta-sigma modulator operates
        self.sampling_frequency_hz = (
            self.nyquist_frequency_hz * self.oversampling_ratio
        )

        # Auto-calculate filter cutoff if not specified
        # Set to ~1.5x signal frequency for good reconstruction
        # This passes the signal while attenuating quantization noise
        if self.filter_cutoff_frequency_hz is None: 
            self.filter_cutoff_frequency_hz = self.signal_frequency_hz * 1.5

        # Validate all parameters
        self._validate()

    def _validate(self) -> None:
        """Validate configuration parameters."""
        # Check modulator order
        if self.modulator_order < 1:
            raise ValueError("Modulator order must be at least 1")

        if self.modulator_order > 5:
            print(
                f"WARNING: Order {self.modulator_order} modulators are prone to "
                f"instability. Consider using order 1-3 for reliable operation."
            )

        # Check OSR
        if self.oversampling_ratio < 2:
            raise ValueError("OSR must be at least 2")

        # Check amplitude
        if self.signal_amplitude <= 0 or self.signal_amplitude > 1:
            raise ValueError("Signal amplitude must be in (0, 1]")

        # Check if FPGA clock is sufficient
        if self.sampling_frequency_hz > self.fpga_clock_frequency_hz: 
            raise ValueError(
                f"Required sampling rate ({self.sampling_frequency_hz/1e6:.1f} MHz) "
                f"exceeds FPGA clock ({self.fpga_clock_frequency_hz/1e6:.1f} MHz). "
                f"Reduce OSR or increase FPGA clock."
            )

        # Warn about stability for high orders with high amplitude
        if self.modulator_order >= 3 and self.signal_amplitude > 0.5:
            print(
                f"WARNING: Order {self.modulator_order} with amplitude "
                f"{self.signal_amplitude} may be unstable. Consider reducing "
                f"amplitude to {0.6 - 0.1 * (self.modulator_order - 2):.1f} or lower."
            )

        # Check filter cutoff
        if self.filter_cutoff_frequency_hz <= self.signal_frequency_hz:
            print(
                f"WARNING: Filter cutoff ({self.filter_cutoff_frequency_hz} Hz) "
                f"is at or below signal frequency ({self.signal_frequency_hz} Hz). "
                f"This will attenuate your signal!"
            )

    def get_summary_dict(self) -> Dict[str, Any]:
        """Return a dictionary summary of the configuration."""
        return {
            "modulator_order": self.modulator_order,
            "oversampling_ratio": self.oversampling_ratio,
            "signal_frequency_hz": self.signal_frequency_hz,
            "signal_amplitude": self.signal_amplitude,
            "sampling_frequency_hz": self.sampling_frequency_hz,
            "filter_cutoff_frequency_hz":  self.filter_cutoff_frequency_hz,
            "number_of_samples": self.number_of_samples,
            "input_word_length_bits":  self.input_word_length_bits,
            "use_ideal_filter": self.use_ideal_filter
        }


@dataclass
class SimulationResults:
    """
    Container for all simulation results.

    This dataclass holds all outputs from a simulation run, including
    signals, metrics, and analysis results. 

    Attributes:
        configuration: The SimulationConfiguration used for this run.
        signals: SignalContainer with all signal data.
        snr_time_domain_db:  SNR measured in time domain (dB).
        snr_in_band_db: In-band SNR (dB) - most relevant for DSM. 
        enob_measured:  Effective number of bits from measured SNR.
        enob_theoretical: Theoretical ENOB for comparison.
        switching_rate: Output switching rate (0 to 0.5).
        transitions_per_second:  Absolute transition rate.
        relative_power: Power indicator relative to worst case.
        is_stable: Whether the modulator remained stable.
        max_integrator_magnitude: Maximum integrator value seen.
        simulation_completed: Whether simulation finished successfully.
    """
    # Configuration used
    configuration: SimulationConfiguration

    # Signals
    signals: SignalContainer

    # Performance metrics
    snr_time_domain_db: float = 0.0
    snr_in_band_db: float = 0.0
    enob_measured: float = 0.0
    enob_theoretical: float = 0.0

    # FPGA metrics
    switching_rate: float = 0.0
    transitions_per_second: float = 0.0
    relative_power: float = 0.0

    # Stability
    is_stable: bool = True
    max_integrator_magnitude: float = 0.0

    # Timing info
    simulation_completed: bool = False

    def print_summary(self) -> None:
        """
        Print a formatted summary of the simulation results. 

        This method displays all key metrics in a readable format,
        useful for quick analysis and comparison of different runs.
        """
        print("\n" + "=" * 70)
        print("SIMULATION RESULTS SUMMARY")
        print("=" * 70)

        # Configuration section
        print("\n--- Configuration ---")
        print(f"  Modulator Order:          {self.configuration.modulator_order}")
        print(f"  Oversampling Ratio:       {self.configuration.oversampling_ratio}")
        print(f"  Signal Frequency:        {self.configuration.signal_frequency_hz / 1000:.1f} kHz")
        print(f"  Signal Amplitude:        {self.configuration.signal_amplitude}")
        print(f"  Sampling Frequency:      {self.configuration.sampling_frequency_hz / 1e6:.3f} MHz")
        print(f"  Filter Cutoff:           {self.configuration.filter_cutoff_frequency_hz / 1000:.1f} kHz")
        print(f"  Number of Samples:       {self.configuration.number_of_samples}")

        # Performance metrics section
        print("\n--- Performance Metrics ---")
        print(f"  SNR (time domain):       {self.snr_time_domain_db:.1f} dB")
        print(f"  SNR (in-band):           {self.snr_in_band_db:.1f} dB")
        print(f"  ENOB (measured):         {self.enob_measured:.1f} bits")
        print(f"  ENOB (theoretical):      {self.enob_theoretical:.1f} bits")
        
        # ENOB comparison
        enob_difference:  float = self.enob_theoretical - self.enob_measured
        if enob_difference > 2: 
            print(f"  ENOB Gap:                {enob_difference:.1f} bits (check stability/amplitude)")
        else:
            print(f"  ENOB Gap:                {enob_difference:.1f} bits (good)")

        # Stability section
        print("\n--- Stability ---")
        stability_status:  str = "STABLE" if self.is_stable else "UNSTABLE!"
        print(f"  Status:                  {stability_status}")
        print(f"  Max Integrator Value:    {self.max_integrator_magnitude:.2f}")
        
        if not self.is_stable:
            print("  WARNING: Modulator became unstable!")
            print("  Recommendations:")
            print("    - Reduce input amplitude")
            print("    - Use a lower modulator order")
            print("    - Enable/adjust integrator saturation limit")

        # FPGA/Power section
        print("\n--- FPGA / Power Metrics ---")
        print(f"  Switching Rate:          {self.switching_rate:.3f} ({self.switching_rate * 100:.1f}%)")
        print(f"  Transitions/Second:      {self.transitions_per_second / 1e6:.2f} M/s")
        print(f"  Relative Power:          {self.relative_power:.2f} (1.0 = worst case)")
        
        # Power assessment
        if self.relative_power < 0.5:
            power_assessment = "Excellent (low power)"
        elif self.relative_power < 0.7:
            power_assessment = "Good"
        elif self.relative_power < 0.9:
            power_assessment = "Moderate"
        else:
            power_assessment = "High (consider lower amplitude or order)"
        print(f"  Power Assessment:        {power_assessment}")

        # Completion status
        print("\n--- Status ---")
        print(f"  Simulation Completed:    {'Yes' if self.simulation_completed else 'No'}")

        print("\n" + "=" * 70)

    def get_metrics_dict(self) -> Dict[str, Any]:
        """
        Return all metrics as a dictionary.

        Useful for programmatic access, logging, or export to files. 

        Returns:
            Dict containing all simulation metrics.
        """
        return {
            "snr_time_domain_db":  self.snr_time_domain_db,
            "snr_in_band_db": self.snr_in_band_db,
            "enob_measured": self.enob_measured,
            "enob_theoretical": self.enob_theoretical,
            "switching_rate": self.switching_rate,
            "transitions_per_second": self.transitions_per_second,
            "relative_power": self.relative_power,
            "is_stable": self.is_stable,
            "max_integrator_magnitude": self.max_integrator_magnitude,
            "simulation_completed": self.simulation_completed
        }


class SimulationRunner:
    """
    Main simulation orchestrator for delta-sigma DAC evaluation.

    This class coordinates all components of the simulation: 
    1. Signal generation (DigitalSignalGenerator)
    2. Delta-sigma modulation (DeltaSigmaModulator)
    3. Signal reconstruction (LowPassFilter)
    4. Performance metrics (SNR, ENOB, FPGA metrics)

    The SimulationRunner provides a high-level interface for running
    simulations without needing to manually wire up all components. 

    Usage:
        config = SimulationConfiguration(
            modulator_order=2,
            oversampling_ratio=256,
            signal_frequency_hz=1000.0
        )
        runner = SimulationRunner(config)
        results = runner.run()
        results.print_summary()

    Attributes:
        configuration: The SimulationConfiguration for this runner.
        signal_generator: The DigitalSignalGenerator instance.
        modulator: The DeltaSigmaModulator instance. 
        reconstruction_filter: The reconstruction filter instance.
        fpga_metrics_calculator: The FPGAMetricsCalculator instance.
    """

    def __init__(self, configuration: SimulationConfiguration) -> None:
        """
        Initialize the simulation runner with a configuration.

        This constructor creates and configures all necessary components
        based on the provided configuration.

        Args:
            configuration: SimulationConfiguration with all parameters. 
        """
        self.configuration:  SimulationConfiguration = configuration

        # ===== CREATE SIGNAL GENERATOR =====
        self.signal_generator: DigitalSignalGenerator = DigitalSignalGenerator(
            sampling_frequency_hz=configuration.sampling_frequency_hz,
            number_of_samples=configuration.number_of_samples,
            word_length_bits=configuration.input_word_length_bits
        )

        # ===== CREATE MODULATOR COMPONENTS =====
        # Binary quantizer (1-bit output:  +1 or -1)
        self.quantizer: BinaryQuantizer = BinaryQuantizer(
            threshold=0.0,
            positive_output_level=1.0,
            negative_output_level=-1.0
        )

        # Ideal feedback DAC (digital feedback path)
        self.feedback_dac: FeedbackDigitalToAnalogConverter = (
            FeedbackDigitalToAnalogConverter(gain=1.0, offset=0.0)
        )

        # Delta-sigma modulator (core algorithm)
        self.modulator: DeltaSigmaModulator = DeltaSigmaModulator(
            modulator_order=configuration.modulator_order,
            quantizer=self.quantizer,
            feedback_dac=self.feedback_dac,
            feedback_coefficients=None,  # Default CIFB coefficients
            integrator_saturation_limit=configuration.integrator_saturation_limit
        )

        # ===== CREATE RECONSTRUCTION FILTER =====
        if configuration.use_ideal_filter:
            self.reconstruction_filter: IdealLowPassFilter = IdealLowPassFilter(
                cutoff_frequency_hz=configuration.filter_cutoff_frequency_hz,
                sampling_frequency_hz=configuration.sampling_frequency_hz
            )
        else:
            self.reconstruction_filter: FirstOrderRCLowPassFilter = (
                FirstOrderRCLowPassFilter(
                    cutoff_frequency_hz=configuration.filter_cutoff_frequency_hz,
                    sampling_frequency_hz=configuration.sampling_frequency_hz
                )
            )

        # ===== CREATE FPGA METRICS CALCULATOR =====
        self.fpga_metrics_calculator: FPGAMetricsCalculator = FPGAMetricsCalculator(
            modulator_order=configuration.modulator_order,
            oversampling_ratio=configuration.oversampling_ratio,
            signal_bandwidth_hz=configuration.signal_frequency_hz,
            fpga_clock_hz=configuration.fpga_clock_frequency_hz
        )

    def run(self, verbose: bool = True) -> SimulationResults: 
        """
        Execute the complete simulation. 

        This method runs the full simulation pipeline: 
        1. Generate input signal
        2. Process through delta-sigma modulator
        3. Reconstruct with low-pass filter
        4. Calculate all performance metrics
        5. Package results

        Args:
            verbose: If True, print progress messages.

        Returns:
            SimulationResults containing all outputs and metrics.
        """
        config = self.configuration

        if verbose:
            print("\n" + "-" * 50)
            print(f"Running simulation:  Order {config.modulator_order}, "
                  f"OSR {config.oversampling_ratio}")
            print("-" * 50)

        # ===== STEP 1: GENERATE INPUT SIGNAL =====
        if verbose:
            print("  [1/5] Generating input signal...")

        input_signal:  np.ndarray = self.signal_generator.generate_sinusoidal_signal(
            signal_frequency_hz=config.signal_frequency_hz,
            amplitude=config.signal_amplitude,
            phase_radians=0.0,
            direct_current_offset=0.0
        )
        time_axis: np.ndarray = self.signal_generator.get_time_axis()

        # ===== STEP 2: PROCESS THROUGH MODULATOR =====
        if verbose:
            print("  [2/5] Processing through delta-sigma modulator...")

        # Reset modulator to clean state
        self.modulator.reset()

        # Process signal
        modulator_output, integrator_history = self.modulator.process_signal(
            input_signal=input_signal,
            store_integrator_history=config.store_integrator_history
        )

        # Check stability
        is_stable:  bool = self.modulator.check_stability(maximum_allowed_magnitude=10.0)
        max_integrator_magnitude: float = 0.0
        if integrator_history is not None:
            max_integrator_magnitude = float(np.max(np.abs(integrator_history)))

        # ===== STEP 3: RECONSTRUCT SIGNAL =====
        if verbose: 
            print("  [3/5] Reconstructing analog signal...")

        # Reset filter if it has state
        if hasattr(self.reconstruction_filter, 'reset'):
            self.reconstruction_filter.reset()

        reconstructed_signal: np.ndarray = self.reconstruction_filter.filter_signal(
            modulator_output
        )

        # ===== STEP 4: CALCULATE METRICS =====
        if verbose: 
            print("  [4/5] Calculating performance metrics...")

        # Remove transient samples for accurate metrics
        # Use 5 signal periods as transient settling time
        transient_samples:  int = int(
            config.sampling_frequency_hz / config.signal_frequency_hz * 5
        )
        transient_samples = min(transient_samples, config.number_of_samples // 4)

        # Steady-state signals (transient removed)
        input_steady:  np.ndarray = input_signal[transient_samples:]
        reconstructed_steady: np.ndarray = reconstructed_signal[transient_samples:]
        modulator_output_steady: np.ndarray = modulator_output[transient_samples:]

        # SNR in time domain
        snr_time_domain_db: float = compute_signal_to_noise_ratio_time_domain(
            reconstructed_signal=reconstructed_steady,
            reference_signal=input_steady
        )

        # In-band SNR
        snr_in_band_db: float = compute_in_band_snr(
            signal=modulator_output_steady,
            sampling_frequency_hz=config.sampling_frequency_hz,
            signal_frequency_hz=config.signal_frequency_hz,
            signal_bandwidth_hz=config.filter_cutoff_frequency_hz
        )

        # ENOB from measured SNR
        enob_measured: float = compute_effective_number_of_bits(snr_time_domain_db)

        # Theoretical ENOB for comparison
        enob_theoretical:  float = compute_theoretical_enob_for_delta_sigma(
            modulator_order=config.modulator_order,
            oversampling_ratio=config.oversampling_ratio,
            quantizer_bits=1
        )

        # FPGA switching metrics
        switching_metrics: Dict[str, float] = (
            self.fpga_metrics_calculator.calculate_switching_activity(modulator_output)
        )

        # ===== STEP 5: PACKAGE RESULTS =====
        if verbose:
            print("  [5/5] Packaging results...")

        # Create signal container
        signals: SignalContainer = SignalContainer(
            time_axis_seconds=time_axis,
            input_signal_digital_pcm=input_signal,
            modulator_output_bitstream=modulator_output,
            reconstructed_analog_signal=reconstructed_signal,
            integrator_state_history=integrator_history,
            sampling_frequency_hz=config.sampling_frequency_hz,
            signal_frequency_hz=config.signal_frequency_hz,
            oversampling_ratio=config.oversampling_ratio,
            modulator_order=config.modulator_order
        )

        # Create results object
        results:  SimulationResults = SimulationResults(
            configuration=config,
            signals=signals,
            snr_time_domain_db=snr_time_domain_db,
            snr_in_band_db=snr_in_band_db,
            enob_measured=enob_measured,
            enob_theoretical=enob_theoretical,
            switching_rate=switching_metrics['switching_rate'],
            transitions_per_second=switching_metrics['transitions_per_second'],
            relative_power=switching_metrics['estimated_relative_power'],
            is_stable=is_stable,
            max_integrator_magnitude=max_integrator_magnitude,
            simulation_completed=True
        )

        if verbose:
            print(f"  Simulation complete! SNR={snr_time_domain_db:.1f} dB, "
                  f"ENOB={enob_measured:.1f} bits")

        return results

    def run_with_different_amplitude(
        self,
        amplitude:  float,
        verbose: bool = False
    ) -> SimulationResults:
        """
        Run simulation with a different input amplitude.

        Useful for finding the optimal amplitude for stability vs performance.

        Args:
            amplitude: New amplitude to test (0 to 1).
            verbose: If True, print progress. 

        Returns:
            SimulationResults for the new amplitude.
        """
        # Temporarily modify configuration
        original_amplitude:  float = self.configuration.signal_amplitude
        self.configuration.signal_amplitude = amplitude

        # Run simulation
        results:  SimulationResults = self.run(verbose=verbose)

        # Restore original amplitude
        self.configuration.signal_amplitude = original_amplitude

        return results

    def find_optimal_amplitude(
        self,
        amplitude_range: tuple = (0.1, 0.9),
        steps: int = 9,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Find the optimal input amplitude for best SNR while maintaining stability.

        This method tests multiple amplitudes and returns the one that gives
        the best SNR while keeping the modulator stable.

        Args:
            amplitude_range: (min, max) amplitude to test. 
            steps: Number of amplitude values to test.
            verbose: If True, print progress and results.

        Returns:
            Dict with optimal amplitude and results.
        """
        if verbose:
            print("\n" + "=" * 50)
            print("FINDING OPTIMAL AMPLITUDE")
            print("=" * 50)

        amplitudes: np.ndarray = np.linspace(
            amplitude_range[0], amplitude_range[1], steps
        )
        
        results_list: list = []
        best_snr: float = -float('inf')
        best_amplitude: float = amplitudes[0]
        best_result: Optional[SimulationResults] = None

        for amp in amplitudes:
            if verbose:
                print(f"\n  Testing amplitude = {amp:.2f}...")

            result = self.run_with_different_amplitude(amp, verbose=False)
            results_list.append({
                'amplitude': amp,
                'snr_db': result.snr_time_domain_db,
                'enob':  result.enob_measured,
                'is_stable': result.is_stable,
                'max_integrator':  result.max_integrator_magnitude
            })

            if verbose:
                stable_str = "Stable" if result.is_stable else "UNSTABLE"
                print(f"    SNR={result.snr_time_domain_db:.1f} dB, "
                      f"ENOB={result.enob_measured:.1f}, {stable_str}")

            # Update best if stable and better SNR
            if result.is_stable and result.snr_time_domain_db > best_snr:
                best_snr = result.snr_time_domain_db
                best_amplitude = amp
                best_result = result

        if verbose:
            print("\n" + "-" * 50)
            print(f"OPTIMAL AMPLITUDE: {best_amplitude:.2f}")
            print(f"  SNR:   {best_snr:.1f} dB")
            print(f"  ENOB: {best_result.enob_measured:.1f} bits" if best_result else "")
            print("-" * 50)

        return {
            'optimal_amplitude': best_amplitude,
            'optimal_snr_db': best_snr,
            'optimal_result': best_result,
            'all_results': results_list
        }