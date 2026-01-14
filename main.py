"""
Delta-Sigma DAC Simulation - Main Entry Point
==============================================

This is the main entry point for the delta-sigma DAC simulation project. 

This file demonstrates how to use all components of the simulation package
to evaluate delta-sigma modulator parameters for FPGA implementation. 

The simulation workflow is:
1. Configure simulation parameters
2. Generate test signal (sinusoidal)
3. Process through delta-sigma modulator
4. Reconstruct signal with low-pass filter
5. Calculate performance metrics (SNR, ENOB)
6. Visualize results
7. Estimate FPGA resources

Usage:
    python main.py

Or import and use programmatically:
    from delta_sigma_dac.main import run_single_simulation, run_comparison_study
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

# ============================================================================
# IMPORT ALL COMPONENTS
# ============================================================================

# Signal generation
from signals.digital_signal_generator import DigitalSignalGenerator
from signals.signal_container import SignalContainer

# Modulator components
from modulator.quantizer import BinaryQuantizer
from modulator.feedback_digital_to_analog_converter import FeedbackDigitalToAnalogConverter
from modulator.delta_sigma_modulator import DeltaSigmaModulator

# Reconstruction filters
from reconstruction. cascaded_rc_low_pass_filter import (
    CascadedRCLowPassFilter,
    create_filter_for_modulator_order
)
from reconstruction.ideal_low_pass_filter import IdealLowPassFilter

# Metrics
from metrics.signal_to_noise_ratio import (
    compute_signal_to_noise_ratio_time_domain,
    compute_snr_with_diagnostics,
    compute_in_band_snr,
    compute_sinad_in_band
)
from metrics.effective_number_of_bits import (
    compute_effective_number_of_bits,
    compute_theoretical_enob_for_delta_sigma,
    print_enob_table
)
from metrics.fpga_metrics import FPGAMetricsCalculator

# Visualization
from visualization.delta_sigma_plotter import DeltaSigmaPlotter


# ============================================================================
# SIMULATION FUNCTIONS
# ============================================================================

def run_single_simulation(
    modulator_order: int = 2,
    oversampling_ratio: int = 256,
    signal_frequency_hz:  float = 1000.0,
    signal_amplitude: float = 0.5,
    number_of_samples: int = 16384,
    input_word_length_bits: int = 16,
    dither_std: float = 0.0,
    dither_seed: Optional[int] = None,
    filter_cutoff_frequency_hz: Optional[float] = None,
    use_ideal_filter: bool = False,
    integrator_saturation_limit: Optional[float] = 4.0,
    plot_results: bool = True,
    verbose: bool = True
) -> Dict: 
    """
    Run a complete delta-sigma DAC simulation with specified parameters.

    This function orchestrates the entire simulation pipeline: 
    1. Creates the input signal
    2. Configures and runs the delta-sigma modulator
    3. Applies the reconstruction filter
    4. Calculates all performance metrics
    5. Optionally plots the results

    Args:
        modulator_order:  Order of the delta-sigma modulator (1-5).
            Higher order = better noise shaping but stability concerns.
            Recommended: Start with order 2.

        oversampling_ratio: The OSR value. 
            Higher OSR = better SNR but higher clock rate needed.
            For 10 kHz signal with 200 MHz FPGA:  max OSR = 10000. 
            Recommended: 256 to 1024.

        signal_frequency_hz: Frequency of test sine wave in Hz.
            Your application range: 1 kHz to 10 kHz.

        signal_amplitude: Peak amplitude of test signal (0 to 1).
            IMPORTANT for stability: 
            - Order 1-2: Use 0.5 to 0.7
            - Order 3: Use 0.4 to 0.5
            - Order 4-5: Use 0.3 to 0.4

        number_of_samples: Total samples to simulate. 
            More samples = better frequency resolution in analysis. 
            Recommended: 8192 to 32768 (powers of 2 for efficient FFT).

        input_word_length_bits: Bit depth of input PCM signal.
            Typical values: 16 (CD quality) or 24 (professional).

        filter_cutoff_frequency_hz: Cutoff for reconstruction filter.
            If None, auto-set to 1.5x signal frequency. 
            Should be above signal frequency but below noise band.

        use_ideal_filter: If True, use ideal brick-wall filter.
            False (default) uses realistic first-order RC filter.

        integrator_saturation_limit: Limits integrator magnitude.
            Helps prevent instability.  Set to None to disable. 
            Recommended: 4.0 for most cases.

        plot_results: If True, display plots of results. 

        verbose: If True, print progress and results.

    Returns:
        Dict containing all simulation results: 
        - 'signals': SignalContainer with all signals
        - 'snr_db':  Measured SNR in dB
        - 'enob':  Effective number of bits
        - 'enob_theoretical': Theoretical ENOB for comparison
        - 'is_stable': Boolean indicating stability
        - 'switching_metrics': Power-related metrics
        - 'fpga_resources':  Estimated FPGA resource usage
        - 'configuration': All input parameters
    """

    # ========================================================================
    # STEP 0: CALCULATE DERIVED PARAMETERS
    # ========================================================================
    
    # Nyquist frequency is 2x the signal frequency
    nyquist_frequency_hz:  float = 2.0 * signal_frequency_hz
    
    # Sampling frequency = Nyquist * OSR
    # This is the clock rate at which the modulator operates
    sampling_frequency_hz:  float = nyquist_frequency_hz * oversampling_ratio
    
    # Auto-calculate filter cutoff if not specified
    if filter_cutoff_frequency_hz is None:
        # Set cutoff to 1.5x signal frequency
        # This passes the signal while attenuating noise
        filter_cutoff_frequency_hz = signal_frequency_hz * 1.5
    
    if verbose:
        print("\n" + "=" * 70)
        print("DELTA-SIGMA DAC SIMULATION")
        print("=" * 70)
        print(f"\n--- Configuration ---")
        print(f"  Modulator Order:         {modulator_order}")
        print(f"  Oversampling Ratio:     {oversampling_ratio}")
        print(f"  Signal Frequency:       {signal_frequency_hz / 1000:.1f} kHz")
        print(f"  Signal Amplitude:       {signal_amplitude}")
        print(f"  Sampling Frequency:     {sampling_frequency_hz / 1e6:.3f} MHz")
        print(f"  Filter Cutoff:          {filter_cutoff_frequency_hz / 1000:.1f} kHz")
        print(f"  Number of Samples:      {number_of_samples}")
        print(f"  Input Word Length:      {input_word_length_bits} bits")

    # ========================================================================
    # STEP 1: GENERATE INPUT SIGNAL
    # ========================================================================
    
    if verbose:
        print(f"\n--- Step 1: Generating Input Signal ---")
    
    # Create the signal generator
    # This generates a quantized sinusoidal signal representing digital PCM input
    signal_generator:  DigitalSignalGenerator = DigitalSignalGenerator(
        sampling_frequency_hz=sampling_frequency_hz,
        number_of_samples=number_of_samples,
        word_length_bits=input_word_length_bits
    )
    
    # Generate the sinusoidal test signal
    input_signal: np.ndarray = signal_generator.generate_sinusoidal_signal(
        signal_frequency_hz=signal_frequency_hz,
        amplitude=signal_amplitude,
        phase_radians=0.0,  # Start at zero crossing
        direct_current_offset=0.0  # No DC offset
    )

    # Optionally add small Gaussian dither to the input signal
    if dither_std and dither_std > 0.0:
        rng = np.random.default_rng(dither_seed)
        dither = rng.normal(0.0, float(dither_std), size=input_signal.shape)
        input_signal = input_signal + dither
        if verbose:
            print(f"  Added Gaussian dither to input (std={dither_std})")
    
    # Get the time axis for plotting
    time_axis: np.ndarray = signal_generator.get_time_axis()
    
    if verbose: 
        signal_params = signal_generator.get_signal_parameters_summary(signal_frequency_hz)
        print(f"  Samples per period:      {signal_params['samples_per_period']:.1f}")
        print(f"  Complete periods:       {signal_params['number_of_complete_periods']:.1f}")
        print(f"  Input signal generated successfully")

    # ========================================================================
    # STEP 2: CREATE AND CONFIGURE THE DELTA-SIGMA MODULATOR
    # ========================================================================
    
    if verbose: 
        print(f"\n--- Step 2: Configuring Delta-Sigma Modulator ---")
    
    # Create the binary (1-bit) quantizer
    # This outputs +1 or -1 based on the sign of the input
    quantizer:  BinaryQuantizer = BinaryQuantizer(
        threshold=0.0,              # Decision threshold at zero
        positive_output_level=1.0,  # Output +1 for positive input
        negative_output_level=-1.0  # Output -1 for negative input
    )
    
    # Create the feedback DAC (ideal for simulation)
    # In a real FPGA, this is just a wire (digital feedback)
    feedback_dac: FeedbackDigitalToAnalogConverter = FeedbackDigitalToAnalogConverter(
        gain=1.0,    # Unity gain (ideal)
        offset=0.0   # No offset (ideal)
    )
    
    # Create the delta-sigma modulator
    # This is the core of the simulation
    modulator: DeltaSigmaModulator = DeltaSigmaModulator(
        modulator_order=modulator_order,
        quantizer=quantizer,
        feedback_dac=feedback_dac,
        feedback_coefficients=None,  # Use default [1.0, 1.0, ... ] for standard CIFB
        integrator_saturation_limit=integrator_saturation_limit
    )
    
    if verbose:
        print(f"  Quantizer:              1-bit (binary)")
        print(f"  Topology:               CIFB (Cascade of Integrators with Feedback)")
        print(f"  Saturation Limit:       {integrator_saturation_limit}")

    # ========================================================================
    # STEP 3: PROCESS SIGNAL THROUGH MODULATOR
    # ========================================================================
    
    if verbose:
        print(f"\n--- Step 3: Running Delta-Sigma Modulation ---")
    
    # Reset modulator to ensure clean start
    modulator.reset()
    
    # Process the entire input signal
    # This returns the 1-bit output bitstream and optionally the integrator history
    modulator_output, integrator_history = modulator.process_signal(
        input_signal=input_signal,
        store_integrator_history=True  # Store for stability analysis
    )
    
    # Check stability by examining integrator states
    is_stable:  bool = modulator.check_stability(maximum_allowed_magnitude=10.0)
    max_integrator_magnitude: float = 0.0
    if integrator_history is not None:
        max_integrator_magnitude = float(np.max(np.abs(integrator_history)))
    
    if verbose:
        print(f"  Processing complete")
        print(f"  Modulator stable:        {'Yes' if is_stable else 'NO - UNSTABLE!'}")
        print(f"  Max integrator value:   {max_integrator_magnitude:.2f}")
        
        if not is_stable:
            print(f"  WARNING: Modulator became unstable!  Try reducing input amplitude.")

    # ========================================================================
    # STEP 4: RECONSTRUCT ANALOG SIGNAL WITH LOW-PASS FILTER
    # ========================================================================
    
    if verbose:
        print(f"\n--- Step 4: Reconstructing Analog Signal ---")
    
    # Choose filter type based on configuration
    if use_ideal_filter:
        # Ideal brick-wall filter (for theoretical analysis)
        reconstruction_filter = IdealLowPassFilter(
            cutoff_frequency_hz=filter_cutoff_frequency_hz,
            sampling_frequency_hz=sampling_frequency_hz
        )
        reconstructed_signal:  np.ndarray = reconstruction_filter.filter_signal(modulator_output)
        filter_type = "Ideal (brick-wall)"
    else:
        # Use cascaded RC filter matched to modulator order
        # This ensures proper noise rejection for any DSM order
        reconstruction_filter = create_filter_for_modulator_order(
            modulator_order=modulator_order,
            cutoff_frequency_hz=filter_cutoff_frequency_hz,
            sampling_frequency_hz=sampling_frequency_hz,
            # extra_stages=0  # Set to 1 for extra margin
        )
        # Reset filter state
        reconstruction_filter.reset()
        reconstructed_signal: np.ndarray = reconstruction_filter.filter_signal(modulator_output)
        filter_type = f"Cascaded RC ({modulator_order} stages, {20*modulator_order} dB/decade)"
    
    if verbose:
        print(f"  Filter type:             {filter_type}")
        print(f"  Cutoff frequency:       {filter_cutoff_frequency_hz / 1000:.1f} kHz")
        print(f"  Reconstruction complete")

    # ========================================================================
    # STEP 5: CALCULATE PERFORMANCE METRICS
    # ========================================================================
    
    if verbose:
        print(f"\n--- Step 5: Calculating Performance Metrics ---")
    
    # Remove transient samples from the beginning for accurate metrics
    # The modulator needs time to settle; remove first few periods
    transient_samples: int = int(sampling_frequency_hz / signal_frequency_hz * 5)  # 5 periods
    transient_samples = min(transient_samples, number_of_samples // 4)  # At most 25% of signal
    
    # Signals without transient
    input_steady:  np.ndarray = input_signal[transient_samples:]
    reconstructed_steady: np.ndarray = reconstructed_signal[transient_samples:]
    modulator_output_steady: np.ndarray = modulator_output[transient_samples:]
    
    # Calculate SNR in time domain (comparing reconstructed to input)
    snr_time_domain_db = compute_signal_to_noise_ratio_time_domain(
        reconstructed_signal=reconstructed_steady,
        reference_signal=input_steady,
        signal_frequency_hz=signal_frequency_hz,
        sampling_frequency_hz=sampling_frequency_hz
    )    
    # snr_time_domain_db: float = compute_signal_to_noise_ratio_time_domain(
    #     reconstructed_signal=reconstructed_steady,
    #     reference_signal=input_steady
    # )
    
    # Calculate SINAD in-band
    sinad_in_band_db, sinad_enob_in_band_db = compute_sinad_in_band(
        signal=reconstructed_steady,
        sampling_frequency_hz=sampling_frequency_hz,
        signal_frequency_hz=signal_frequency_hz,
        signal_bandwidth_hz=filter_cutoff_frequency_hz
    )

    # Robust time-domain least-squares SINAD (canonical)
    time_steady = time_axis[transient_samples:]
    try:
        a = np.column_stack([
            np.cos(2.0 * np.pi * signal_frequency_hz * time_steady),
            np.sin(2.0 * np.pi * signal_frequency_hz * time_steady)
        ])
        coeffs, *_ = np.linalg.lstsq(a, reconstructed_steady, rcond=None)
        fund = a.dot(coeffs)
        resid = reconstructed_steady - fund
        fund_power = float(np.mean(fund ** 2))
        resid_power = float(np.mean(resid ** 2))
        sinad_time_domain_db = 200.0 if resid_power < 1e-30 or fund_power <= 0.0 else 10.0 * np.log10(fund_power / resid_power)
    except Exception:
        sinad_time_domain_db = float('nan')

    # Compute input quantization SNR as an upper bound for SINAD/ENOB
    try:
        ideal = signal_amplitude * np.sin(2.0 * np.pi * signal_frequency_hz * time_axis)
        ideal_steady = ideal[transient_samples:]
        err_q = ideal_steady - input_steady
        sig_power_q = float(np.mean(ideal_steady ** 2))
        err_power_q = float(np.mean(err_q ** 2))
        if err_power_q < 1e-30:
            input_quant_snr_db = 200.0
        else:
            input_quant_snr_db = 10.0 * np.log10(sig_power_q / err_power_q)
    except Exception:
        input_quant_snr_db = float('nan')

    # Clamp LS SINAD to input quantization SNR (conservative canonical value)
    if not np.isnan(sinad_time_domain_db) and not np.isnan(input_quant_snr_db):
        sinad_time_domain_clamped_db = min(sinad_time_domain_db, input_quant_snr_db)
    else:
        sinad_time_domain_clamped_db = sinad_time_domain_db

    # ENOB from canonical (clamped LS) SINAD
    enob_from_sinad_canonical = compute_effective_number_of_bits(sinad_time_domain_clamped_db)

    # Calculate in-band SNR (most relevant metric for delta-sigma)
    snr_in_band_db: float = compute_in_band_snr(
        signal=modulator_output_steady,
        sampling_frequency_hz=sampling_frequency_hz,
        signal_frequency_hz=signal_frequency_hz,
        signal_bandwidth_hz=filter_cutoff_frequency_hz
    )
    
    # Calculate ENOB from measured SNR
    enob_measured: float = compute_effective_number_of_bits(snr_in_band_db)
    # enob_measured: float = compute_effective_number_of_bits(snr_time_domain_db)
    
    # Calculate theoretical ENOB for comparison
    enob_theoretical:  float = compute_theoretical_enob_for_delta_sigma(
        modulator_order=modulator_order,
        oversampling_ratio=oversampling_ratio,
        quantizer_bits=1
    )
    
    if verbose:
        print(f"  Transient samples:      {transient_samples} (removed)")
        # print(f"  SNR (time domain):      {snr_time_domain_db:.1f} dB")
        print(f"  SNR (in-band):          {snr_in_band_db:.1f} dB")
        print(f"  ENOB (from SNR):        {enob_measured:.1f} bits")
        # print(f"  ENOB (theoretical):     {enob_theoretical:.1f} bits")
        print(f"  SINAD (in-band FFT):    {sinad_in_band_db:.1f} dB")
        print(f"  ENOB (in-band FFT):     {sinad_enob_in_band_db:.1f} bits")
        print(f"  SINAD (LS time-domain): {sinad_time_domain_db:.3f} dB")
        print(f"  SINAD (LS clamped):     {sinad_time_domain_clamped_db:.3f} dB")
        print(f"  ENOB (canonical):       {enob_from_sinad_canonical:.3f} bits")

    # ========================================================================
    # STEP 6: CALCULATE FPGA-SPECIFIC METRICS
    # ========================================================================
    
    if verbose:
        print(f"\n--- Step 6: FPGA Implementation Metrics ---")
    
    # Create FPGA metrics calculator
    fpga_calculator:  FPGAMetricsCalculator = FPGAMetricsCalculator(
        modulator_order=modulator_order,
        oversampling_ratio=oversampling_ratio,
        signal_bandwidth_hz=signal_frequency_hz,
        fpga_clock_hz=200_000_000  # 200 MHz FPGA clock
    )
    
    # Calculate switching activity (related to power consumption)
    switching_metrics: Dict = fpga_calculator.calculate_switching_activity(modulator_output)
    
    # Estimate FPGA resources
    fpga_resources = fpga_calculator.estimate_resources(
        input_word_length_bits=input_word_length_bits,
        accumulator_guard_bits=8
    )
    
    # Get timing requirements
    timing_requirements: Dict = fpga_calculator.calculate_timing_requirements()
    
    if verbose:
        print(f"  Switching rate:         {switching_metrics['switching_rate']:.3f} ({switching_metrics['switching_rate']*100:.1f}%)")
        print(f"  Relative power:         {switching_metrics['estimated_relative_power']:.2f}")
        print(f"  Required sample rate:   {timing_requirements['required_sampling_frequency_hz']/1e6:.3f} MHz")
        print(f"  Timing feasible:        {'Yes' if timing_requirements['is_timing_feasible'] else 'NO!'}")
        print(f"  Est. accumulator bits:   {fpga_resources.accumulator_bit_width}")
        print(f"  Est. LUTs:              ~{fpga_resources.estimated_lut_count}")

    # ========================================================================
    # STEP 7: CREATE SIGNAL CONTAINER AND VISUALIZE
    # ========================================================================
    
    # Package all signals into a container
    signals:  SignalContainer = SignalContainer(
        time_axis_seconds=time_axis,
        input_signal_digital_pcm=input_signal,
        modulator_output_bitstream=modulator_output,
        reconstructed_analog_signal=reconstructed_signal,
        integrator_state_history=integrator_history,
        sampling_frequency_hz=sampling_frequency_hz,
        signal_frequency_hz=signal_frequency_hz,
        oversampling_ratio=oversampling_ratio,
        modulator_order=modulator_order
    )
    
    # Plot results if requested
    if plot_results:
        if verbose:
            print(f"\n--- Step 7: Generating Plots ---")
        
        # Plot time-domain signals (show 5 periods for clarity)
        samples_to_show: int = int(5 * sampling_frequency_hz / signal_frequency_hz)
        samples_to_show = min(samples_to_show, number_of_samples)
        
        DeltaSigmaPlotter.plot_time_domain_signals(
            time_axis_seconds=time_axis,
            input_signal=input_signal,
            modulator_output=modulator_output,
            reconstructed_signal=reconstructed_signal,
            title_prefix=f"Order {modulator_order}, OSR {oversampling_ratio}:  ",
            samples_to_show=samples_to_show
        )
        
        # Plot frequency spectrum
        DeltaSigmaPlotter.plot_frequency_spectrum(
            signal=modulator_output,
            sampling_frequency_hz=sampling_frequency_hz,
            signal_label="Modulator Output Spectrum",
            signal_frequency_hz=signal_frequency_hz,
            cutoff_frequency_hz=filter_cutoff_frequency_hz
        )
        
        # Plot integrator states (for stability analysis)
        if integrator_history is not None: 
            DeltaSigmaPlotter.plot_integrator_states(
                time_axis_seconds=time_axis,
                integrator_history=integrator_history,
                modulator_order=modulator_order,
                samples_to_show=samples_to_show
            )
        
        # Plot performance summary dashboard
        # DeltaSigmaPlotter.plot_performance_summary(
        #     metrics_dict={
        #         'snr_db': snr_time_domain_db,
        #         'enob': enob_measured,
        #         'switching_rate': switching_metrics['switching_rate'],
        #         'modulator_order': modulator_order,
        #         'osr': oversampling_ratio
        #     }
        # )

    # ========================================================================
    # STEP 8: COMPILE AND RETURN RESULTS
    # ========================================================================
    
    results: Dict = {
        'signals': signals,
        'snr_time_domain_db': snr_time_domain_db,
        'snr_in_band_db':  snr_in_band_db,
        'sinad_in_band_db': sinad_in_band_db,
        'sinad_time_domain_db': sinad_time_domain_db,
        'sinad_time_domain_clamped_db': sinad_time_domain_clamped_db,
        'enob_measured': enob_measured,
        'enob_canonical_from_sinad': enob_from_sinad_canonical,
        'enob_theoretical': enob_theoretical,
        'is_stable': is_stable,
        'max_integrator_magnitude': max_integrator_magnitude,
        'switching_metrics': switching_metrics,
        'fpga_resources': fpga_resources,
        'timing_requirements':  timing_requirements,
        'configuration':  {
            'modulator_order':  modulator_order,
            'oversampling_ratio': oversampling_ratio,
            'signal_frequency_hz': signal_frequency_hz,
            'signal_amplitude': signal_amplitude,
            'sampling_frequency_hz': sampling_frequency_hz,
            'filter_cutoff_frequency_hz': filter_cutoff_frequency_hz,
            'number_of_samples': number_of_samples
        }
    }
    
    if verbose:
        print(f"\n" + "=" * 70)
        print("SIMULATION COMPLETE")
        print("=" * 70)
    
    return results


def run_comparison_study(
    orders_to_test: List[int] = [1, 2, 3],
    osr_values_to_test: List[int] = [64, 128, 256, 512],
    signal_frequency_hz: float = 1000.0,
    number_of_samples: int = 16384,
    plot_comparison: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Run a comparison study across different modulator orders and OSR values. 

    This function is useful for determining the optimal configuration for
    your FPGA implementation by comparing performance across configurations.

    Args:
        orders_to_test:  List of modulator orders to test. 
        osr_values_to_test: List of OSR values to test.
        signal_frequency_hz: Test signal frequency. 
        number_of_samples:  Samples per simulation. 
        plot_comparison: If True, plot comparison charts.
        verbose: If True, print progress. 

    Returns:
        Dict containing comparison results for all configurations. 
    """
    
    if verbose:
        print("\n" + "=" * 70)
        print("DELTA-SIGMA DAC COMPARISON STUDY")
        print("=" * 70)
        print(f"\nOrders to test:     {orders_to_test}")
        print(f"OSR values to test: {osr_values_to_test}")
        print(f"Signal frequency:   {signal_frequency_hz / 1000:.1f} kHz")
    
    # Storage for results
    all_results: Dict = {}
    snr_results: Dict[str, List[float]] = {}  # For plotting
    
    # Recommended amplitudes for different orders (for stability)
    amplitude_by_order: Dict[int, float] = {
        1: 0.7,
        2: 0.6,
        3: 0.5,
        4: 0.4,
        5: 0.35
    }
    
    # Run simulations for each configuration
    total_configs: int = len(orders_to_test) * len(osr_values_to_test)
    config_num: int = 0
    
    for order in orders_to_test:
        order_label:  str = f"Order {order}"
        snr_results[order_label] = []
        
        # Get appropriate amplitude for this order
        amplitude:  float = amplitude_by_order.get(order, 0.5)
        
        for osr in osr_values_to_test:
            config_num += 1
            config_key:  str = f"order{order}_osr{osr}"
            
            if verbose:
                print(f"\n--- Configuration {config_num}/{total_configs}:  Order {order}, OSR {osr} ---")
            
            try:
                # Run simulation for this configuration
                result = run_single_simulation(
                    modulator_order=order,
                    oversampling_ratio=osr,
                    signal_frequency_hz=signal_frequency_hz,
                    signal_amplitude=amplitude,
                    number_of_samples=number_of_samples,
                    plot_results=False,  # Don't plot individual results
                    verbose=False
                )
                
                all_results[config_key] = result
                snr_results[order_label].append(result['snr_time_domain_db'])
                
                if verbose:
                    print(f"  SNR: {result['snr_time_domain_db']:.1f} dB, "
                          f"ENOB: {result['enob_measured']:.1f} bits, "
                          f"Stable: {'Yes' if result['is_stable'] else 'No'}")
                    
            except Exception as e:
                if verbose:
                    print(f"  ERROR: {str(e)}")
                snr_results[order_label].append(float('nan'))
    
    # Print summary table
    if verbose:
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        
        # Header
        header = f"{'Order':<10}"
        for osr in osr_values_to_test: 
            header += f"OSR={osr:<8}"
        print(header)
        print("-" * 70)
        
        # Data rows
        for order in orders_to_test:
            row = f"Order {order: <4}"
            for osr in osr_values_to_test: 
                config_key = f"order{order}_osr{osr}"
                if config_key in all_results: 
                    snr = all_results[config_key]['snr_time_domain_db']
                    row += f"{snr: <12.1f}"
                else:
                    row += f"{'N/A':<12}"
            print(row)
    
    # Plot comparison if requested
    if plot_comparison:
        # Plot SNR vs OSR for different orders
        DeltaSigmaPlotter.plot_snr_vs_osr(
            osr_values=osr_values_to_test,
            snr_values=snr_results
        )
        
        # Also collect modulator outputs for noise shaping comparison
        # Use the highest OSR for fair comparison
        max_osr:  int = max(osr_values_to_test)
        noise_comparison_signals: Dict[str, np.ndarray] = {}
        
        for order in orders_to_test:
            config_key = f"order{order}_osr{max_osr}"
            if config_key in all_results:
                signals = all_results[config_key]['signals']
                noise_comparison_signals[f"Order {order}"] = signals.modulator_output_bitstream
        
        if noise_comparison_signals:
            # Get sampling frequency from one of the results
            sample_config = f"order{orders_to_test[0]}_osr{max_osr}"
            sample_fs = all_results[sample_config]['configuration']['sampling_frequency_hz']
            
            DeltaSigmaPlotter.plot_noise_shaping_comparison(
                signals_dict=noise_comparison_signals,
                sampling_frequency_hz=sample_fs,
                signal_bandwidth_hz=signal_frequency_hz * 1.5
            )
    
    return {
        'all_results': all_results,
        'snr_by_order': snr_results,
        'osr_values':  osr_values_to_test,
        'orders_tested': orders_to_test
    }


def print_design_recommendations(
    signal_frequency_hz: float = 10000.0,
    fpga_clock_hz: float = 200_000_000,
    target_enob: float = 12.0
) -> None:
    """
    Print design recommendations based on requirements.

    This function helps you choose appropriate parameters for your
    FPGA implementation based on your signal requirements.

    Args:
        signal_frequency_hz: Maximum signal frequency in Hz.
        fpga_clock_hz: FPGA clock frequency in Hz.
        target_enob:  Desired effective number of bits. 
    """
    
    print("\n" + "=" * 70)
    print("DESIGN RECOMMENDATIONS")
    print("=" * 70)
    
    print(f"\n--- Your Requirements ---")
    print(f"  Max Signal Frequency:    {signal_frequency_hz / 1000:.1f} kHz")
    print(f"  FPGA Clock:             {fpga_clock_hz / 1e6:.1f} MHz")
    print(f"  Target ENOB:            {target_enob:.1f} bits")
    
    # Calculate maximum possible OSR
    nyquist_hz:  float = 2.0 * signal_frequency_hz
    max_osr: int = int(fpga_clock_hz / nyquist_hz)
    
    print(f"\n--- Calculated Parameters ---")
    print(f"  Nyquist Frequency:      {nyquist_hz / 1000:.1f} kHz")
    print(f"  Maximum Possible OSR:   {max_osr}")
    
    # Print theoretical ENOB table
    print(f"\n--- Theoretical ENOB for Different Configurations ---")
    print_enob_table(max_order=5, osr_values=[64, 128, 256, 512, 1024, 2048, 10000])
    
    # Find configurations that meet the target ENOB
    print(f"\n--- Configurations Meeting Target ENOB ({target_enob:.1f} bits) ---")
    
    recommended_configs: List[Tuple[int, int, float]] = []
    
    for order in range(1, 6):
        for osr in [32, 64, 128, 256, 512, 1024, 2048, 4096, 10000]:
            if osr > max_osr:
                continue
            
            theoretical_enob = compute_theoretical_enob_for_delta_sigma(
                modulator_order=order,
                oversampling_ratio=osr,
                quantizer_bits=1
            )
            
            if theoretical_enob >= target_enob: 
                recommended_configs.append((order, osr, theoretical_enob))
    
    # Sort by OSR (lower is better for power) then by order (lower is better for stability)
    recommended_configs.sort(key=lambda x: (x[1], x[0]))
    
    if recommended_configs:
        print(f"\n  {'Order':<8}{'OSR':<10}{'ENOB':<12}{'Notes'}")
        print("  " + "-" * 50)
        
        for order, osr, enob in recommended_configs[: 10]:  # Show top 10
            required_fs = nyquist_hz * osr
            notes = []
            
            if order <= 2:
                notes.append("Stable")
            elif order == 3:
                notes.append("Usually stable")
            else:
                notes.append("Stability risk")
            
            if required_fs > fpga_clock_hz * 0.5:
                notes.append("High clock usage")
            
            print(f"  {order:<8}{osr:<10}{enob:<12.1f}{', '.join(notes)}")
        
        # Best recommendation
        best = recommended_configs[0]
        print(f"\n  RECOMMENDED: Order {best[0]}, OSR {best[1]}")
        print(f"  This achieves {best[2]:.1f} ENOB with lowest power consumption.")
    else:
        print("  No configuration meets the target ENOB with given constraints.")
        print("  Consider:  increasing FPGA clock, reducing target ENOB, or")
        print("  using a higher-order modulator with careful stability design.")
    
    # RC filter recommendation
    print(f"\n--- Reconstruction Filter Recommendation ---")
    recommended_cutoff = signal_frequency_hz * 1.5
    print(f"  Cutoff Frequency:       {recommended_cutoff / 1000:.1f} kHz")
    
    # Calculate RC values
    time_constant = 1.0 / (2.0 * np.pi * recommended_cutoff)
    r_value = 10000  # 10k ohm
    c_value = time_constant / r_value
    
    print(f"  Suggested R value:      {r_value / 1000:.1f} kΩ")
    print(f"  Suggested C value:      {c_value * 1e9:.2f} nF")
    print(f"  Time Constant:          {time_constant * 1e6:.1f} µs")


# ============================================================================
# EXAMPLE USAGE AND MAIN ENTRY POINT
# ============================================================================

def example_basic_simulation():
    """
    Example 1: Basic single simulation. 
    
    This example shows the simplest way to run a delta-sigma DAC simulation. 
    """
    print("\n" + "#" * 70)
    print("# EXAMPLE 1: Basic Single Simulation")
    print("#" * 70)
    
    # Run a simulation with default-ish parameters suitable for your application
    results = run_single_simulation(
        modulator_order=1,           # Second-order modulator (good balance)
        oversampling_ratio=256,      # 256x oversampling
        signal_frequency_hz=1000.0,  # 1 kHz test signal
        signal_amplitude=0.5,        # 50% amplitude (safe for stability)
        number_of_samples=16384,     # 2^14 samples
        plot_results=True,           # Show all plots
        verbose=True                 # Print detailed output
    )
    
    return results


def example_compare_orders():
    """
    Example 2: Compare different modulator orders. 
    
    This example shows how to compare performance across different
    modulator orders to find the best one for your application.
    """
    print("\n" + "#" * 70)
    print("# EXAMPLE 2: Comparing Different Modulator Orders")
    print("#" * 70)
    
    # Compare orders 1, 2, and 3 across several OSR values
    comparison_results = run_comparison_study(
        orders_to_test=[1, 2, 3],
        osr_values_to_test=[64, 128, 256, 512],
        signal_frequency_hz=1000.0,
        number_of_samples=16384,
        plot_comparison=True,
        verbose=True
    )
    
    return comparison_results


def example_fpga_resource_analysis():
    """
    Example 3: Detailed FPGA resource and power analysis.
    
    This example focuses on FPGA-specific metrics that are important
    for your cryogenic low-power application.
    """
    print("\n" + "#" * 70)
    print("# EXAMPLE 3: FPGA Resource and Power Analysis")
    print("#" * 70)
    
    # Run simulation and get detailed FPGA metrics
    results = run_single_simulation(
        modulator_order=2,
        oversampling_ratio=256,
        signal_frequency_hz=5000.0,  # 5 kHz signal (mid-range for your app)
        signal_amplitude=0.5,
        number_of_samples=16384,
        plot_results=False,  # Skip plots for this analysis
        verbose=True
    )
    
    # Print detailed FPGA analysis
    print("\n--- Detailed FPGA Analysis ---")
    
    fpga_calc = FPGAMetricsCalculator(
        modulator_order=2,
        oversampling_ratio=256,
        signal_bandwidth_hz=5000.0,
        fpga_clock_hz=200_000_000
    )
    
    # Print comprehensive summary
    fpga_calc.print_summary(output_bitstream=results['signals'].modulator_output_bitstream)
    
    return results


def example_filter_comparison():
    """
    Example 4: Compare ideal vs realistic filter. 
    
    This example shows the difference between an ideal brick-wall filter
    (theoretical best case) and a realistic first-order RC filter
    (what you'll actually use with your FPGA).
    """
    print("\n" + "#" * 70)
    print("# EXAMPLE 4: Filter Comparison (Ideal vs RC)")
    print("#" * 70)
    
    # Run with ideal filter
    print("\n--- Simulation with Ideal Filter ---")
    results_ideal = run_single_simulation(
        modulator_order=2,
        oversampling_ratio=256,
        signal_frequency_hz=1000.0,
        signal_amplitude=0.5,
        use_ideal_filter=True,
        plot_results=False,
        verbose=True
    )
    
    # Run with realistic RC filter
    print("\n--- Simulation with RC Filter ---")
    results_rc = run_single_simulation(
        modulator_order=2,
        oversampling_ratio=256,
        signal_frequency_hz=1000.0,
        signal_amplitude=0.5,
        use_ideal_filter=False,
        plot_results=False,
        verbose=True
    )
    
    # Compare
    print("\n--- Comparison ---")
    print(f"  Ideal Filter SNR:        {results_ideal['snr_time_domain_db']:.1f} dB")
    print(f"  RC Filter SNR:          {results_rc['snr_time_domain_db']:.1f} dB")
    print(f"  Difference:             {results_ideal['snr_time_domain_db'] - results_rc['snr_time_domain_db']:.1f} dB")
    print(f"\n  The RC filter has lower SNR because it doesn't perfectly")
    print(f"  block out-of-band noise.   This is the realistic performance")
    print(f"  you'll see with your FPGA + external RC circuit.")
    
    return results_ideal, results_rc


def example_high_frequency_signal():
    """
    Example 5: Test with maximum frequency (10 kHz).
    
    This example tests the system with your maximum signal frequency
    to ensure the design works across your entire operating range.
    """
    print("\n" + "#" * 70)
    print("# EXAMPLE 5: Maximum Frequency Test (10 kHz)")
    print("#" * 70)
    
    results = run_single_simulation(
        modulator_order=2,
        oversampling_ratio=2048,
        signal_frequency_hz=10000.0,
        signal_amplitude=0.4,
        number_of_samples=655360,
        # number_of_samples=32768,
        use_ideal_filter=False,
        # dither_std=1e-3,
        dither_seed=42,
        filter_cutoff_frequency_hz=15000.0,
        plot_results=True,
        verbose=True
    )
    
    return results


def run_diagnostic_checks(sim_results: Dict, harmonics: int = 6, transient_periods: int = 5, dither_std: float = 0.0, dither_seed: Optional[int] = None) -> None:
    """
    Run additional diagnostic checks and print raw power numbers used in SINAD.

    Prints:
      - Input quantization SNR (dB)
      - Raw signal / distortion / noise powers used for in-band SINAD

    This helps detect numerical or measurement artifacts when SINAD/ENOB
    appears unrealistically large.
    """
    if not isinstance(sim_results, dict):
        print("No simulation results provided for diagnostics.")
        return

    signals = sim_results.get('signals')
    cfg = sim_results.get('configuration', {})
    if signals is None:
        print("No signals found in results for diagnostics.")
        return

    # Safely extract metadata from SignalContainer-like object
    fs = None
    f_sig = None
    num_samples = None
    try:
        if hasattr(signals, 'sampling_frequency_hz'):
            fs = getattr(signals, 'sampling_frequency_hz')
        if hasattr(signals, 'signal_frequency_hz'):
            f_sig = getattr(signals, 'signal_frequency_hz')
        if hasattr(signals, 'get_number_of_samples') and callable(getattr(signals, 'get_number_of_samples')):
            try:
                num_samples = signals.get_number_of_samples()
            except Exception:
                num_samples = None
        else:
            # Fallback to time axis length if available
            ta = getattr(signals, 'time_axis_seconds', None)
            if ta is not None:
                try:
                    num_samples = len(ta)
                except Exception:
                    num_samples = None
    except Exception:
        fs = None
        f_sig = None
        num_samples = None

    filter_cutoff = cfg.get('filter_cutoff_frequency_hz', None)

    if fs is None or f_sig is None:
        print('Insufficient signal metadata for diagnostics.')
        return

    # Transient removal (same logic as simulator)
    transient_samples = int(fs / f_sig * transient_periods)
    transient_samples = min(transient_samples, num_samples // 4)

    time = signals.time_axis_seconds
    input_q = signals.input_signal_digital_pcm
    reconstructed = signals.reconstructed_analog_signal

    if input_q is None or reconstructed is None:
        print('Missing input or reconstructed signals for diagnostics.')
        return

    # Estimate input amplitude if not in configuration
    amplitude = cfg.get('signal_amplitude', None)
    if amplitude is None:
        amplitude = float((np.max(input_q) - np.min(input_q)) / 2.0)

    # Ideal high-precision sine (for quantization error estimate)
    ideal = amplitude * np.sin(2.0 * np.pi * f_sig * time)

    ideal_steady = ideal[transient_samples:]
    quant_steady = input_q[transient_samples:]

    # Input quantization SNR
    err = ideal_steady - quant_steady
    sig_power = float(np.mean(ideal_steady ** 2))
    err_power = float(np.mean(err ** 2))
    if err_power < 1e-30:
        input_snr_db = 200.0
    else:
        input_snr_db = 10.0 * np.log10(sig_power / err_power)

    print('\n--- Diagnostic: Input Quantization ---')
    print(f'  Input amplitude (est): {amplitude:.6f}')
    print(f'  Input quantization SNR: {input_snr_db:.3f} dB')

    # Compute raw powers used by compute_sinad_in_band (for reconstructed signal)
    rec = reconstructed[transient_samples:]
    # Optionally add small dither to reveal true noise floor
    if dither_std and dither_std > 0.0:
        rng = np.random.default_rng(dither_seed)
        noise = rng.normal(0.0, float(dither_std), size=rec.shape)
        rec = rec + noise
        print(f"\nNote: added Gaussian dither (std={dither_std}) to reconstructed signal for diagnostics.")
    N = len(rec)
    if N < 8:
        print('Not enough steady samples for FFT diagnostics.')
        return

    # Compute powers via masking in the FFT domain then inverse-transform
    # back to time-domain so Parseval/normalization issues are avoided.
    # Use unwindowed FFT here so masked IFFT components map directly to
    # time-domain components with the same units as `rec`.
    spectrum = np.fft.fft(rec)
    half = N // 2
    freq = np.fft.fftfreq(N, d=1.0 / fs)
    freq_res = fs / N
    fund_bin = int(round(f_sig / freq_res)) if f_sig is not None else None

    # Helper to include both positive and negative bins
    def include_bins(bins_list):
        idxs = set()
        for b in bins_list:
            if b is None:
                continue
            # positive frequency index
            if 0 <= b < N:
                idxs.add(b)
            # symmetric negative index
            nb = (-b) % N
            if 0 <= nb < N:
                idxs.add(nb)
        return sorted(idxs)

    # Determine in-band frequency indices
    inband_idxs = [i for i in range(N) if (filter_cutoff is None or abs(freq[i]) <= filter_cutoff)]

    # Signal bins: ±2 bins around fundamental (if fundamental known)
    signal_bins_pos = []
    if fund_bin is not None:
        signal_bins_pos = [b for b in range(fund_bin - 2, fund_bin + 3) if 0 <= b < N]
    signal_idxs = include_bins(signal_bins_pos)

    # Harmonic bins
    harm_idxs = set()
    if fund_bin is not None:
        for h in range(2, harmonics + 1):
            hb = fund_bin * h
            for off in range(-2, 3):
                b = hb + off
                if 0 <= b < N:
                    harm_idxs.add(b)
    harm_idxs = include_bins(sorted(harm_idxs))

    # Mask and reconstruct time-domain components
    mask_signal = np.zeros(N, dtype=complex)
    for i in signal_idxs:
        if filter_cutoff is None or abs(freq[i]) <= filter_cutoff:
            mask_signal[i] = spectrum[i]

    mask_harm = np.zeros(N, dtype=complex)
    for i in harm_idxs:
        if filter_cutoff is None or abs(freq[i]) <= filter_cutoff:
            mask_harm[i] = spectrum[i]

    mask_noise = np.array(spectrum, copy=True)
    for i in set(signal_idxs) | set(harm_idxs):
        mask_noise[i] = 0.0
    # Also zero out bins outside reconstruction band
    for i in range(N):
        if filter_cutoff is not None and abs(freq[i]) > filter_cutoff:
            mask_noise[i] = 0.0
            mask_signal[i] = 0.0
            mask_harm[i] = 0.0

    comp_signal = np.fft.ifft(mask_signal).real
    comp_harm = np.fft.ifft(mask_harm).real
    comp_noise = np.fft.ifft(mask_noise).real

    signal_power = float(np.mean(comp_signal ** 2))
    distortion_power = float(np.mean(comp_harm ** 2))
    noise_power = float(np.mean(comp_noise ** 2))

    denom = noise_power + distortion_power
    sinad_db = 200.0 if denom < 1e-30 or signal_power <= 0.0 else 10.0 * np.log10(signal_power / denom)

    print('\n--- Diagnostic: Reconstructed spectrum powers (in-band, time-domain units) ---')
    print(f'  Signal power (time-domain):    {signal_power:.6e}')
    print(f'  Distortion power (time-domain): {distortion_power:.6e}')
    print(f'  Noise power (time-domain):     {noise_power:.6e}')
    print(f'  Computed SINAD (dB):           {sinad_db:.3f}')

    # Consistency check: total power vs component sum
    try:
        total_power = float(np.mean(rec ** 2))
        comp_sum = signal_power + distortion_power + noise_power
        print('\n--- Power Consistency Check ---')
        print(f'  Total rec power:              {total_power:.6e}')
        print(f'  Sum of components:           {comp_sum:.6e}')
    except Exception:
        pass

    # Time-domain least-squares fit to fundamental (robust SINAD estimate)
    if f_sig is not None:
        try:
            time_steady = time[transient_samples:]
            a = np.column_stack([
                np.cos(2.0 * np.pi * f_sig * time_steady),
                np.sin(2.0 * np.pi * f_sig * time_steady)
            ])
            coeffs, *_ = np.linalg.lstsq(a, rec, rcond=None)
            fund = a.dot(coeffs)
            resid = rec - fund
            fund_power = float(np.mean(fund ** 2))
            resid_power = float(np.mean(resid ** 2))
            sinad_td = 200.0 if resid_power < 1e-30 or fund_power <= 0.0 else 10.0 * np.log10(fund_power / resid_power)

            print('\n--- Time-Domain LS SINAD ---')
            print(f'  Fundamental power (time-domain): {fund_power:.6e}')
            print(f'  Residual power (time-domain):    {resid_power:.6e}')
            print(f'  SINAD (time-domain, dB):         {sinad_td:.3f}')
            print(f'  Input quantization SNR:          {input_snr_db:.3f} dB')
        except Exception:
            print('\nTime-domain LS SINAD calculation failed.')

    # If the simulation results include the canonical/clamped SINAD and ENOB,
    # print them here for convenience so diagnostics show raw+canonical values.
    try:
        if isinstance(sim_results, dict):
            clamped = sim_results.get('sinad_time_domain_clamped_db', None)
            enob_canon = sim_results.get('enob_canonical_from_sinad', None)
            if clamped is not None:
                print('\n--- Canonical (clamped) SINAD/ENOB from simulation ---')
                print(f'  SINAD (clamped, dB):            {clamped:.3f}')
                if enob_canon is not None:
                    print(f'  ENOB (canonical, bits):         {enob_canon:.3f}')
    except Exception:
        pass

    return


if __name__ == "__main__":
    """
    Main entry point when running the script directly.
    
    This runs a series of examples demonstrating the simulation capabilities. 
    You can modify this section to run specific tests for your needs.
    """
    
    print("\n" + "=" * 70)
    print("   DELTA-SIGMA DAC SIMULATION FOR FPGA IMPLEMENTATION")
    print("   Designed for Cryogenic Low-Power Applications")
    print("=" * 70)
    
    for order in [1, 2, 3]: 
        results = run_single_simulation(
            modulator_order=order,
            oversampling_ratio=512,
            signal_frequency_hz=10000.0,
            signal_amplitude=0.5 if order <= 2 else 0.4,
            number_of_samples=16384,
            plot_results=False,
            verbose=False
        )
        print(f"Order {order}: SNR = {results['snr_time_domain_db']:.1f} dB, "
              f"ENOB = {results['enob_measured']:.1f} bits, "
              f"Stable = {results['is_stable']}, "
              f"Max Integrator = {results['max_integrator_magnitude']:.2f}")    
    
    # delta_f = sampling_frequency_hz / number_of_samples
    # At OSR = 512 and fs = 20 MHz, number_of_samples = 32768 delta_f = 610.3515625 Hz -> Width of each FFT bin
    
    # Print design recommendations first
    print_design_recommendations(
        signal_frequency_hz=10000.0,
        fpga_clock_hz=200_000_000,
        target_enob=12.0
    )
    
    # Example 1: Basic simulation
    # results = example_basic_simulation()
    
    # Example 2: Compare different orders (takes longer)
    # comparison = example_compare_orders()
    
    # Example 3: FPGA resource analysis
    # fpga_results = example_fpga_resource_analysis()
    
    # Example 4: Filter comparison
    # ideal_results, rc_results = example_filter_comparison()
    
    # Example 5: High frequency test
    hf_results = example_high_frequency_signal()
    # run_diagnostic_checks(hf_results, harmonics=6, transient_periods=5)
    # # --- Quick diagnostics ---
    # try:
    #     # Extract signals and config from the results
    #     signals = hf_results.get('signals') if isinstance(hf_results, dict) else None
    #     config = hf_results.get('configuration') if isinstance(hf_results, dict) else None

    #     if signals is not None and signals.reconstructed_analog_signal is not None:
    #         sampling_frequency_hz = getattr(signals, 'sampling_frequency_hz', None)
    #         signal_frequency_hz = getattr(signals, 'signal_frequency_hz', None)
    #         number_of_samples = signals.get_number_of_samples()

    #         # Use the same transient-removal logic as the simulator
    #         transient_samples = 0
    #         if sampling_frequency_hz and signal_frequency_hz:
    #             transient_samples = int(sampling_frequency_hz / signal_frequency_hz * 5)
    #             transient_samples = min(transient_samples, number_of_samples // 4)

    #         input_steady = signals.input_signal_digital_pcm[transient_samples:]
    #         reconstructed_steady = signals.reconstructed_analog_signal[transient_samples:]

    #         print('\n--- Quick SNR Diagnostics ---')
    #         snr_db, diag = compute_snr_with_diagnostics(
    #             reconstructed_signal=reconstructed_steady,
    #             reference_signal=input_steady,
    #             signal_frequency_hz=signal_frequency_hz,
    #             sampling_frequency_hz=sampling_frequency_hz,
    #             verbose=True
    #         )
    #         print('\nDiagnostics dict:')
    #         print(diag)

    #         # Compute SINAD and ENOB within the reconstruction band (recommended for ENOB)
    #         try:
    #             filter_cutoff = None
    #             if isinstance(hf_results, dict):
    #                 cfg = hf_results.get('configuration', {})
    #                 filter_cutoff = cfg.get('filter_cutoff_frequency_hz', None)

    #             # Fallback: use 1.5x signal frequency if no cutoff found
    #             if filter_cutoff is None:
    #                 filter_cutoff = signal_frequency_hz * 1.5 if signal_frequency_hz is not None else None

    #             if filter_cutoff is not None:
    #                 sinad_db, enob_bits = compute_sinad_in_band(
    #                     reconstructed_steady,
    #                     sampling_frequency_hz,
    #                     signal_frequency_hz,
    #                     filter_cutoff
    #                 )
    #                 print('\n--- SINAD / ENOB (in-band) ---')
    #                 print(f'  SINAD (dB):            {sinad_db:.2f}')
    #                 print(f'  ENOB (bits):           {enob_bits:.2f}')
    #             else:
    #                 print('\nNo filter cutoff available to compute in-band SINAD.')
    #         except Exception as _d_e:
    #             print(f'Failed to compute SINAD/ENOB: {_d_e}')

    #         # Integrator history diagnostics
    #         integ = signals.integrator_state_history
    #         if integ is not None:
    #             import numpy as _np
    #             max_val = float(_np.max(_np.abs(integ)))
    #             min_val = float(_np.min(integ))
    #             pct_saturated = 0.0
    #             if max_val > 0:
    #                 pct_saturated = float(_np.sum(_np.isclose(_np.abs(integ), max_val, atol=1e-6))) / integ.size * 100.0

    #             print('\n--- Integrator Diagnostics ---')
    #             print(f'  Integrator min: {min_val:.6f}, max abs: {max_val:.6f}')
    #             print(f'  % samples at peak magnitude: {pct_saturated:.3f}%')
    #     else:
    #         print('\nNo reconstructed signal available for diagnostics.')
    # except Exception as _e:
    #     print(f'Failed to run diagnostics: {_e}')

    print("\n" + "=" * 70)
    print("   SIMULATION SESSION COMPLETE")
    print("=" * 70)
    print("\nYou can now:")
    print("  1. Analyze the results dictionary for detailed data")
    print("  2. Modify parameters and re-run simulations")
    print("  3. Use the comparison study to find optimal settings")
    print("  4. Export results for documentation")
    print("\nFor VHDL implementation, refer to the comments in:")
    print("  - delta_sigma_modulator.py (core algorithm)")
    print("  - integrator.py (accumulator implementation)")
    print("  - quantizer.py (comparator implementation)")