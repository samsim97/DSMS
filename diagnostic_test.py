"""
Diagnostic script V4 - Properly handles filter attenuation and phase delay. 

The key insight:   RC filter attenuation at signal frequency is NOT noise. 
We need to: 
1. Measure the filter's gain at the signal frequency
2. Scale the reconstructed signal to compensate
3. Find the phase delay and align
4.  THEN calculate SNR
"""

import numpy as np
import matplotlib.pyplot as plt

from signals.digital_signal_generator import DigitalSignalGenerator
from modulator.quantizer import BinaryQuantizer
from modulator.feedback_digital_to_analog_converter import FeedbackDigitalToAnalogConverter
from modulator.delta_sigma_modulator import DeltaSigmaModulator
from reconstruction.cascaded_rc_low_pass_filter import CascadedRCLowPassFilter
from reconstruction.ideal_low_pass_filter import IdealLowPassFilter


def calculate_rc_filter_response(
    number_of_stages: int,
    stage_cutoff_hz: float,
    signal_frequency_hz: float
) -> tuple: 
    """
    Calculate the gain and phase of cascaded RC filter at a specific frequency.
    
    For a single RC stage:
        H(f) = 1 / (1 + j*f/fc)
        |H(f)| = 1 / sqrt(1 + (f/fc)²)
        phase(f) = -arctan(f/fc)
    
    For N stages:
        |H(f)| = 1 / sqrt(1 + (f/fc)²)^N
        phase(f) = -N * arctan(f/fc)
    
    Returns:
        (gain, phase_radians, phase_degrees, group_delay_samples)
    """
    freq_ratio = signal_frequency_hz / stage_cutoff_hz
    
    # Magnitude
    gain = 1.0 / np.sqrt((1.0 + freq_ratio**2) ** number_of_stages)
    
    # Phase  
    phase_per_stage = -np.arctan(freq_ratio)
    total_phase_radians = number_of_stages * phase_per_stage
    total_phase_degrees = np.degrees(total_phase_radians)
    
    return gain, total_phase_radians, total_phase_degrees


def align_signals_by_correlation(
    reference:  np.ndarray,
    test: np.ndarray,
    max_shift: int
) -> tuple:
    """
    Find optimal alignment using normalized cross-correlation.
    Returns (shift, correlation, aligned_test).
    """
    best_shift = 0
    best_corr = -np.inf
    
    ref_norm = reference - np.mean(reference)
    ref_norm = ref_norm / (np.std(ref_norm) + 1e-10)
    
    for shift in range(-max_shift, max_shift + 1):
        if shift < 0:
            ref_seg = ref_norm[: shift]
            test_seg = test[-shift:]
        elif shift > 0:
            ref_seg = ref_norm[shift:]
            test_seg = test[:-shift]
        else:
            ref_seg = ref_norm
            test_seg = test
        
        min_len = min(len(ref_seg), len(test_seg))
        if min_len < 100: 
            continue
            
        test_norm = test_seg[: min_len] - np.mean(test_seg[:min_len])
        test_norm = test_norm / (np.std(test_norm) + 1e-10)
        
        corr = np.mean(ref_seg[: min_len] * test_norm)
        
        if corr > best_corr:
            best_corr = corr
            best_shift = shift
    
    # Apply best shift
    if best_shift < 0:
        aligned_ref = reference[: best_shift]
        aligned_test = test[-best_shift:]
    elif best_shift > 0:
        aligned_ref = reference[best_shift:]
        aligned_test = test[:-best_shift]
    else:
        aligned_ref = reference
        aligned_test = test
    
    min_len = min(len(aligned_ref), len(aligned_test))
    return best_shift, best_corr, aligned_ref[: min_len], aligned_test[:min_len]


def calculate_snr_properly(
    reference: np.ndarray,
    reconstructed: np.ndarray,
    expected_gain: float,
    max_shift_samples: int
) -> dict:
    """
    Calculate SNR with proper handling of gain and phase.
    
    Steps:
    1. Scale reconstructed by 1/expected_gain to compensate for filter attenuation
    2. Find optimal alignment via correlation
    3. Calculate SNR on aligned signals
    """
    # Step 1: Compensate for expected filter attenuation
    reconstructed_compensated = reconstructed / expected_gain
    
    # Step 2: Find optimal alignment
    shift, corr, ref_aligned, rec_aligned = align_signals_by_correlation(
        reference, reconstructed_compensated, max_shift_samples
    )
    
    # Step 3: Fine-tune amplitude (should be close to 1.0 now)
    # Use least-squares to find optimal scale
    scale = np.dot(ref_aligned, rec_aligned) / (np.dot(rec_aligned, rec_aligned) + 1e-10)
    rec_scaled = rec_aligned * scale
    
    # Step 4: Calculate SNR
    error = ref_aligned - rec_scaled
    signal_power = np.mean(ref_aligned ** 2)
    noise_power = np.mean(error ** 2)
    
    snr_db = 10.0 * np.log10(signal_power / (noise_power + 1e-20))
    
    return {
        'snr_db': snr_db,
        'shift_samples': shift,
        'correlation': corr,
        'gain_compensation': 1.0 / expected_gain,
        'fine_scale': scale,
        'total_scale': scale / expected_gain,
        'ref_std': np.std(ref_aligned),
        'rec_std':  np.std(rec_scaled),
        'error_std': np.std(error)
    }


def run_diagnostic_v4(
    modulator_order: int = 2,
    oversampling_ratio:  int = 256,
    signal_frequency_hz: float = 1000.0,
    signal_amplitude: float = 0.5
):
    """Run diagnostic with proper gain and phase compensation."""
    
    print("\n" + "=" * 70)
    print(f"DIAGNOSTIC V4: Order {modulator_order}, OSR {oversampling_ratio}")
    print("=" * 70)

    # Setup
    nyquist_hz = 2.0 * signal_frequency_hz
    sampling_frequency_hz = nyquist_hz * oversampling_ratio
    number_of_samples = 32768  # More samples for better accuracy
    filter_cutoff_hz = signal_frequency_hz * 2.0  # Higher cutoff to reduce attenuation

    samples_per_period = int(sampling_frequency_hz / signal_frequency_hz)
    
    print(f"\n--- Parameters ---")
    print(f"  Signal Frequency:       {signal_frequency_hz} Hz")
    print(f"  Sampling Frequency:     {sampling_frequency_hz / 1e6:.3f} MHz")
    print(f"  Samples per period:      {samples_per_period}")
    print(f"  Filter Cutoff:          {filter_cutoff_hz} Hz")
    print(f"  Total samples:          {number_of_samples}")

    # Generate input signal
    signal_generator = DigitalSignalGenerator(
        sampling_frequency_hz=sampling_frequency_hz,
        number_of_samples=number_of_samples,
        word_length_bits=16
    )

    input_signal = signal_generator.generate_sinusoidal_signal(
        signal_frequency_hz=signal_frequency_hz,
        amplitude=signal_amplitude
    )
    time_axis = signal_generator.get_time_axis()

    # Run modulator with appropriate saturation limit
    saturation_limit = 2.0 + modulator_order * 2.0  # Scale with order
    
    modulator = DeltaSigmaModulator(
        modulator_order=modulator_order,
        quantizer=BinaryQuantizer(),
        feedback_dac=FeedbackDigitalToAnalogConverter(),
        integrator_saturation_limit=saturation_limit
    )

    modulator.reset()
    modulator_output, integrator_history = modulator.process_signal(
        input_signal=input_signal,
        store_integrator_history=True
    )

    max_integrator = np.max(np.abs(integrator_history))
    is_saturating = max_integrator >= saturation_limit * 0.95
    
    print(f"\n--- Modulator ---")
    print(f"  Saturation Limit:       {saturation_limit}")
    print(f"  Max Integrator:         {max_integrator:.2f}")
    print(f"  Saturating:             {'YES - Results degraded!' if is_saturating else 'No'}")

    # Create and analyze cascaded RC filter
    cascaded_filter = CascadedRCLowPassFilter(
        number_of_stages=modulator_order,
        cutoff_frequency_hz=filter_cutoff_hz,
        sampling_frequency_hz=sampling_frequency_hz,
        compensate_for_cascade=True
    )
    
    # Calculate expected filter response at signal frequency
    stage_cutoff = cascaded_filter.stage_cutoff_frequency_hz
    expected_gain, phase_rad, phase_deg = calculate_rc_filter_response(
        number_of_stages=modulator_order,
        stage_cutoff_hz=stage_cutoff,
        signal_frequency_hz=signal_frequency_hz
    )
    
    print(f"\n--- Filter Analysis at Signal Frequency ---")
    print(f"  Number of stages:        {modulator_order}")
    print(f"  Stage cutoff:           {stage_cutoff:.1f} Hz")
    print(f"  Expected gain at {signal_frequency_hz} Hz:  {expected_gain:.4f} ({20*np.log10(expected_gain):.2f} dB)")
    print(f"  Expected phase:         {phase_deg:.1f}°")

    # Apply filters
    ideal_filter = IdealLowPassFilter(
        cutoff_frequency_hz=filter_cutoff_hz,
        sampling_frequency_hz=sampling_frequency_hz
    )
    reconstructed_ideal = ideal_filter.filter_signal(modulator_output)

    cascaded_filter.reset()
    reconstructed_cascaded = cascaded_filter.filter_signal(modulator_output)

    # Remove transient (use more samples)
    transient = number_of_samples // 5
    input_steady = input_signal[transient:]
    ideal_steady = reconstructed_ideal[transient:]
    cascaded_steady = reconstructed_cascaded[transient:]

    # Maximum shift to search (less than half a period)
    max_shift = samples_per_period // 3

    # Calculate SNR for ideal filter
    print(f"\n--- SNR Results ---")
    
    ideal_result = calculate_snr_properly(
        reference=input_steady,
        reconstructed=ideal_steady,
        expected_gain=1.0,  # Ideal filter has unity gain in passband
        max_shift_samples=max_shift
    )
    print(f"\n  [Ideal Filter]")
    print(f"    SNR:             {ideal_result['snr_db']:.1f} dB")
    print(f"    Shift:           {ideal_result['shift_samples']} samples")
    print(f"    Correlation:     {ideal_result['correlation']:.4f}")

    # Calculate SNR for cascaded RC filter
    cascaded_result = calculate_snr_properly(
        reference=input_steady,
        reconstructed=cascaded_steady,
        expected_gain=expected_gain,  # Compensate for known attenuation
        max_shift_samples=max_shift
    )
    print(f"\n  [Cascaded RC Filter]")
    print(f"    SNR:             {cascaded_result['snr_db']:.1f} dB")
    print(f"    Shift:           {cascaded_result['shift_samples']} samples")
    print(f"    Correlation:     {cascaded_result['correlation']:.4f}")
    print(f"    Gain comp:       {cascaded_result['gain_compensation']:.4f}")
    print(f"    Fine scale:      {cascaded_result['fine_scale']:.4f}")
    print(f"    Total scale:     {cascaded_result['total_scale']:.4f}")

    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(
        f'Order {modulator_order}:  Ideal SNR={ideal_result["snr_db"]:.1f} dB, '
        f'RC SNR={cascaded_result["snr_db"]:.1f} dB',
        fontsize=14
    )

    # Show 3 periods
    samples_to_show = min(3 * samples_per_period, len(input_steady) - max_shift)
    time_ms = np.arange(samples_to_show) / sampling_frequency_hz * 1000

    # Apply compensation to cascaded for plotting
    cascaded_compensated = cascaded_steady / expected_gain
    
    # Find shift for plotting
    shift = cascaded_result['shift_samples']
    if shift > 0:
        plot_input = input_steady[shift: shift + samples_to_show]
        plot_cascaded = cascaded_compensated[: samples_to_show]
    elif shift < 0:
        plot_input = input_steady[: samples_to_show]
        plot_cascaded = cascaded_compensated[-shift:-shift + samples_to_show]
    else:
        plot_input = input_steady[: samples_to_show]
        plot_cascaded = cascaded_compensated[: samples_to_show]
    
    # Ensure same length
    min_len = min(len(plot_input), len(plot_cascaded), len(time_ms))
    plot_input = plot_input[:min_len]
    plot_cascaded = plot_cascaded[:min_len]
    time_ms = time_ms[:min_len]

    # Plot 1: Aligned signals
    axes[0].plot(time_ms, plot_input, 'b-', linewidth=2, label='Input')
    axes[0].plot(time_ms, plot_cascaded * cascaded_result['fine_scale'], 'r--', 
                 linewidth=2, alpha=0.8, label='RC Filter (compensated)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Aligned & Gain-Compensated Signals')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Error
    error = plot_input - plot_cascaded * cascaded_result['fine_scale']
    axes[1].plot(time_ms, error, 'r-', linewidth=1)
    axes[1].set_ylabel('Error')
    axes[1].set_title(f'Error Signal (std = {np.std(error):.5f})')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Original vs raw reconstructed (showing attenuation and phase)
    axes[2].plot(time_ms, input_steady[: min_len], 'b-', linewidth=2, label='Input', alpha=0.7)
    axes[2].plot(time_ms, cascaded_steady[:min_len], 'r-', linewidth=2, 
                 label=f'RC Filter (raw, gain={expected_gain:.3f})', alpha=0.7)
    axes[2].set_ylabel('Amplitude')
    axes[2].set_xlabel('Time (ms)')
    axes[2].set_title('Raw Signals (before compensation) - Notice amplitude and phase difference')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {
        'snr_ideal': ideal_result['snr_db'],
        'snr_cascaded': cascaded_result['snr_db'],
        'expected_gain': expected_gain,
        'shift': cascaded_result['shift_samples'],
        'is_saturating': is_saturating,
        'max_integrator': max_integrator
    }


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("# DIAGNOSTIC V4 - PROPER GAIN AND PHASE COMPENSATION")
    print("#" * 70)
    
    results = {}
    for order in [1, 2, 3]: 
        amp = 0.5 if order <= 2 else 0.35
        results[order] = run_diagnostic_v4(
            modulator_order=order,
            oversampling_ratio=256,
            signal_frequency_hz=1000.0,
            signal_amplitude=amp
        )
    
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"{'Order':<8}{'Ideal SNR':<14}{'RC SNR':<14}{'Filter Gain':<14}{'Saturating'}")
    print("-" * 60)
    for order, res in results.items():
        print(f"{order:<8}{res['snr_ideal']:.1f} dB{'':<6}{res['snr_cascaded']:.1f} dB{'':<6}"
              f"{res['expected_gain']:.4f}{'':<6}{'YES' if res['is_saturating'] else 'No'}")
    
    print("\n" + "=" * 70)
    print("NOTES:")
    print("  - RC SNR should be close to Ideal SNR (within 5-10 dB)")
    print("  - If saturating, increase saturation limit or reduce amplitude")
    print("  - Filter gain shows attenuation at signal frequency")
    print("=" * 70)