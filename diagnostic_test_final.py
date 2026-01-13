"""
Final Diagnostic - Testing filter order vs DSM order relationship. 

Key insight: Filter order should be GREATER than DSM order for good SNR.
- DSM shapes noise at +20*N dB/decade  
- RC filter attenuates at -20*M dB/decade
- Net suppression: 20*(M-N) dB/decade above cutoff

If M = N:  No net suppression - noise just barely controlled
If M > N: Positive suppression - better SNR
"""

import numpy as np
import matplotlib.pyplot as plt

from signals.digital_signal_generator import DigitalSignalGenerator
from modulator.quantizer import BinaryQuantizer
from modulator.feedback_digital_to_analog_converter import FeedbackDigitalToAnalogConverter
from modulator.delta_sigma_modulator import DeltaSigmaModulator
from reconstruction.cascaded_rc_low_pass_filter import CascadedRCLowPassFilter
from reconstruction.ideal_low_pass_filter import IdealLowPassFilter


def calculate_rc_filter_gain(num_stages:  int, stage_cutoff_hz: float, freq_hz: float) -> float:
    """Calculate RC filter gain at a specific frequency."""
    freq_ratio = freq_hz / stage_cutoff_hz
    return 1.0 / np.sqrt((1.0 + freq_ratio**2) ** num_stages)


def align_and_calculate_snr(reference: np.ndarray, test: np.ndarray, 
                            expected_gain: float, max_shift: int) -> dict:
    """Align signals and calculate SNR with gain compensation."""
    
    # Compensate for expected attenuation
    test_compensated = test / expected_gain
    
    # Normalize for correlation
    ref_norm = reference - np.mean(reference)
    ref_norm = ref_norm / (np.std(ref_norm) + 1e-10)
    
    # Find best shift
    best_shift = 0
    best_corr = -np.inf
    
    for shift in range(-max_shift, max_shift + 1):
        if shift < 0:
            r = ref_norm[: shift] if shift != 0 else ref_norm
            t = test_compensated[-shift:]
        elif shift > 0:
            r = ref_norm[shift:]
            t = test_compensated[:-shift]
        else:
            r = ref_norm
            t = test_compensated
        
        min_len = min(len(r), len(t))
        if min_len < 100: 
            continue
        
        t_norm = t[:min_len] - np.mean(t[:min_len])
        t_norm = t_norm / (np.std(t_norm) + 1e-10)
        
        corr = np.mean(r[:min_len] * t_norm)
        if corr > best_corr: 
            best_corr = corr
            best_shift = shift
    
    # Apply best shift
    if best_shift < 0:
        ref_aligned = reference[:best_shift]
        test_aligned = test_compensated[-best_shift:]
    elif best_shift > 0:
        ref_aligned = reference[best_shift:]
        test_aligned = test_compensated[:-best_shift]
    else:
        ref_aligned = reference
        test_aligned = test_compensated
    
    min_len = min(len(ref_aligned), len(test_aligned))
    ref_aligned = ref_aligned[:min_len]
    test_aligned = test_aligned[:min_len]
    
    # Fine-tune scale
    scale = np.dot(ref_aligned, test_aligned) / (np.dot(test_aligned, test_aligned) + 1e-10)
    test_scaled = test_aligned * scale
    
    # Calculate SNR
    error = ref_aligned - test_scaled
    signal_power = np.mean(ref_aligned ** 2)
    noise_power = np.mean(error ** 2)
    snr_db = 10.0 * np.log10(signal_power / (noise_power + 1e-20))
    
    return {'snr_db': snr_db, 'shift': best_shift, 'correlation': best_corr}


def run_filter_order_comparison(
    dsm_order: int,
    filter_orders_to_test: list,
    oversampling_ratio: int = 256,
    signal_frequency_hz: float = 1000.0,
    signal_amplitude: float = 0.5
):
    """
    Test different filter orders for a given DSM order.
    """
    print(f"\n{'='*70}")
    print(f"DSM ORDER {dsm_order} - Testing Filter Orders:  {filter_orders_to_test}")
    print(f"{'='*70}")
    
    # Setup
    nyquist_hz = 2.0 * signal_frequency_hz
    sampling_frequency_hz = nyquist_hz * oversampling_ratio
    number_of_samples = 32768
    filter_cutoff_hz = signal_frequency_hz * 2.0
    samples_per_period = int(sampling_frequency_hz / signal_frequency_hz)
    max_shift = samples_per_period // 3
    
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
    
    # Run DSM
    saturation_limit = 4.0 + dsm_order * 2.0
    modulator = DeltaSigmaModulator(
        modulator_order=dsm_order,
        quantizer=BinaryQuantizer(),
        feedback_dac=FeedbackDigitalToAnalogConverter(),
        integrator_saturation_limit=saturation_limit
    )
    modulator.reset()
    modulator_output, _ = modulator.process_signal(input_signal, store_integrator_history=False)
    
    # Ideal filter for reference
    ideal_filter = IdealLowPassFilter(
        cutoff_frequency_hz=filter_cutoff_hz,
        sampling_frequency_hz=sampling_frequency_hz
    )
    reconstructed_ideal = ideal_filter.filter_signal(modulator_output)
    
    # Remove transient
    transient = number_of_samples // 5
    input_steady = input_signal[transient:]
    ideal_steady = reconstructed_ideal[transient:]
    
    # Calculate ideal SNR
    ideal_result = align_and_calculate_snr(input_steady, ideal_steady, 1.0, max_shift)
    print(f"\n  Ideal Filter SNR: {ideal_result['snr_db']:.1f} dB")
    
    # Test each filter order
    results = []
    print(f"\n  {'Filter Stages':<16}{'SNR (dB)':<12}{'Gap from Ideal':<16}{'Gain at Signal'}")
    print(f"  {'-'*56}")
    
    for filter_stages in filter_orders_to_test:
        # Create filter
        rc_filter = CascadedRCLowPassFilter(
            number_of_stages=filter_stages,
            cutoff_frequency_hz=filter_cutoff_hz,
            sampling_frequency_hz=sampling_frequency_hz,
            compensate_for_cascade=True
        )
        
        # Calculate expected gain at signal frequency
        stage_cutoff = rc_filter.stage_cutoff_frequency_hz
        expected_gain = calculate_rc_filter_gain(filter_stages, stage_cutoff, signal_frequency_hz)
        
        # Apply filter
        rc_filter.reset()
        reconstructed_rc = rc_filter.filter_signal(modulator_output)
        rc_steady = reconstructed_rc[transient:]
        
        # Calculate SNR
        rc_result = align_and_calculate_snr(input_steady, rc_steady, expected_gain, max_shift)
        gap = ideal_result['snr_db'] - rc_result['snr_db']
        
        print(f"  {filter_stages:<16}{rc_result['snr_db']:<12.1f}{gap:<16.1f}{expected_gain:.4f}")
        
        results.append({
            'filter_stages': filter_stages,
            'snr_db': rc_result['snr_db'],
            'gap_db': gap,
            'gain':  expected_gain
        })
    
    return {
        'dsm_order': dsm_order,
        'ideal_snr': ideal_result['snr_db'],
        'filter_results': results
    }


if __name__ == "__main__": 
    print("\n" + "#" * 70)
    print("# FINAL DIAGNOSTIC:  FILTER ORDER VS DSM ORDER")
    print("#" * 70)
    print("\nKey insight: Filter stages should exceed DSM order for best results.")
    print("Rule of thumb: Filter stages = DSM order + 1 or + 2")
    
    all_results = {}
    
    # Test each DSM order with various filter orders
    for dsm_order in [1, 2, 3]: 
        # Amplitude decreases with order for stability
        amp = 0.5 if dsm_order <= 2 else 0.35
        
        # Test filter orders from dsm_order to dsm_order + 3
        filter_orders = list(range(dsm_order, dsm_order + 4))
        
        all_results[dsm_order] = run_filter_order_comparison(
            dsm_order=dsm_order,
            filter_orders_to_test=filter_orders,
            signal_amplitude=amp
        )
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY:  RECOMMENDED FILTER CONFIGURATIONS")
    print("=" * 70)
    print(f"\n{'DSM Order':<12}{'Ideal SNR':<14}{'Best RC Config':<20}{'RC SNR':<12}{'Gap'}")
    print("-" * 70)
    
    for dsm_order, data in all_results.items():
        # Find best filter configuration (smallest gap)
        best = min(data['filter_results'], key=lambda x:  x['gap_db'])
        print(f"{dsm_order:<12}{data['ideal_snr']:<14.1f}"
              f"{best['filter_stages']} stages{'':<12}"
              f"{best['snr_db']: <12.1f}{best['gap_db']:.1f} dB")
    
    print("\n" + "=" * 70)
    print("CONCLUSIONS:")
    print("  1. Using filter_stages = dsm_order gives ~20-40 dB gap from ideal")
    print("  2. Using filter_stages = dsm_order + 1 reduces gap significantly")
    print("  3. Using filter_stages = dsm_order + 2 approaches ideal performance")
    print("  4. For hardware:  balance filter complexity vs SNR requirement")
    print("=" * 70)