"""
Spectrum Plotting Examples
==========================

This script demonstrates the enhanced spectrum plotting capabilities
for delta-sigma modulator analysis.

Examples include:
1. Generating modulator output with the simulation framework
2. Plotting spectrum in 'normalized' mode (f/f_signal)
3. Plotting spectrum in 'zoom' mode (around signal)
4. Creating combined plot with inset
5. Comparing different OSR values

Usage:
    python examples/plot_spectrum_example.py

Or import and use functions:
    from examples.plot_spectrum_example import example_normalized_plot
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signals.digital_signal_generator import DigitalSignalGenerator
from modulator.quantizer import BinaryQuantizer
from modulator.feedback_digital_to_analog_converter import FeedbackDigitalToAnalogConverter
from modulator.delta_sigma_modulator import DeltaSigmaModulator
from visualization.spectrum_plot import (
    plot_modulator_spectrum,
    plot_spectrum_with_inset
)


def generate_modulator_output(
    signal_frequency_hz: float = 1000.0,
    oversampling_ratio: int = 256,
    modulator_order: int = 2,
    signal_amplitude: float = 0.5,
    number_of_samples: int = 16384
) -> tuple:
    """
    Generate delta-sigma modulator output for plotting examples.
    
    Args:
        signal_frequency_hz: Input signal frequency in Hz.
        oversampling_ratio: Oversampling ratio (OSR).
        modulator_order: Order of the delta-sigma modulator.
        signal_amplitude: Signal amplitude (0 to 1).
        number_of_samples: Number of samples to generate.
    
    Returns:
        Tuple of (modulator_output, sampling_frequency_hz, filter_cutoff_hz)
    """
    # Calculate sampling frequency
    nyquist_frequency_hz = 2.0 * signal_frequency_hz
    sampling_frequency_hz = nyquist_frequency_hz * oversampling_ratio
    
    # Generate input signal
    signal_generator = DigitalSignalGenerator(
        sampling_frequency_hz=sampling_frequency_hz,
        number_of_samples=number_of_samples,
        word_length_bits=16
    )
    
    input_signal = signal_generator.generate_sinusoidal_signal(
        signal_frequency_hz=signal_frequency_hz,
        amplitude=signal_amplitude,
        phase_radians=0.0,
        direct_current_offset=0.0
    )
    
    # Create modulator components
    quantizer = BinaryQuantizer(
        threshold=0.0,
        positive_output_level=1.0,
        negative_output_level=-1.0
    )
    
    feedback_dac = FeedbackDigitalToAnalogConverter(
        gain=1.0,
        offset=0.0
    )
    
    # Create and run modulator
    modulator = DeltaSigmaModulator(
        modulator_order=modulator_order,
        quantizer=quantizer,
        feedback_dac=feedback_dac,
        feedback_coefficients=None,
        integrator_saturation_limit=4.0
    )
    
    modulator.reset()
    modulator_output, _ = modulator.process_signal(
        input_signal=input_signal,
        store_integrator_history=False
    )
    
    # Calculate filter cutoff
    filter_cutoff_hz = signal_frequency_hz * 1.5
    
    return modulator_output, sampling_frequency_hz, filter_cutoff_hz


def example_normalized_plot():
    """
    Example 1: Normalized frequency plot (f/f_signal).
    
    This mode shows the spectrum with frequency normalized by the signal
    frequency, making it easy to see noise shaping relative to the signal.
    """
    print("\n" + "=" * 70)
    print("Example 1: Normalized Frequency Plot")
    print("=" * 70)
    
    # Generate modulator output
    signal_freq = 1000.0  # 1 kHz
    osr = 256
    
    print(f"\nGenerating modulator output...")
    print(f"  Signal frequency: {signal_freq} Hz")
    print(f"  OSR: {osr}")
    
    modulator_output, sampling_freq, filter_cutoff = generate_modulator_output(
        signal_frequency_hz=signal_freq,
        oversampling_ratio=osr,
        modulator_order=2,
        signal_amplitude=0.5,
        number_of_samples=16384
    )
    
    # Plot normalized spectrum
    print("\nPlotting normalized spectrum...")
    fig, ax = plot_modulator_spectrum(
        modulator_output=modulator_output,
        sampling_frequency_hz=sampling_freq,
        signal_frequency_hz=signal_freq,
        filter_cutoff_hz=filter_cutoff,
        mode='normalized',
        use_welch=True,
        show_diagnostics=True
    )
    
    print("\nNormalized plot complete. The x-axis shows frequency as multiples")
    print("of the signal frequency (f/f_signal), making it easy to compare")
    print("different signal frequencies and OSR values.")
    
    return fig, ax


def example_zoom_plot():
    """
    Example 2: Zoom mode plot showing detail around signal.
    
    This mode zooms in to show detail around the signal frequency,
    useful for examining the signal peak and nearby noise floor.
    """
    print("\n" + "=" * 70)
    print("Example 2: Zoom Mode Plot")
    print("=" * 70)
    
    # Generate modulator output
    signal_freq = 5000.0  # 5 kHz
    osr = 512
    
    print(f"\nGenerating modulator output...")
    print(f"  Signal frequency: {signal_freq} Hz")
    print(f"  OSR: {osr}")
    
    modulator_output, sampling_freq, filter_cutoff = generate_modulator_output(
        signal_frequency_hz=signal_freq,
        oversampling_ratio=osr,
        modulator_order=2,
        signal_amplitude=0.5,
        number_of_samples=16384
    )
    
    # Plot zoomed spectrum
    print("\nPlotting zoomed spectrum...")
    zoom_factor = 3.0
    fig, ax = plot_modulator_spectrum(
        modulator_output=modulator_output,
        sampling_frequency_hz=sampling_freq,
        signal_frequency_hz=signal_freq,
        filter_cutoff_hz=filter_cutoff,
        mode='zoom',
        zoom_factor=zoom_factor,
        use_welch=True,
        show_diagnostics=True
    )
    
    print(f"\nZoom plot complete. Showing 0 to {zoom_factor}× signal frequency")
    print(f"({zoom_factor * signal_freq / 1000:.1f} kHz) for detailed view.")
    
    return fig, ax


def example_full_spectrum_plot():
    """
    Example 3: Full Nyquist range plot.
    
    This mode shows the complete spectrum from 0 to fs/2, useful for
    seeing the full noise shaping profile and high-frequency behavior.
    """
    print("\n" + "=" * 70)
    print("Example 3: Full Nyquist Range Plot")
    print("=" * 70)
    
    # Generate modulator output with higher OSR
    signal_freq = 1000.0  # 1 kHz
    osr = 1024
    
    print(f"\nGenerating modulator output...")
    print(f"  Signal frequency: {signal_freq} Hz")
    print(f"  OSR: {osr}")
    
    modulator_output, sampling_freq, filter_cutoff = generate_modulator_output(
        signal_frequency_hz=signal_freq,
        oversampling_ratio=osr,
        modulator_order=3,
        signal_amplitude=0.4,
        number_of_samples=16384
    )
    
    # Plot full spectrum
    print("\nPlotting full Nyquist range...")
    fig, ax = plot_modulator_spectrum(
        modulator_output=modulator_output,
        sampling_frequency_hz=sampling_freq,
        signal_frequency_hz=signal_freq,
        filter_cutoff_hz=filter_cutoff,
        mode='full',
        use_welch=True,
        show_diagnostics=True
    )
    
    print("\nFull spectrum plot complete. Shows 0 to fs/2 (Nyquist frequency).")
    
    return fig, ax


def example_combined_plot_with_inset():
    """
    Example 4: Combined normalized plot with zoom inset.
    
    This creates a comprehensive visualization with both a normalized
    spectrum and a zoomed inset showing detail around the signal.
    """
    print("\n" + "=" * 70)
    print("Example 4: Combined Plot with Inset")
    print("=" * 70)
    
    # Generate modulator output
    signal_freq = 2000.0  # 2 kHz
    osr = 512
    
    print(f"\nGenerating modulator output...")
    print(f"  Signal frequency: {signal_freq} Hz")
    print(f"  OSR: {osr}")
    
    modulator_output, sampling_freq, filter_cutoff = generate_modulator_output(
        signal_frequency_hz=signal_freq,
        oversampling_ratio=osr,
        modulator_order=2,
        signal_amplitude=0.5,
        number_of_samples=32768  # More samples for better resolution
    )
    
    # Create combined plot
    print("\nCreating combined plot with inset...")
    fig, (ax_main, ax_inset) = plot_spectrum_with_inset(
        modulator_output=modulator_output,
        sampling_frequency_hz=sampling_freq,
        signal_frequency_hz=signal_freq,
        filter_cutoff_hz=filter_cutoff,
        zoom_factor=4.0,
        use_welch=True
    )
    
    print("\nCombined plot complete. Main plot shows normalized spectrum,")
    print("inset shows zoomed view in kHz around signal frequency.")
    
    return fig, (ax_main, ax_inset)


def example_osr_comparison():
    """
    Example 5: Compare different OSR values.
    
    This demonstrates how the plotting function helps compare
    noise shaping effectiveness across different OSR values.
    """
    print("\n" + "=" * 70)
    print("Example 5: OSR Comparison")
    print("=" * 70)
    
    signal_freq = 1000.0  # 1 kHz
    osr_values = [64, 256, 1024]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f'Noise Shaping Comparison for Different OSR Values (f_signal = {signal_freq} Hz)',
        fontsize=14,
        fontweight='bold'
    )
    
    for idx, osr in enumerate(osr_values):
        print(f"\n  Generating and plotting for OSR = {osr}...")
        
        modulator_output, sampling_freq, filter_cutoff = generate_modulator_output(
            signal_frequency_hz=signal_freq,
            oversampling_ratio=osr,
            modulator_order=2,
            signal_amplitude=0.5,
            number_of_samples=16384
        )
        
        # Plot on subplot
        plot_modulator_spectrum(
            modulator_output=modulator_output,
            sampling_frequency_hz=sampling_freq,
            signal_frequency_hz=signal_freq,
            filter_cutoff_hz=filter_cutoff,
            mode='normalized',
            use_welch=True,
            show_diagnostics=False,
            fig_ax=(fig, axes[idx])
        )
        
        axes[idx].set_title(f'OSR = {osr}', fontsize=11)
    
    plt.tight_layout()
    
    print("\nOSR comparison complete. Notice how higher OSR values push")
    print("noise to higher frequencies, improving in-band SNR.")
    
    return fig, axes


def run_all_examples():
    """
    Run all examples in sequence.
    """
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#  SPECTRUM PLOTTING EXAMPLES FOR DELTA-SIGMA MODULATOR ANALYSIS  #")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    
    # Run each example
    example_normalized_plot()
    example_zoom_plot()
    example_full_spectrum_plot()
    example_combined_plot_with_inset()
    example_osr_comparison()
    
    print("\n" + "=" * 70)
    print("All examples complete!")
    print("=" * 70)
    print("\nKey takeaways:")
    print("  • 'normalized' mode: Best for comparing different signal frequencies")
    print("  • 'zoom' mode: Best for examining signal peak detail")
    print("  • 'full' mode: Best for seeing complete noise shaping profile")
    print("  • Welch method: Provides robust PSD estimates for large datasets")
    print("  • Inset plots: Combine overview and detail in one figure")
    print("\nClose all plot windows to exit.")
    print("=" * 70)
    
    plt.show()


if __name__ == "__main__":
    """
    Main entry point for running examples.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Spectrum Plotting Examples for Delta-Sigma Modulator'
    )
    parser.add_argument(
        '--example',
        type=str,
        choices=['normalized', 'zoom', 'full', 'inset', 'osr', 'all'],
        default='all',
        help='Which example to run (default: all)'
    )
    
    args = parser.parse_args()
    
    if args.example == 'all':
        run_all_examples()
    elif args.example == 'normalized':
        example_normalized_plot()
        plt.show()
    elif args.example == 'zoom':
        example_zoom_plot()
        plt.show()
    elif args.example == 'full':
        example_full_spectrum_plot()
        plt.show()
    elif args.example == 'inset':
        example_combined_plot_with_inset()
        plt.show()
    elif args.example == 'osr':
        example_osr_comparison()
        plt.show()
