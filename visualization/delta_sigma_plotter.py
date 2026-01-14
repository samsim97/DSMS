"""
Delta-Sigma Plotter
===================

This module provides comprehensive plotting functions for visualizing
and analyzing delta-sigma modulator signals. 

Plots included:
1. Time-domain signals (input, bitstream, reconstructed)
2. Frequency-domain analysis (spectrum, noise shaping)
3. Integrator state evolution (stability analysis)
4. Performance comparison across configurations
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider
from typing import Optional, List, Tuple, Dict, Any


class DeltaSigmaPlotter:
    """
    Plotting utilities for delta-sigma DAC simulation analysis.

    This class provides static methods for creating various plots
    to visualize and analyze simulation results.

    All methods are static to allow easy use without instantiation.
    """

    # Default figure size for consistency
    DEFAULT_FIGURE_SIZE: Tuple[int, int] = (14, 10)
    DEFAULT_SINGLE_PLOT_SIZE: Tuple[int, int] = (12, 6)

    @staticmethod
    def plot_time_domain_signals(
        time_axis_seconds: np.ndarray,
        input_signal: np.ndarray,
        modulator_output: np.ndarray,
        reconstructed_signal: np.ndarray,
        title_prefix: str = "",
        samples_to_show: Optional[int] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot the three main signals in the delta-sigma DAC chain.

        This creates a 3-subplot figure showing:
        1. Original input signal (digital PCM)
        2. Modulator output (1-bit bitstream)
        3. Reconstructed signal (after low-pass filter)

        Args:
            time_axis_seconds: Time values for x-axis. 
            input_signal: The original input to the modulator.
            modulator_output: The 1-bit output bitstream.
            reconstructed_signal: Signal after reconstruction filter.
            title_prefix: Optional prefix for the main title.
            samples_to_show:  Limit display to this many samples (for clarity).
            save_path: If provided, save figure to this path.
        """
        # Limit samples for display if requested
        if samples_to_show is not None:
            samples_to_show = min(samples_to_show, len(time_axis_seconds))
            time_axis_seconds = time_axis_seconds[:samples_to_show]
            input_signal = input_signal[:samples_to_show]
            modulator_output = modulator_output[:samples_to_show]
            reconstructed_signal = reconstructed_signal[:samples_to_show]

        # Convert time to milliseconds for readability
        time_axis_ms: np.ndarray = time_axis_seconds * 1000

        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        fig.suptitle(
            f"{title_prefix}Delta-Sigma DAC Signal Chain",
            fontsize=14,
            fontweight='bold'
        )

        # ===== SUBPLOT 1: Input Signal =====
        axes[0].plot(time_axis_ms, input_signal, 'b-', linewidth=0.8)
        axes[0].set_ylabel('Amplitude', fontsize=10)
        axes[0].set_title('Input Signal (Digital PCM)', fontsize=11)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(-1.2, 1.2)
        axes[0].axhline(y=0, color='k', linewidth=0.5)

        # ===== SUBPLOT 2: Modulator Output (Bitstream) =====
        # Use step plot for binary signal
        axes[1].step(
            time_axis_ms, modulator_output,
            'r-', linewidth=0.5, where='post'
        )
        axes[1].set_ylabel('Output Level', fontsize=10)
        axes[1].set_title('Modulator Output (1-bit Bitstream)', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(-1.5, 1.5)
        axes[1].set_yticks([-1, 0, 1])
        axes[1].axhline(y=0, color='k', linewidth=0.5)

        # ===== SUBPLOT 3: Reconstructed Signal =====
        axes[2].plot(time_axis_ms, reconstructed_signal, 'g-', linewidth=0.8)
        axes[2].plot(
            time_axis_ms, input_signal,
            'b--', linewidth=0.5, alpha=0.5, label='Original Input'
        )
        axes[2].set_xlabel('Time (ms)', fontsize=10)
        axes[2].set_ylabel('Amplitude', fontsize=10)
        axes[2].set_title('Reconstructed Signal (After Low-Pass Filter)', fontsize=11)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim(-1.2, 1.2)
        axes[2].legend(loc='upper right', fontsize=9)
        axes[2].axhline(y=0, color='k', linewidth=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to:  {save_path}")

        plt.show()

    @staticmethod
    def plot_frequency_spectrum(
        signal: np.ndarray,
        sampling_frequency_hz: float,
        signal_label: str = "Signal",
        signal_frequency_hz: Optional[float] = None,
        cutoff_frequency_hz: Optional[float] = None,
        show_full_spectrum: bool = False,
        frequency_multiplier: int = 20,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot the frequency spectrum of a signal.

        Shows the power spectral density in dB, useful for analyzing
        noise shaping effectiveness. 

        Args:
            signal:  The signal to analyze.
            sampling_frequency_hz: Sampling rate in Hz.
            signal_label: Label for the plot legend.
            signal_frequency_hz: If provided, mark signal frequency.
            cutoff_frequency_hz: If provided, mark filter cutoff. 
            show_full_spectrum:  If True, show full spectrum to Nyquist.
            save_path: If provided, save figure to this path.
        """
        number_of_samples: int = len(signal)

        # Apply Hanning window to reduce spectral leakage
        window: np.ndarray = np.hanning(number_of_samples)
        windowed_signal: np.ndarray = signal * window

        # Compute FFT
        spectrum: np.ndarray = np. fft.fft(windowed_signal)

        # Compute power spectrum in dB
        power_spectrum:  np.ndarray = np.abs(spectrum) ** 2
        # Avoid log(0) by adding small value
        power_spectrum_db: np.ndarray = 10 * np.log10(power_spectrum + 1e-20)

        # Frequency axis
        frequency_axis: np.ndarray = np.fft.fftfreq(
            number_of_samples,
            d=1.0 / sampling_frequency_hz
        )

        # Only show positive frequencies
        positive_mask: np.ndarray = frequency_axis >= 0
        frequencies_positive: np.ndarray = frequency_axis[positive_mask]
        spectrum_positive: np.ndarray = power_spectrum_db[positive_mask]

        # Limit frequency range for better visibility.
        # Use `frequency_multiplier` to allow a wider default display around cutoff.
        if not show_full_spectrum and cutoff_frequency_hz is not None:
            freq_limit = min(cutoff_frequency_hz * frequency_multiplier, sampling_frequency_hz / 2)
        else:
            freq_limit = sampling_frequency_hz / 2

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Convert to kHz for readability
        frequencies_khz: np.ndarray = frequencies_positive / 1000

        ax.plot(frequencies_khz, spectrum_positive, 'b-', linewidth=0.5, label=signal_label)

        # Mark signal frequency
        if signal_frequency_hz is not None:
            ax.axvline(
                x=signal_frequency_hz / 1000,
                color='g', linestyle='--', linewidth=1.5,
                label=f"Signal:  {signal_frequency_hz/1000:.1f} kHz"
            )

        # Mark cutoff frequency
        if cutoff_frequency_hz is not None: 
            ax.axvline(
                x=cutoff_frequency_hz / 1000,
                color='r', linestyle=':', linewidth=1.5,
                label=f'Filter Cutoff: {cutoff_frequency_hz/1000:.1f} kHz'
            )

        ax.set_xlabel('Frequency (kHz)', fontsize=11)
        ax.set_ylabel('Power Spectral Density (dB)', fontsize=11)
        ax.set_title('Frequency Spectrum Analysis', fontsize=12, fontweight='bold')
        ax.set_xlim(0, freq_limit / 1000)
        ax.set_ylim(-120, max(spectrum_positive) + 10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)

        # Make room for interactive widgets below the plot
        plt.subplots_adjust(bottom=0.32)

        # Interactive RangeSlider to control visible frequency range (in kHz)
        try:
            ax_slider = plt.axes([0.15, 0.02, 0.7, 0.04], facecolor='lightgoldenrodyellow')
            slider = RangeSlider(
                ax=ax_slider,
                label='Visible Frequency Range (kHz)',
                valmin=0.0,
                valmax=float(frequencies_khz.max()),
                valinit=(0.0, float(freq_limit / 1000)),
                valfmt='%.2f'
            )

            def _update(val):
                vmin, vmax = slider.val
                ax.set_xlim(vmin, vmax)
                fig.canvas.draw_idle()

            slider.on_changed(_update)
        except Exception:
            # If interactive widgets are not available in the current backend,
            # silently continue without interactive controls.
            pass

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

    @staticmethod
    def plot_noise_shaping_comparison(
        signals_dict: Dict[str, np.ndarray],
        sampling_frequency_hz: float,
        signal_bandwidth_hz: float,
        save_path: Optional[str] = None
    ) -> None:
        """
        Compare frequency spectra of multiple signals/configurations.

        Useful for comparing different modulator orders. 

        Args:
            signals_dict: Dictionary of {label: signal_array}. 
            sampling_frequency_hz:  Sampling rate. 
            signal_bandwidth_hz:  Bandwidth of interest.
            save_path: If provided, save figure to this path. 
        """
        fig, ax = plt.subplots(figsize=(12, 7))

        colors = ['b', 'r', 'g', 'm', 'c', 'orange', 'brown']

        # Track global min/max for adaptive y-limits to avoid clipping peaks
        max_spec_db = -np.inf
        min_spec_db = np.inf

        for idx, (label, signal) in enumerate(signals_dict.items()):
            number_of_samples: int = len(signal)

            # Apply window and compute spectrum
            window: np.ndarray = np.hanning(number_of_samples)
            spectrum: np.ndarray = np.fft.fft(signal * window)
            power_db: np.ndarray = 10 * np.log10(np.abs(spectrum) ** 2 + 1e-20)

            frequency_axis: np.ndarray = np.fft.fftfreq(
                number_of_samples, d=1.0 / sampling_frequency_hz
            )

            # Positive frequencies only
            positive_mask: np.ndarray = frequency_axis >= 0
            freq_khz: np.ndarray = frequency_axis[positive_mask] / 1000
            spec_db: np.ndarray = power_db[positive_mask]

            color = colors[idx % len(colors)]
            ax.plot(freq_khz, spec_db, color=color, linewidth=0.7, label=label, alpha=0.8)

            # update global min/max
            if spec_db.size:
                max_spec_db = max(max_spec_db, float(np.max(spec_db)))
                min_spec_db = min(min_spec_db, float(np.min(spec_db)))

        # Mark signal bandwidth
        ax.axvline(
            x=signal_bandwidth_hz / 1000,
            color='k', linestyle='--', linewidth=2,
            label=f'Signal BW: {signal_bandwidth_hz/1000:.1f} kHz'
        )

        ax.set_xlabel('Frequency (kHz)', fontsize=11)
        ax.set_ylabel('Power Spectral Density (dB)', fontsize=11)
        ax.set_title(
            'Noise Shaping Comparison Across Different Orders',
            fontsize=12, fontweight='bold'
        )
        ax.set_xlim(0, signal_bandwidth_hz * 5 / 1000)

        # Adaptive y-limits: leave some headroom above the highest peak and floor below
        if max_spec_db == -np.inf:
            # fallback if no data
            ax.set_ylim(-100, 20)
        else:
            upper = max(20.0, max_spec_db + 10.0)
            lower = min(-120.0, min_spec_db - 10.0) if min_spec_db != np.inf else -120.0
            ax.set_ylim(lower, upper)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        # Make room for interactive slider below the plot
        plt.subplots_adjust(bottom=0.32)

        # Add RangeSlider for visible frequency range (kHz)
        try:
            valinit_max = float(min(signal_bandwidth_hz * 5 / 1000, (freq_khz.max() if 'freq_khz' in locals() else frequency_axis.max()/1000)))
            ax_slider = plt.axes([0.15, 0.02, 0.7, 0.04], facecolor='lightgoldenrodyellow')
            slider = RangeSlider(
                ax=ax_slider,
                label='Visible Frequency Range (kHz)',
                valmin=0.0,
                valmax=float(freq_khz.max()),
                valinit=(0.0, valinit_max),
                valfmt='%.2f'
            )

            def _update(val):
                vmin, vmax = slider.val
                ax.set_xlim(vmin, vmax)
                fig.canvas.draw_idle()

            slider.on_changed(_update)
        except Exception:
            pass

        plt.tight_layout()

        if save_path: 
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

    @staticmethod
    def plot_integrator_states(
        time_axis_seconds: np.ndarray,
        integrator_history: np.ndarray,
        modulator_order: int,
        samples_to_show: Optional[int] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot the evolution of integrator states over time.

        This is crucial for stability analysis.  Unbounded growth
        indicates instability.

        Args:
            time_axis_seconds: Time values. 
            integrator_history: Array of shape (samples, order).
            modulator_order: Number of integrators. 
            samples_to_show:  Limit samples for clarity.
            save_path: If provided, save figure to this path. 
        """
        if samples_to_show is not None:
            samples_to_show = min(samples_to_show, len(time_axis_seconds))
            time_axis_seconds = time_axis_seconds[:samples_to_show]
            integrator_history = integrator_history[:samples_to_show, :]

        time_ms: np.ndarray = time_axis_seconds * 1000

        fig, axes = plt.subplots(
            modulator_order, 1,
            figsize=(12, 3 * modulator_order),
            sharex=True
        )

        if modulator_order == 1:
            axes = [axes]

        colors = ['blue', 'red', 'green', 'purple', 'orange']

        for i in range(modulator_order):
            ax = axes[i]
            color = colors[i % len(colors)]

            ax.plot(time_ms, integrator_history[:, i], color=color, linewidth=0.5)
            ax.set_ylabel(f'State', fontsize=10)
            ax.set_title(f'Integrator {i + 1} State', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linewidth=0.5)

            # Show max absolute value
            max_val = np.max(np.abs(integrator_history[:, i]))
            ax.text(
                0.02, 0.95, f'Max |state|: {max_val:.2f}',
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )

        axes[-1].set_xlabel('Time (ms)', fontsize=10)

        fig.suptitle(
            'Integrator State Evolution (Stability Analysis)',
            fontsize=12, fontweight='bold'
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

    @staticmethod
    def plot_snr_vs_osr(
        osr_values: List[int],
        snr_values:  Dict[str, List[float]],
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot SNR vs OSR for different modulator orders.

        Args:
            osr_values:  List of OSR values tested.
            snr_values: Dict of {order_label: [snr_values]}.
            save_path: If provided, save figure to this path.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        markers = ['o', 's', '^', 'D', 'v']
        colors = ['blue', 'red', 'green', 'purple', 'orange']

        for idx, (label, snr_list) in enumerate(snr_values.items()):
            marker = markers[idx % len(markers)]
            color = colors[idx % len(colors)]
            ax.plot(
                osr_values, snr_list,
                marker=marker, color=color, linewidth=2,
                markersize=8, label=label
            )

        ax.set_xlabel('Oversampling Ratio (OSR)', fontsize=11)
        ax.set_ylabel('SNR (dB)', fontsize=11)
        ax.set_title('SNR vs Oversampling Ratio', fontsize=12, fontweight='bold')
        ax.set_xscale('log', base=2)
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(loc='lower right', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

    @staticmethod
    def plot_performance_summary(
        metrics_dict: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> None:
        """
        Create a summary dashboard with key metrics.

        Args:
            metrics_dict: Dictionary containing all metrics.
            save_path: If provided, save figure to this path.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Unpack metrics
        snr_db = metrics_dict.get('snr_db', 0)
        enob = metrics_dict.get('enob', 0)
        switching_rate = metrics_dict.get('switching_rate', 0)
        modulator_order = metrics_dict.get('modulator_order', 1)
        osr = metrics_dict.get('osr', 64)

        # ===== Quadrant 1: Key Metrics Text =====
        ax1 = axes[0, 0]
        ax1.axis('off')
        metrics_text = (
            f"PERFORMANCE SUMMARY\n"
            f"{'=' * 30}\n\n"
            f"Modulator Order:      {modulator_order}\n"
            f"Oversampling Ratio:  {osr}\n\n"
            f"SNR:                  {snr_db:.1f} dB\n"
            f"ENOB:                {enob:.1f} bits\n\n"
            f"Switching Rate:      {switching_rate:.1%}\n"
            f"Relative Power:      {switching_rate/0.5:.2f}"
        )
        ax1.text(
            0.1, 0.9, metrics_text,
            transform=ax1.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        )

        # ===== Quadrant 2: SNR Bar =====
        ax2 = axes[0, 1]
        bar_colors = ['red' if snr_db < 40 else 'orange' if snr_db < 60 else 'green']
        ax2.barh(['SNR'], [snr_db], color=bar_colors, height=0.5)
        ax2.set_xlim(0, max(100, snr_db + 10))
        ax2.set_xlabel('dB', fontsize=10)
        ax2.set_title('Signal-to-Noise Ratio', fontsize=11)
        ax2.axvline(x=60, color='g', linestyle='--', alpha=0.5, label='Good (60 dB)')
        ax2.axvline(x=90, color='b', linestyle='--', alpha=0.5, label='Excellent (90 dB)')
        ax2.legend(fontsize=8)

        # ===== Quadrant 3: ENOB Gauge =====
        ax3 = axes[1, 0]
        ax3.barh(['ENOB'], [enob], color='purple', height=0.5)
        ax3.set_xlim(0, max(20, enob + 2))
        ax3.set_xlabel('bits', fontsize=10)
        ax3.set_title('Effective Number of Bits', fontsize=11)
        # Reference lines
        for bits in [8, 12, 16]: 
            ax3.axvline(x=bits, color='gray', linestyle=':', alpha=0.5)
            ax3.text(bits, 0.7, f'{bits}b', fontsize=8, ha='center')

        # ===== Quadrant 4: Power Indicator =====
        ax4 = axes[1, 1]
        power_indicator = switching_rate / 0.5  # 0 to ~1
        colors_power = ['green' if power_indicator < 0.5 else
                        'orange' if power_indicator < 0.8 else 'red']
        ax4.barh(['Relative\nPower'], [power_indicator], color=colors_power, height=0.5)
        ax4.set_xlim(0, 1.2)
        ax4.set_xlabel('Normalized (1.0 = worst case)', fontsize=10)
        ax4.set_title('Power Consumption Indicator', fontsize=11)
        ax4.axvline(x=0.5, color='g', linestyle='--', alpha=0.5, label='Target')
        ax4.legend(fontsize=8)

        fig.suptitle(
            'Delta-Sigma Modulator Performance Dashboard',
            fontsize=14, fontweight='bold'
        )
        plt.tight_layout()

        if save_path: 
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()