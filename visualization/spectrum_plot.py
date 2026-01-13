"""
Spectrum Plotting Module for Delta-Sigma Modulator
===================================================

This module provides advanced frequency spectrum plotting capabilities
with support for Welch's method, multiple visualization modes, and
adaptive resolution based on sampling frequency.

Features:
- Welch PSD estimation for robust spectral analysis with large fs
- Three plotting modes: 'full' (Nyquist), 'zoom' (around signal), 'normalized' (f/f_signal)
- Adaptive frequency resolution based on sampling frequency
- Vertical markers for signal frequency and filter cutoff
- Diagnostic output for fs/OSR relationship
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict
from scipy import signal as scipy_signal


def compute_psd_welch(
    signal_data: np.ndarray,
    sampling_frequency_hz: float,
    nperseg: Optional[int] = None,
    nfft: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Power Spectral Density using Welch's method with Hanning window.
    
    Welch's method reduces variance in PSD estimates by averaging multiple
    overlapping FFT segments. This is particularly useful for large datasets
    with high sampling frequencies.
    
    Args:
        signal_data: Input signal array.
        sampling_frequency_hz: Sampling frequency in Hz.
        nperseg: Length of each segment for Welch. If None, auto-calculated.
        nfft: FFT size. If None, uses nperseg.
    
    Returns:
        Tuple of (frequencies_hz, psd_linear) where psd_linear is in linear scale.
    """
    n_samples = len(signal_data)
    
    # Auto-calculate nperseg for good frequency resolution
    if nperseg is None:
        # Use segments that give reasonable frequency resolution
        # Aim for ~256-1024 frequency bins, but limit segment size
        if n_samples >= 8192:
            nperseg = min(2048, n_samples // 4)
        elif n_samples >= 2048:
            nperseg = min(1024, n_samples // 4)
        else:
            nperseg = min(512, n_samples // 2)
    
    # Ensure nperseg doesn't exceed signal length
    nperseg = min(nperseg, n_samples)
    
    # Use nfft for zero-padding if specified (higher frequency resolution)
    if nfft is None:
        nfft = nperseg
    
    # Compute Welch PSD with Hanning window and 50% overlap
    frequencies, psd = scipy_signal.welch(
        signal_data,
        fs=sampling_frequency_hz,
        window='hann',
        nperseg=nperseg,
        nfft=nfft,
        noverlap=nperseg // 2,
        scaling='density',
        return_onesided=True
    )
    
    return frequencies, psd


def compute_psd_fft(
    signal_data: np.ndarray,
    sampling_frequency_hz: float,
    nfft: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Power Spectral Density using FFT with Hanning window.
    
    Fallback method for cases where Welch is not needed or available.
    Uses zero-padding to improve frequency resolution if nfft is specified.
    
    Args:
        signal_data: Input signal array.
        sampling_frequency_hz: Sampling frequency in Hz.
        nfft: FFT size (for zero-padding). If None, uses signal length.
    
    Returns:
        Tuple of (frequencies_hz, psd_linear) where psd_linear is in linear scale.
    """
    n_samples = len(signal_data)
    
    # Apply Hanning window to reduce spectral leakage
    window = scipy_signal.windows.hann(n_samples)
    windowed_signal = signal_data * window
    
    # Determine FFT size
    if nfft is None:
        nfft = n_samples
    
    # Compute FFT
    spectrum = np.fft.fft(windowed_signal, n=nfft)
    
    # Compute power spectrum (one-sided)
    psd = np.abs(spectrum[:nfft // 2 + 1]) ** 2
    
    # Normalize by window power and sampling frequency
    window_power = np.sum(window ** 2)
    psd = psd / (sampling_frequency_hz * window_power)
    
    # Double power (except DC and Nyquist) for one-sided spectrum
    # Only double if we have more than 2 frequency bins
    if len(psd) > 2:
        psd[1:-1] *= 2
    
    # Frequency axis
    frequencies = np.fft.fftfreq(nfft, d=1.0 / sampling_frequency_hz)[:nfft // 2 + 1]
    
    return frequencies, psd


def plot_modulator_spectrum(
    modulator_output: np.ndarray,
    sampling_frequency_hz: float,
    signal_frequency_hz: float,
    filter_cutoff_hz: Optional[float] = None,
    mode: str = 'normalized',
    zoom_factor: float = 5.0,
    use_welch: bool = True,
    nfft: Optional[int] = None,
    nperseg: Optional[int] = None,
    title: Optional[str] = None,
    show_diagnostics: bool = True,
    save_path: Optional[str] = None,
    fig_ax: Optional[Tuple] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot frequency spectrum of delta-sigma modulator output with multiple modes.
    
    This function provides flexible spectrum visualization with three modes:
    - 'full': Shows full Nyquist range (0 to fs/2) in kHz
    - 'zoom': Shows zoomed range around signal (0 to zoom_factor * signal_frequency)
    - 'normalized': Shows frequency normalized by signal frequency (f/f_signal)
    
    By default, uses Welch's method for robust PSD estimation, especially
    beneficial for large sampling frequencies. Falls back to FFT for smaller datasets.
    
    Args:
        modulator_output: The 1-bit modulator output bitstream.
        sampling_frequency_hz: Sampling frequency in Hz.
        signal_frequency_hz: Input signal frequency in Hz.
        filter_cutoff_hz: Reconstruction filter cutoff frequency in Hz (optional).
        mode: Plot mode - 'full', 'zoom', or 'normalized'. Default: 'normalized'.
        zoom_factor: For 'zoom' mode, shows 0 to zoom_factor * signal_frequency_hz.
        use_welch: If True, use Welch PSD; if False, use FFT. Default: True.
        nfft: FFT size for zero-padding (higher resolution). If None, auto-determined.
        nperseg: Segment length for Welch. If None, auto-determined.
        title: Custom plot title. If None, auto-generated.
        show_diagnostics: If True, print fs/OSR diagnostics to console.
        save_path: If provided, save figure to this path.
        fig_ax: Optional tuple of (fig, ax) to plot on existing axes.
    
    Returns:
        Tuple of (fig, ax) matplotlib objects.
    
    Example:
        >>> # Generate test signal and modulator output
        >>> from signals.digital_signal_generator import DigitalSignalGenerator
        >>> from modulator.delta_sigma_modulator import DeltaSigmaModulator
        >>> 
        >>> fs = 256000  # 256 kHz
        >>> f_signal = 1000  # 1 kHz
        >>> # ... create modulator output ...
        >>> 
        >>> # Plot normalized spectrum
        >>> fig, ax = plot_modulator_spectrum(
        ...     modulator_output=output,
        ...     sampling_frequency_hz=fs,
        ...     signal_frequency_hz=f_signal,
        ...     mode='normalized'
        ... )
    """
    # Validate inputs
    if mode not in ['full', 'zoom', 'normalized']:
        raise ValueError(f"mode must be 'full', 'zoom', or 'normalized', got '{mode}'")
    
    # Calculate derived parameters
    nyquist_freq_hz = sampling_frequency_hz / 2.0
    osr = sampling_frequency_hz / (2.0 * signal_frequency_hz)
    
    # Print diagnostics if requested
    if show_diagnostics:
        print(f"\n--- Spectrum Plot Diagnostics ---")
        print(f"  Sampling Frequency:      {sampling_frequency_hz / 1e3:.2f} kHz")
        print(f"  Signal Frequency:        {signal_frequency_hz / 1e3:.2f} kHz")
        print(f"  Nyquist Frequency:       {nyquist_freq_hz / 1e3:.2f} kHz")
        print(f"  Oversampling Ratio:      {osr:.1f}")
        print(f"  Number of Samples:       {len(modulator_output)}")
        if filter_cutoff_hz:
            print(f"  Filter Cutoff:           {filter_cutoff_hz / 1e3:.2f} kHz")
        print(f"  PSD Method:              {'Welch' if use_welch else 'FFT'}")
    
    # Compute PSD
    if use_welch:
        try:
            frequencies, psd_linear = compute_psd_welch(
                modulator_output,
                sampling_frequency_hz,
                nperseg=nperseg,
                nfft=nfft
            )
        except Exception as e:
            # Fallback to FFT if Welch fails
            if show_diagnostics:
                print(f"  Warning: Welch failed ({e}), using FFT fallback")
            frequencies, psd_linear = compute_psd_fft(
                modulator_output,
                sampling_frequency_hz,
                nfft=nfft
            )
    else:
        frequencies, psd_linear = compute_psd_fft(
            modulator_output,
            sampling_frequency_hz,
            nfft=nfft
        )
    
    # Convert to dB (avoid log(0))
    psd_db = 10 * np.log10(psd_linear + 1e-20)
    
    # Create figure if not provided
    if fig_ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig, ax = fig_ax
    
    # Plot based on mode
    if mode == 'full':
        # Full Nyquist range in kHz
        freq_khz = frequencies / 1000.0
        ax.plot(freq_khz, psd_db, 'b-', linewidth=0.6, label='Modulator Output')
        ax.set_xlabel('Frequency (kHz)', fontsize=11)
        ax.set_xlim(0, nyquist_freq_hz / 1000.0)
        
        # Mark signal frequency
        ax.axvline(
            x=signal_frequency_hz / 1000.0,
            color='g', linestyle='--', linewidth=1.5,
            label=f'Signal: {signal_frequency_hz / 1000.0:.2f} kHz'
        )
        
        # Mark filter cutoff
        if filter_cutoff_hz:
            ax.axvline(
                x=filter_cutoff_hz / 1000.0,
                color='r', linestyle=':', linewidth=1.5,
                label=f'Filter Cutoff: {filter_cutoff_hz / 1000.0:.2f} kHz'
            )
        
        plot_title = title or f'Frequency Spectrum - Full Nyquist Range (OSR={osr:.1f})'
    
    elif mode == 'zoom':
        # Zoom around signal frequency
        max_freq = zoom_factor * signal_frequency_hz
        freq_khz = frequencies / 1000.0
        
        # Filter to zoom range
        zoom_mask = frequencies <= max_freq
        freq_khz_zoom = freq_khz[zoom_mask]
        psd_db_zoom = psd_db[zoom_mask]
        
        ax.plot(freq_khz_zoom, psd_db_zoom, 'b-', linewidth=0.6, label='Modulator Output')
        ax.set_xlabel('Frequency (kHz)', fontsize=11)
        ax.set_xlim(0, max_freq / 1000.0)
        
        # Mark signal frequency
        ax.axvline(
            x=signal_frequency_hz / 1000.0,
            color='g', linestyle='--', linewidth=1.5,
            label=f'Signal: {signal_frequency_hz / 1000.0:.2f} kHz'
        )
        
        # Mark filter cutoff if in range
        if filter_cutoff_hz and filter_cutoff_hz <= max_freq:
            ax.axvline(
                x=filter_cutoff_hz / 1000.0,
                color='r', linestyle=':', linewidth=1.5,
                label=f'Filter Cutoff: {filter_cutoff_hz / 1000.0:.2f} kHz'
            )
        
        plot_title = title or f'Frequency Spectrum - Zoom to {zoom_factor}× Signal (OSR={osr:.1f})'
    
    else:  # mode == 'normalized'
        # Normalize frequency by signal frequency
        freq_normalized = frequencies / signal_frequency_hz
        ax.plot(freq_normalized, psd_db, 'b-', linewidth=0.6, label='Modulator Output')
        ax.set_xlabel('Frequency (f / f_signal)', fontsize=11)
        
        # Limit x-axis to reasonable range (e.g., 0 to OSR/2 or 100, whichever is smaller)
        max_normalized_freq = min(osr / 2, 100)
        ax.set_xlim(0, max_normalized_freq)
        
        # Mark signal frequency at f/f_signal = 1
        ax.axvline(
            x=1.0,
            color='g', linestyle='--', linewidth=1.5,
            label='Signal (1× f_signal)'
        )
        
        # Mark filter cutoff in normalized units
        if filter_cutoff_hz:
            cutoff_normalized = filter_cutoff_hz / signal_frequency_hz
            if cutoff_normalized <= max_normalized_freq:
                ax.axvline(
                    x=cutoff_normalized,
                    color='r', linestyle=':', linewidth=1.5,
                    label=f'Filter Cutoff ({cutoff_normalized:.1f}× f_signal)'
                )
        
        plot_title = title or f'Frequency Spectrum - Normalized (OSR={osr:.1f})'
    
    # Common plot formatting
    ax.set_ylabel('Power Spectral Density (dB)', fontsize=11)
    ax.set_title(plot_title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper right', fontsize=9)
    
    # Auto-scale y-axis with some headroom
    y_min = np.percentile(psd_db[psd_db > -np.inf], 1)  # 1st percentile
    y_max = np.percentile(psd_db[psd_db < np.inf], 99)  # 99th percentile
    ax.set_ylim(y_min - 10, y_max + 10)
    
    # Only apply tight_layout if we created the figure (no inset yet)
    if fig_ax is None:
        plt.tight_layout()
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        if show_diagnostics:
            print(f"  Figure saved to: {save_path}")
    
    return fig, ax


def plot_spectrum_with_inset(
    modulator_output: np.ndarray,
    sampling_frequency_hz: float,
    signal_frequency_hz: float,
    filter_cutoff_hz: Optional[float] = None,
    zoom_factor: float = 5.0,
    use_welch: bool = True,
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    Plot normalized spectrum with a zoomed inset showing detail around signal.
    
    Creates a figure with:
    - Main plot: Normalized frequency spectrum (f/f_signal)
    - Inset: Zoomed view in kHz showing detail around signal frequency
    
    This provides both a comprehensive view of noise shaping and
    detailed view of the signal region.
    
    Args:
        modulator_output: The 1-bit modulator output bitstream.
        sampling_frequency_hz: Sampling frequency in Hz.
        signal_frequency_hz: Input signal frequency in Hz.
        filter_cutoff_hz: Reconstruction filter cutoff frequency in Hz (optional).
        zoom_factor: Inset shows 0 to zoom_factor * signal_frequency_hz.
        use_welch: If True, use Welch PSD; if False, use FFT.
        save_path: If provided, save figure to this path.
    
    Returns:
        Tuple of (fig, (ax_main, ax_inset)) matplotlib objects.
    
    Example:
        >>> fig, (ax_main, ax_inset) = plot_spectrum_with_inset(
        ...     modulator_output=output,
        ...     sampling_frequency_hz=256000,
        ...     signal_frequency_hz=1000,
        ...     filter_cutoff_hz=1500,
        ...     zoom_factor=3.0
        ... )
    """
    # Create figure with main plot
    fig, ax_main = plt.subplots(figsize=(14, 7))
    
    # Plot main normalized spectrum
    plot_modulator_spectrum(
        modulator_output=modulator_output,
        sampling_frequency_hz=sampling_frequency_hz,
        signal_frequency_hz=signal_frequency_hz,
        filter_cutoff_hz=filter_cutoff_hz,
        mode='normalized',
        use_welch=use_welch,
        show_diagnostics=True,
        fig_ax=(fig, ax_main)
    )
    
    # Create inset axes for zoomed view
    # Position: [left, bottom, width, height] in figure coordinates
    ax_inset = fig.add_axes([0.55, 0.55, 0.35, 0.30])
    
    # Plot zoomed spectrum in inset
    plot_modulator_spectrum(
        modulator_output=modulator_output,
        sampling_frequency_hz=sampling_frequency_hz,
        signal_frequency_hz=signal_frequency_hz,
        filter_cutoff_hz=filter_cutoff_hz,
        mode='zoom',
        zoom_factor=zoom_factor,
        use_welch=use_welch,
        show_diagnostics=False,
        fig_ax=(fig, ax_inset)
    )
    
    # Update inset title to be more compact
    osr = sampling_frequency_hz / (2.0 * signal_frequency_hz)
    ax_inset.set_title(f'Zoom: 0-{zoom_factor}× Signal Frequency', fontsize=9)
    ax_inset.tick_params(labelsize=8)
    ax_inset.legend(fontsize=7)
    
    # Add border to inset
    for spine in ax_inset.spines.values():
        spine.set_edgecolor('gray')
        spine.set_linewidth(1.5)
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure with inset saved to: {save_path}")
    
    return fig, (ax_main, ax_inset)


# Module-level usage example
if __name__ == "__main__":
    """
    Module usage example - demonstrates the spectrum plotting capabilities.
    """
    print("=" * 70)
    print("Spectrum Plot Module - Usage Example")
    print("=" * 70)
    
    # Generate synthetic modulator output for demonstration
    print("\nGenerating synthetic delta-sigma modulator output...")
    
    # Parameters
    signal_freq = 1000  # 1 kHz
    osr = 256
    sampling_freq = 2 * signal_freq * osr  # 512 kHz
    n_samples = 16384
    
    # Create synthetic bitstream with signal at f_signal
    t = np.arange(n_samples) / sampling_freq
    signal_component = 0.5 * np.sin(2 * np.pi * signal_freq * t)
    
    # Add high-frequency noise (delta-sigma shaped)
    noise = np.random.randn(n_samples) * 0.3
    
    # Simple 1-bit quantization to simulate modulator output
    combined = signal_component + noise
    modulator_output = np.sign(combined)
    
    print(f"  Signal frequency: {signal_freq} Hz")
    print(f"  Sampling frequency: {sampling_freq / 1e3:.1f} kHz")
    print(f"  OSR: {osr}")
    print(f"  Number of samples: {n_samples}")
    
    # Example 1: Normalized spectrum
    print("\n--- Example 1: Normalized Spectrum ---")
    fig1, ax1 = plot_modulator_spectrum(
        modulator_output=modulator_output,
        sampling_frequency_hz=sampling_freq,
        signal_frequency_hz=signal_freq,
        filter_cutoff_hz=signal_freq * 1.5,
        mode='normalized',
        use_welch=True
    )
    
    # Example 2: Zoom mode
    print("\n--- Example 2: Zoom Mode ---")
    fig2, ax2 = plot_modulator_spectrum(
        modulator_output=modulator_output,
        sampling_frequency_hz=sampling_freq,
        signal_frequency_hz=signal_freq,
        filter_cutoff_hz=signal_freq * 1.5,
        mode='zoom',
        zoom_factor=5.0,
        use_welch=True
    )
    
    # Example 3: Full Nyquist range
    print("\n--- Example 3: Full Nyquist Range ---")
    fig3, ax3 = plot_modulator_spectrum(
        modulator_output=modulator_output,
        sampling_frequency_hz=sampling_freq,
        signal_frequency_hz=signal_freq,
        filter_cutoff_hz=signal_freq * 1.5,
        mode='full',
        use_welch=True
    )
    
    print("\n" + "=" * 70)
    print("Examples complete. Close plot windows to continue.")
    print("=" * 70)
    
    plt.show()
