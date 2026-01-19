# Power Spectral Density (PSD) Analysis in DSMS

## Overview

This document explains how to use the true PSD computation and spectrum-based metrics in the Delta-Sigma Modulator Simulation (DSMS) package for comparing configurations and analyzing performance.

## Why True PSD Matters

### The Problem with Simple FFT Plots

The original `plot_frequency_spectrum()` method computed:
```
FFT_power = |FFT(windowed_signal)|²
PSD_display = 10 * log10(FFT_power)
```

This approach has several issues:
1. **Not normalized per Hz**: The y-axis shows relative power in dB, but values aren't calibrated to power per Hz (V²/Hz).
2. **No averaging**: Single FFT has high variance, making noise floor measurements unreliable.
3. **Difficult to compare**: Without proper normalization, comparing different segment lengths or sampling rates gives misleading results.

### The Solution: Welch's Method

Welch's PSD method provides:
1. **True power spectral density** in V²/Hz (or dB/Hz)
2. **Reduced variance** through averaging of multiple overlapping segments
3. **Proper normalization** with `scaling='density'` that accounts for window energy
4. **One-sided PSD** for real signals (doubles power at positive frequencies)

## Usage

### Basic Simulation with PSD Analysis

```python
from main import run_single_simulation

results = run_single_simulation(
    modulator_order=2,
    oversampling_ratio=2048,
    signal_frequency_hz=10000.0,
    signal_amplitude=0.4,
    number_of_samples=655360,
    filter_cutoff_frequency_hz=15000.0,
    plot_results=True,
    
    # PSD-specific parameters
    spectrum_method='welch',          # Use Welch's method (default)
    spectrum_nperseg=8192,            # Segment length for Welch
    spectrum_max_freq_hz=300000,      # Zoom to 300 kHz
    spectrum_remove_transient_fraction=0.2,  # Remove first 20% as transient
    spectrum_db_ref=None              # Use dB/Hz (not dBFS)
)

# Access PSD-based metrics
psd_metrics = results['psd_metrics']
print(f"SNR (from PSD): {psd_metrics['snr_db']:.1f} dB")
print(f"SNDR (from PSD): {psd_metrics['sndr_db']:.1f} dB")
print(f"ENOB (from PSD): {psd_metrics['enob']:.2f} bits")
print(f"THD: {psd_metrics['thd_db']:.1f} dB")
print(f"SFDR: {psd_metrics['sfdr_db']:.1f} dB")
```

### Comparing Different Word Lengths

When comparing configurations with different `input_word_length_bits`, use PSD-based metrics:

```python
# Test different word lengths
word_lengths = [12, 14, 16, 18, 20]
results_by_word_length = {}

for bits in word_lengths:
    results = run_single_simulation(
        modulator_order=2,
        oversampling_ratio=2048,
        signal_frequency_hz=10000.0,
        input_word_length_bits=bits,
        number_of_samples=655360,
        filter_cutoff_frequency_hz=15000.0,
        plot_results=False,
        spectrum_method='welch'
    )
    results_by_word_length[bits] = results['psd_metrics']

# Compare SNDR (includes distortion)
for bits, metrics in results_by_word_length.items():
    print(f"{bits} bits: SNDR = {metrics['sndr_db']:.2f} dB, "
          f"ENOB = {metrics['enob']:.2f} bits")
```

### Standalone PSD Computation

You can also use the PSD utilities directly:

```python
from metrics.psd_utils import compute_welch_psd, psd_to_db
from metrics.spectrum_metrics import compute_all_metrics_from_psd

# Compute PSD
frequencies, psd_linear = compute_welch_psd(
    signal_data=reconstructed_signal,
    sampling_frequency_hz=fs,
    nperseg=8192,
    window='hann',
    remove_dc=True
)

# Convert to dB
psd_db = psd_to_db(psd_linear)

# Compute all metrics
metrics = compute_all_metrics_from_psd(
    frequencies=frequencies,
    psd_linear=psd_linear,
    fundamental_frequency_hz=10000.0,
    band_limit_hz=15000.0,
    num_harmonics=6
)
```

### Plotting PSD

```python
from visualization.delta_sigma_plotter import DeltaSigmaPlotter

DeltaSigmaPlotter.plot_psd_welch(
    signal=reconstructed_signal,
    sampling_frequency_hz=fs,
    signal_label="Reconstructed Signal",
    signal_frequency_hz=10000.0,
    cutoff_frequency_hz=15000.0,
    max_frequency_hz=300000,  # Zoom to 300 kHz
    nperseg=8192,
    return_db=True
)
```

## Understanding the Metrics

### Bitstream PSD vs Reconstructed PSD

1. **Bitstream PSD** (`modulator_output`):
   - Shows noise shaping effectiveness
   - High-frequency noise is pushed out of band by the modulator
   - Use this to verify modulator order and noise shaping
   
2. **Reconstructed PSD** (`reconstructed_signal`):
   - Shows actual output quality after low-pass filtering
   - This is what matters for your application
   - Use this for comparing configurations

**Recommendation**: Analyze the **reconstructed signal PSD** for performance metrics and configuration comparisons.

### SNR vs SNDR

- **SNR (Signal-to-Noise Ratio)**: Excludes harmonic distortion from the denominator
  - Only considers noise power in band (excluding signal and harmonics)
  - Higher than SNDR when harmonics are present
  
- **SNDR (Signal-to-Noise-and-Distortion Ratio)**: Includes harmonic distortion
  - Denominator = noise + distortion
  - More realistic measure of usable dynamic range
  - **Use SNDR for ENOB calculation and comparisons**

Formula: `ENOB = (SNDR - 1.76) / 6.02`

### THD (Total Harmonic Distortion)

Measures harmonic distortion relative to fundamental:
- `THD = sqrt(sum of harmonic powers) / fundamental power`
- Expressed in dB (negative values, e.g., -60 dB is good)
- Lower THD = less distortion

### SFDR (Spurious-Free Dynamic Range)

Ratio of fundamental to largest spurious tone (any unwanted component):
- Measures worst-case spur
- Higher SFDR = cleaner spectrum
- Typically 60-100 dB for good converters

## Parameter Guidelines

### `spectrum_nperseg`

Segment length for Welch's method:
- **Default**: `len(signal) // 8`
- **Larger** = better frequency resolution, but more variance
- **Smaller** = less variance, but coarser frequency resolution
- **Recommendation**: Use 8192 or 16384 for typical simulations

### `spectrum_max_freq_hz`

Maximum frequency to display in plots:
- **Default**: 20× cutoff frequency
- Use to zoom in on region of interest
- Example: Set to 300 kHz to see up to 300 kHz

### `spectrum_remove_transient_fraction`

Fraction of signal to discard as transient:
- **Default**: 0.2 (remove first 20%)
- Delta-sigma modulators need settling time
- **Recommendation**: 0.15 to 0.25 depending on modulator order

### `spectrum_db_ref`

Full-scale reference for dBFS:
- **Default**: `None` (uses dB/Hz, absolute)
- Set to `signal_amplitude` for dBFS/Hz (relative to full scale)
- dBFS is useful when comparing signals of different amplitudes

## Configuration Comparison Best Practices

When comparing different configurations (word lengths, OSR, modulator order):

1. **Use the same test signal**: Same frequency, amplitude, and duration
2. **Use SNDR from PSD**: Most reliable metric for comparison
3. **Remove transients**: Always use `spectrum_remove_transient_fraction`
4. **Check convergence**: Ensure `nperseg` is large enough for stable results
5. **In-band only**: Compare metrics within the reconstruction band (up to cutoff)

Example comparison:

```python
configs = [
    {'input_word_length_bits': 12, 'label': '12-bit'},
    {'input_word_length_bits': 16, 'label': '16-bit'},
    {'input_word_length_bits': 20, 'label': '20-bit'},
]

for config in configs:
    results = run_single_simulation(
        modulator_order=2,
        oversampling_ratio=2048,
        signal_frequency_hz=10000.0,
        signal_amplitude=0.4,
        input_word_length_bits=config['input_word_length_bits'],
        filter_cutoff_frequency_hz=15000.0,
        number_of_samples=655360,
        plot_results=False,
        spectrum_method='welch',
        spectrum_nperseg=8192
    )
    
    metrics = results['psd_metrics']
    print(f"{config['label']:10s}: "
          f"SNDR = {metrics['sndr_db']:6.2f} dB, "
          f"ENOB = {metrics['enob']:5.2f} bits")
```

## Legacy FFT Method

The original FFT-based spectrum plot is still available:

```python
results = run_single_simulation(
    # ... parameters ...
    spectrum_method='fft',  # Use legacy FFT method
    plot_results=True
)
```

Or call directly:

```python
DeltaSigmaPlotter.plot_frequency_spectrum(
    signal=signal,
    sampling_frequency_hz=fs,
    signal_frequency_hz=10000.0,
    cutoff_frequency_hz=15000.0
)
```

**Note**: The y-axis label now correctly states "FFT Power (dB, unnormalized)" instead of "Power Spectral Density (dB)".

## References

- Welch, P. (1967). "The use of fast Fourier transform for the estimation of power spectra"
- scipy.signal.welch documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html
- IEEE Standard for Terminology and Test Methods for Analog-to-Digital Converters (IEEE 1241-2010)
