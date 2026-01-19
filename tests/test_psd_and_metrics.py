"""
Tests for PSD Utilities and Spectrum Metrics
============================================

These tests validate the correct behavior of:
- Welch PSD computation
- PSD-based SNR/SNDR calculations
- Scaling and normalization
"""

import pytest
import numpy as np

# Try to import scipy; skip tests if not available
pytest_plugins = []
scipy_available = True
try:
    from scipy import signal as scipy_signal
except ImportError:
    scipy_available = False

pytestmark = pytest.mark.skipif(not scipy_available, reason="scipy not available")


class TestWelchPSD:
    """Test Welch PSD computation."""
    
    def test_welch_psd_sine_wave_basic(self):
        """Test that Welch PSD correctly identifies a sine wave."""
        from metrics.psd_utils import compute_welch_psd
        
        # Generate a clean sine wave
        fs = 10000.0  # 10 kHz sampling
        f0 = 1000.0   # 1 kHz signal
        duration = 1.0
        t = np.arange(0, duration, 1/fs)
        amplitude = 1.0
        signal = amplitude * np.sin(2 * np.pi * f0 * t)
        
        # Compute PSD
        frequencies, psd = compute_welch_psd(
            signal_data=signal,
            sampling_frequency_hz=fs,
            nperseg=1024,
            window='hann',
            remove_dc=True
        )
        
        # Find peak frequency
        peak_idx = np.argmax(psd)
        peak_freq = frequencies[peak_idx]
        
        # Peak should be at signal frequency (within frequency resolution)
        freq_resolution = frequencies[1] - frequencies[0]
        assert abs(peak_freq - f0) < freq_resolution * 2, \
            f"Peak at {peak_freq} Hz, expected {f0} Hz"
        
        # PSD should be positive
        assert np.all(psd >= 0), "PSD should be non-negative"
        
        # Peak should be significantly above noise floor
        peak_power = psd[peak_idx]
        median_power = np.median(psd)
        assert peak_power > median_power * 100, \
            "Signal peak should be much higher than noise floor"
    
    def test_welch_psd_scaling(self):
        """Test that PSD scales correctly with signal amplitude."""
        from metrics.psd_utils import compute_welch_psd
        
        fs = 10000.0
        f0 = 1000.0
        duration = 1.0
        t = np.arange(0, duration, 1/fs)
        
        # Test two amplitudes
        amp1 = 1.0
        amp2 = 2.0
        signal1 = amp1 * np.sin(2 * np.pi * f0 * t)
        signal2 = amp2 * np.sin(2 * np.pi * f0 * t)
        
        # Compute PSDs
        freq1, psd1 = compute_welch_psd(signal1, fs, nperseg=1024)
        freq2, psd2 = compute_welch_psd(signal2, fs, nperseg=1024)
        
        # Find peak powers
        peak_power1 = np.max(psd1)
        peak_power2 = np.max(psd2)
        
        # Power should scale as amplitude squared (approximately)
        # PSD units are V^2/Hz, so doubling amplitude should quadruple power
        expected_ratio = (amp2 / amp1) ** 2
        actual_ratio = peak_power2 / peak_power1
        
        # Allow 10% tolerance due to windowing effects
        assert abs(actual_ratio - expected_ratio) / expected_ratio < 0.1, \
            f"Power ratio {actual_ratio:.2f}, expected {expected_ratio:.2f}"
    
    def test_welch_psd_sine_plus_noise(self):
        """Test PSD computation on sine wave with additive noise."""
        from metrics.psd_utils import compute_welch_psd
        
        fs = 10000.0
        f0 = 1000.0
        duration = 2.0
        t = np.arange(0, duration, 1/fs)
        
        # Signal with SNR ~ 20 dB
        signal_amplitude = 1.0
        noise_std = signal_amplitude / 10.0  # SNR ~ 20 dB
        
        np.random.seed(42)
        signal = signal_amplitude * np.sin(2 * np.pi * f0 * t)
        noise = np.random.normal(0, noise_std, len(t))
        noisy_signal = signal + noise
        
        # Compute PSD
        frequencies, psd = compute_welch_psd(
            noisy_signal, fs, nperseg=1024, window='hann'
        )
        
        # Find signal bin
        peak_idx = np.argmax(psd)
        peak_freq = frequencies[peak_idx]
        freq_resolution = frequencies[1] - frequencies[0]
        
        # Peak should still be at signal frequency
        assert abs(peak_freq - f0) < freq_resolution * 2
        
        # Signal should still be clearly visible above noise
        peak_power = psd[peak_idx]
        # Estimate noise floor from bins far from signal
        noise_bins = (frequencies > 2000) & (frequencies < 4000)
        noise_floor = np.median(psd[noise_bins])
        
        # Signal should be at least 10 dB above noise floor
        snr_psd = 10 * np.log10(peak_power / noise_floor)
        assert snr_psd > 10.0, f"SNR from PSD: {snr_psd:.1f} dB, expected > 10 dB"


class TestSpectrumMetrics:
    """Test spectrum-based metrics computation."""
    
    def test_snr_from_psd_clean_sine(self):
        """Test SNR computation from PSD for a clean sine wave."""
        from metrics.psd_utils import compute_welch_psd
        from metrics.spectrum_metrics import compute_snr_from_psd
        
        fs = 10000.0
        f0 = 1000.0
        band_limit = 2000.0
        duration = 2.0
        t = np.arange(0, duration, 1/fs)
        
        # Clean sine with small numerical noise
        amplitude = 1.0
        signal = amplitude * np.sin(2 * np.pi * f0 * t)
        
        # Compute PSD
        frequencies, psd = compute_welch_psd(signal, fs, nperseg=1024)
        
        # Compute SNR
        snr_db, powers = compute_snr_from_psd(
            frequencies, psd, f0, band_limit
        )
        
        # SNR should be reasonably high for clean signal
        # Note: Welch's method with windowing reduces effective SNR compared to ideal
        # Expect at least 30 dB for a clean sine
        assert snr_db > 30.0, f"SNR {snr_db:.1f} dB, expected > 30 dB for clean sine"
        
        # Signal power should be positive
        assert powers['signal_power'] > 0
        assert powers['noise_power'] >= 0
    
    def test_sndr_from_psd_with_harmonics(self):
        """Test SNDR computation that includes harmonic distortion."""
        from metrics.psd_utils import compute_welch_psd
        from metrics.spectrum_metrics import compute_sndr_from_psd
        
        fs = 10000.0
        f0 = 500.0  # Lower frequency so harmonics are in band
        band_limit = 4000.0
        duration = 2.0
        t = np.arange(0, duration, 1/fs)
        
        # Signal with 2nd harmonic (simulating distortion)
        fundamental = 1.0 * np.sin(2 * np.pi * f0 * t)
        harmonic2 = 0.1 * np.sin(2 * np.pi * 2 * f0 * t)  # 2nd harmonic at -20 dB
        signal = fundamental + harmonic2
        
        # Compute PSD
        frequencies, psd = compute_welch_psd(signal, fs, nperseg=1024)
        
        # Compute SNDR
        sndr_db, enob, powers = compute_sndr_from_psd(
            frequencies, psd, f0, band_limit, num_harmonics=3
        )
        
        # SNDR should be limited by the harmonic (roughly 20 dB)
        # Due to integration over bins, expect value around 15-25 dB
        assert 10 < sndr_db < 30, \
            f"SNDR {sndr_db:.1f} dB, expected ~20 dB due to 2nd harmonic"
        
        # ENOB should be derived from SNDR
        expected_enob = (sndr_db - 1.76) / 6.02
        assert abs(enob - expected_enob) < 0.1
        
        # Distortion power should be non-zero
        assert powers['distortion_power'] > 0
    
    def test_snr_vs_sndr_difference(self):
        """Test that SNR (excludes harmonics) > SNDR (includes harmonics)."""
        from metrics.psd_utils import compute_welch_psd
        from metrics.spectrum_metrics import compute_snr_from_psd, compute_sndr_from_psd
        
        fs = 10000.0
        f0 = 500.0
        band_limit = 4000.0
        duration = 2.0
        t = np.arange(0, duration, 1/fs)
        
        # Signal with strong 3rd harmonic
        fundamental = 1.0 * np.sin(2 * np.pi * f0 * t)
        harmonic3 = 0.2 * np.sin(2 * np.pi * 3 * f0 * t)
        signal = fundamental + harmonic3
        
        frequencies, psd = compute_welch_psd(signal, fs, nperseg=1024)
        
        # Compute both metrics
        snr_db, _ = compute_snr_from_psd(
            frequencies, psd, f0, band_limit, num_harmonics_to_exclude=5
        )
        sndr_db, _, _ = compute_sndr_from_psd(
            frequencies, psd, f0, band_limit, num_harmonics=5
        )
        
        # SNR should be higher because it excludes harmonics
        assert snr_db > sndr_db, \
            f"SNR ({snr_db:.1f} dB) should be > SNDR ({sndr_db:.1f} dB)"
        
        # Difference should be at least a few dB
        assert (snr_db - sndr_db) > 2.0


class TestIntegration:
    """Integration tests with full simulation."""
    
    def test_psd_metrics_integration(self):
        """Test that PSD metrics can be computed from simulation results."""
        # This would require running a full simulation, which might be slow
        # For now, we'll test the API compatibility
        from metrics.psd_utils import compute_welch_psd
        from metrics.spectrum_metrics import compute_all_metrics_from_psd
        
        # Create a simple test signal
        fs = 40960.0  # Similar to OSR=2048, f_sig=10kHz scenario
        f0 = 10000.0
        duration = 1.0
        t = np.arange(0, duration, 1/fs)
        
        signal = np.sin(2 * np.pi * f0 * t)
        
        # Compute PSD
        frequencies, psd = compute_welch_psd(
            signal, fs, nperseg=None, window='hann'
        )
        
        # Compute all metrics
        metrics = compute_all_metrics_from_psd(
            frequencies, psd, f0, band_limit_hz=15000.0, num_harmonics=6
        )
        
        # Check that all expected keys are present
        expected_keys = [
            'snr_db', 'sndr_db', 'enob', 'thd_db', 'sfdr_db',
            'signal_power', 'noise_power', 'distortion_power'
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"
        
        # Check that values are reasonable
        assert metrics['snr_db'] > 40.0, "SNR should be high for clean sine"
        assert metrics['enob'] > 0, "ENOB should be positive"
        assert metrics['signal_power'] > 0, "Signal power should be positive"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
