"""
PSD and single-tone metrics utilities.

Provides:
- Welch PSD (true power spectral density in units^2/Hz)
- Convenience conversion to dB/Hz and optional dBFS/Hz
- In-band single-tone metrics: SNR, SNDR, ENOB (and optional SFDR/THD)

Notes:
- For delta-sigma analysis, you typically:
  - remove transient / settle time
  - remove DC (mean)
  - compute PSD (Welch recommended)
  - compute in-band metrics over [0, band_limit_hz]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Sequence, Dict

import numpy as np

try:
    from scipy import signal as sp_signal
except Exception:  # pragma: no cover
    sp_signal = None


@dataclass(frozen=True)
class WelchPsdConfig:
    window: str = "hann"
    nperseg: int = 16384
    noverlap: Optional[int] = None
    detrend: bool = False
    remove_mean: bool = True


def compute_welch_psd(
    x: np.ndarray,
    fs_hz: float,
    config: WelchPsdConfig = WelchPsdConfig(),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute true one-sided PSD using Welch's method.

    Returns:
        f_hz: frequencies (Hz), one-sided, including DC
        pxx:  PSD values in linear units^2/Hz
    """
    if sp_signal is None:
        raise ImportError(
            "scipy is required for Welch PSD. Install with: pip install scipy"
        )

    x = np.asarray(x, dtype=float)

    if config.remove_mean:
        x = x - np.mean(x)

    noverlap = config.noverlap
    if noverlap is None:
        noverlap = config.nperseg // 2

    detrend = "constant" if config.detrend else False

    f_hz, pxx = sp_signal.welch(
        x,
        fs=fs_hz,
        window=config.window,
        nperseg=config.nperseg,
        noverlap=noverlap,
        detrend=detrend,
        return_onesided=True,
        scaling="density",  # <-- PSD per Hz
    )

    return f_hz, pxx


def psd_to_db_per_hz(pxx: np.ndarray) -> np.ndarray:
    """Convert linear PSD (units^2/Hz) to dB/Hz."""
    pxx = np.asarray(pxx, dtype=float)
    return 10.0 * np.log10(pxx + 1e-300)


def psd_to_dbfs_per_hz(pxx: np.ndarray, full_scale_peak: float = 1.0) -> np.ndarray:
    """
    Convert linear PSD to dBFS/Hz assuming full_scale_peak is the peak full-scale amplitude.

    If your signals are normalized to [-1, 1], full_scale_peak=1.0 is typical.

    dBFS convention here:
      0 dBFS corresponds to a full-scale sine wave RMS power: (A^2)/2 where A=full_scale_peak.
      To express PSD in dBFS/Hz, we divide by that RMS power before log.
    """
    fs_rms_power = (full_scale_peak**2) / 2.0
    return 10.0 * np.log10((pxx / fs_rms_power) + 1e-300)


@dataclass(frozen=True)
class ToneMetricsConfig:
    band_limit_hz: float = 15000.0
    search_hz: float = 200.0  # search ± around expected tone
    tone_bins_half_width: int = 2  # integrate tone over ± this many bins
    harmonic_count: int = 5
    harmonic_bins_half_width: int = 2


@dataclass(frozen=True)
class ToneMetrics:
    f0_est_hz: float
    snr_db: float
    sndr_db: float
    enob_bits: float
    signal_power: float
    noise_power: float
    distortion_power: float
    inband_power: float


def _band_mask(f_hz: np.ndarray, band_limit_hz: float) -> np.ndarray:
    return (f_hz >= 0.0) & (f_hz <= band_limit_hz)


def _nearest_bin(f_hz: np.ndarray, f_target: float) -> int:
    return int(np.argmin(np.abs(f_hz - f_target)))


def compute_single_tone_metrics_from_psd(
    f_hz: np.ndarray,
    pxx: np.ndarray,
    fs_hz: float,
    f0_expected_hz: float,
    cfg: ToneMetricsConfig,
) -> ToneMetrics:
    """
    Compute in-band single-tone metrics from a one-sided PSD.

    Method:
    - Integrate total in-band power: sum(PSD * df)
    - Find the tone bin near expected f0 (search window)
    - Integrate tone power over ±tone_bins_half_width bins around peak
    - Optionally integrate harmonic powers (2f0, 3f0, ...) as distortion
    - Define:
        noise_power = inband_power - tone_power - distortion_power
        SNR  = tone_power / noise_power
        SNDR = tone_power / (noise_power + distortion_power)
    """
    f_hz = np.asarray(f_hz)
    pxx = np.asarray(pxx)

    if len(f_hz) != len(pxx):
        raise ValueError("f_hz and pxx must have same length")

    # Frequency resolution (Welch returns uniform spacing)
    df = float(np.median(np.diff(f_hz)))

    band = _band_mask(f_hz, cfg.band_limit_hz)
    f_band = f_hz[band]
    p_band = pxx[band]

    inband_power = float(np.sum(p_band) * df)

    # Find fundamental near expected
    search = (f_band >= (f0_expected_hz - cfg.search_hz)) & (
        f_band <= (f0_expected_hz + cfg.search_hz)
    )
    if not np.any(search):
        raise ValueError("Search window for tone does not overlap band.")

    idx_peak_local = int(np.argmax(p_band[search]))
    idx_peak = np.where(band)[0][np.where(search)[0][idx_peak_local]]
    f0_est = float(f_hz[idx_peak])

    def integrate_bins(center_idx: int, half_width: int) -> float:
        lo = max(0, center_idx - half_width)
        hi = min(len(pxx) - 1, center_idx + half_width)
        return float(np.sum(pxx[lo : hi + 1]) * df)

    signal_power = integrate_bins(idx_peak, cfg.tone_bins_half_width)

    # Harmonics as distortion (stay within band and below Nyquist)
    distortion_power = 0.0
    for k in range(2, cfg.harmonic_count + 1):
        fk = k * f0_est
        if fk <= 0:
            continue
        if fk > cfg.band_limit_hz:
            break
        if fk >= fs_hz / 2.0:
            break
        idxk = _nearest_bin(f_hz, fk)
        distortion_power += integrate_bins(idxk, cfg.harmonic_bins_half_width)

    # Noise is the remaining in-band power (clip at small positive)
    noise_power = max(inband_power - signal_power - distortion_power, 1e-300)

    snr_db = 10.0 * np.log10(signal_power / noise_power + 1e-300)
    sndr_db = 10.0 * np.log10(signal_power / (noise_power + distortion_power) + 1e-300)
    enob = (sndr_db - 1.76) / 6.02

    return ToneMetrics(
        f0_est_hz=f0_est,
        snr_db=snr_db,
        sndr_db=sndr_db,
        enob_bits=enob,
        signal_power=signal_power,
        noise_power=noise_power,
        distortion_power=distortion_power,
        inband_power=inband_power,
    )