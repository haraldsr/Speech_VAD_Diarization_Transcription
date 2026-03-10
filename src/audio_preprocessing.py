"""
Audio preprocessing / speech enhancement for the conversation pipeline.

Applies adaptive signal conditioning to improve downstream VAD and ASR
quality.  Each step is gated by thresholds derived from the input audio
itself, so the processing adapts to the recording conditions.

Pipeline (in order):
  1. High-pass filter   – removes DC offset and low-frequency rumble
  2. Spectral gating    – reduces stationary background noise
  3. Loudness normalisation – targets -23 LUFS (EBU R 128)
  4. Peak limiter       – hard-clips to prevent clipping after gain

All steps are optional and controlled by the ``AudioPreprocessor``
configuration.  A single call to ``preprocess_audio()`` runs the full
chain and writes a cleaned WAV to disk.

Requires: scipy, soundfile.
Optional: pyloudnorm (for LUFS targeting), librosa (for spectral gating),
          noisereduce (alternative spectral gating).
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast

import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Optional imports — degrade gracefully when libs are missing
# ---------------------------------------------------------------------------
try:
    from scipy.signal import butter, sosfilt

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    import pyloudnorm as pyln

    _HAS_PYLOUDNORM = True
except ImportError:
    _HAS_PYLOUDNORM = False

try:
    import librosa

    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False

try:
    import noisereduce as nr

    _HAS_NOISEREDUCE = True
except ImportError:
    _HAS_NOISEREDUCE = False


def load_audio(path: str) -> Tuple[np.ndarray, int]:
    """Load audio from any supported format into a float32 numpy array.

    Tries soundfile first (WAV, FLAC, OGG, AIFF).  Falls back to
    torchaudio for compressed formats (MP3, M4A, AAC, OPUS, etc.).
    """
    try:
        audio, sr = sf.read(path, dtype="float32")
        return cast(np.ndarray, audio), int(sr)
    except Exception:
        pass

    try:
        import torchaudio  # already in requirements

        waveform, sr = torchaudio.load(path)
        audio = waveform.numpy()
        if audio.ndim == 2:
            if audio.shape[0] == 1:
                audio = audio[0]  # mono: (1, T) → (T,)
            else:
                audio = audio.T  # stereo: (C, T) → (T, C)
        return audio.astype(np.float32), int(sr)
    except Exception as _e:
        ext = Path(path).suffix.lower()
        raise RuntimeError(
            f"Cannot load audio '{path}' (format: {ext}). "
            f"Supported: WAV/FLAC/OGG/AIFF via soundfile; "
            f"MP3/M4A/AAC/OPUS via torchaudio+ffmpeg. Error: {_e}"
        ) from _e


# ===================================================================
# Configuration
# ===================================================================


@dataclass
class PreprocessConfig:
    """Configuration for audio preprocessing.

    Each flag can be toggled independently.  When ``auto_*`` flags are
    True, the corresponding step measures a property of the input signal
    and only applies correction if needed.

    Attributes:
        enabled: Master switch — when False, preprocessing is skipped
            entirely and the original path is returned.
        highpass: Apply a high-pass filter to remove DC offset and rumble.
        highpass_freq: Cut-off frequency in Hz for the high-pass filter.
        noise_reduce: Apply spectral-gating noise reduction.
        noise_reduce_stationary: Assume stationary noise (faster, simpler).
            When False, uses time-frequency masking (better for
            non-stationary noise but slower).
        noise_reduce_prop_decrease: Proportion by which noise is reduced
            (0.0–1.0).  Lower values are gentler.
        loudness_norm: Normalise loudness to a target LUFS value.
        target_lufs: Target loudness in LUFS (default: -23, EBU R 128).
        auto_loudness: Only normalise if measured loudness deviates from
            target by more than ``loudness_tolerance_db``.
        loudness_tolerance_db: Tolerance in dB around ``target_lufs`` —
            normalisation is skipped if the input is already within range.
        peak_limit: Hard-limit peaks to ``peak_ceiling`` after gain.
        peak_ceiling: Maximum absolute sample value (default: 0.95 to
            leave ~0.5 dB headroom).
        output_subdir: Sub-directory name inside the pipeline output dir
            where preprocessed files are written.  Set to ``""`` to write
            alongside the originals.
        output_suffix: Suffix appended to the filename stem.
        sample_rate: If not None, resample audio to this rate (Hz).
            When None the original sample rate is kept.
    """

    enabled: bool = True
    # --- high-pass filter ---
    highpass: bool = True
    highpass_freq: float = 60.0
    # --- noise reduction ---
    noise_reduce: bool = True
    noise_reduce_stationary: bool = True
    noise_reduce_prop_decrease: float = 0.8
    # --- loudness ---
    loudness_norm: bool = True
    target_lufs: float = -23.0
    auto_loudness: bool = True
    loudness_tolerance_db: float = 3.0
    max_loudness_gain_db: float = 20.0  # skip norm if gain required exceeds this
    # --- peak limiting ---
    peak_limit: bool = True
    peak_ceiling: float = 0.95
    # --- output ---
    output_subdir: str = "preprocessed"
    output_suffix: str = "_enhanced"
    # --- resampling ---
    sample_rate: Optional[int] = None


# ---------------------------------------------------------------------------
# Auto-profile presets
# ---------------------------------------------------------------------------

# Named profiles that auto_profile() selects from based on audio stats.
# Each is a partial dict of PreprocessConfig fields to override.

_PROFILES: Dict[str, Dict[str, Any]] = {
    "clean": {
        # Already good quality — only HPF to remove DC/rumble
        "highpass": True,
        "noise_reduce": False,
        "loudness_norm": True,
        "loudness_tolerance_db": 5.0,
        "peak_limit": True,
        "noise_reduce_prop_decrease": 0.5,
    },
    "moderate": {
        # Typical recording — standard processing
        "highpass": True,
        "noise_reduce": True,
        "noise_reduce_stationary": True,
        "noise_reduce_prop_decrease": 0.7,
        "loudness_norm": True,
        "loudness_tolerance_db": 3.0,
        "peak_limit": True,
    },
    "noisy": {
        # Poor recording — aggressive enhancement
        "highpass": True,
        "highpass_freq": 80.0,
        "noise_reduce": True,
        "noise_reduce_stationary": True,
        "noise_reduce_prop_decrease": 0.95,
        "loudness_norm": True,
        "loudness_tolerance_db": 1.0,
        "peak_limit": True,
        "peak_ceiling": 0.90,
    },
}


def auto_profile(
    audio: np.ndarray,
    sr: int,
    *,
    quiet_lufs_threshold: float = -35.0,
    noisy_snr_threshold: float = 18.0,
    clean_lufs_threshold: float = -26.0,
    clean_snr_threshold: float = 25.0,
) -> Tuple[str, PreprocessConfig]:
    """Select a preprocessing profile automatically from audio statistics.

    Analyses the input signal and returns one of three presets:

    - **clean** — LUFS > ``clean_lufs_threshold`` *and*
      SNR > ``clean_snr_threshold``.  Minimal processing.
    - **noisy** — LUFS < ``quiet_lufs_threshold`` *or*
      SNR < ``noisy_snr_threshold``.  Aggressive enhancement.
    - **moderate** — everything else.  Standard processing.

    Args:
        audio: Audio samples (float32/64, mono or multi-channel).
        sr: Sample rate in Hz.
        quiet_lufs_threshold: Below this LUFS the audio is considered
            very quiet (→ noisy profile).
        noisy_snr_threshold: Below this estimated SNR the audio is
            considered noisy.
        clean_lufs_threshold: Above this LUFS the audio *may* be clean.
        clean_snr_threshold: Above this SNR the audio is considered clean
            (together with LUFS condition).

    Returns:
        Tuple of ``(profile_name, PreprocessConfig)`` ready to use.
    """
    stats = analyse_audio(audio, sr)
    lufs = stats.get("lufs")
    snr = stats.get("snr_estimate_db")

    # Default to moderate when stats are unavailable
    if lufs is None:
        lufs = -30.0
    if snr is None:
        snr = 20.0

    if lufs < quiet_lufs_threshold or snr < noisy_snr_threshold:
        profile_name = "noisy"
    elif lufs > clean_lufs_threshold and snr > clean_snr_threshold:
        profile_name = "clean"
    else:
        profile_name = "moderate"

    cfg = PreprocessConfig(**_PROFILES[profile_name])
    return profile_name, cfg


def dual_preprocess(
    audio_path: str,
    output_dir: str,
    *,
    config_mild: Optional[PreprocessConfig] = None,
    config_strong: Optional[PreprocessConfig] = None,
    auto: bool = True,
    verbose: bool = True,
) -> Dict[str, str]:
    """Produce two preprocessed versions: mild (for diarization) and strong (for ASR).

    Diarization is sensitive to artifacts introduced by heavy noise
    reduction, so a gentler version works better.  ASR benefits from
    cleaner, louder audio.

    When ``auto=True`` the mild / strong configs are derived automatically
    from the input audio's statistics.

    Args:
        audio_path: Path to the input audio file.
        output_dir: Pipeline output directory.
        config_mild: Override config for mild processing.
        config_strong: Override config for strong processing.
        auto: Auto-select profiles from audio stats when configs are
            not provided.
        verbose: Print diagnostics.

    Returns:
        Dict with keys ``'mild'`` and ``'strong'``, each mapping to
        the path of the preprocessed file.
    """
    audio, sr = load_audio(audio_path)

    if auto:
        profile_name, _ = auto_profile(audio, sr)
        if verbose:
            print(f"  Auto-detected profile: {profile_name}")
    else:
        profile_name = "moderate"

    # Mild: for diarization — less aggressive noise reduction
    if config_mild is None:
        config_mild = PreprocessConfig(
            **{
                **_PROFILES.get(
                    "clean" if profile_name == "clean" else "moderate",
                    _PROFILES["moderate"],
                ),
                "output_suffix": "_mild",
            }
        )
        # Ensure noise reduction is gentle even in moderate profile
        config_mild.noise_reduce_prop_decrease = min(
            config_mild.noise_reduce_prop_decrease, 0.5
        )

    # Strong: for ASR — prioritise clarity
    if config_strong is None:
        cfg_key = profile_name if profile_name in ("moderate", "noisy") else "moderate"
        config_strong = PreprocessConfig(
            **{**_PROFILES[cfg_key], "output_suffix": "_strong"}
        )
        # Push noise reduction harder for ASR
        config_strong.noise_reduce = True
        config_strong.noise_reduce_prop_decrease = max(
            config_strong.noise_reduce_prop_decrease, 0.85
        )

    mild_path = preprocess_audio(
        audio_path, output_dir, config=config_mild, verbose=verbose
    )
    strong_path = preprocess_audio(
        audio_path, output_dir, config=config_strong, verbose=verbose
    )

    return {"mild": mild_path, "strong": strong_path}


# ===================================================================
# Individual processing steps
# ===================================================================


def _apply_highpass(
    audio: np.ndarray, sr: int, cutoff_hz: float = 60.0, order: int = 5
) -> np.ndarray:
    """Apply a Butterworth high-pass filter.

    Removes DC offset and low-frequency rumble below ``cutoff_hz``.
    """
    if not _HAS_SCIPY:
        warnings.warn(
            "scipy not available — skipping high-pass filter.  "
            "Install with: pip install scipy"
        )
        return audio
    nyq = sr / 2.0
    if cutoff_hz >= nyq:
        return audio
    sos = butter(order, cutoff_hz / nyq, btype="high", output="sos")
    # Apply along time axis; handle mono and multi-channel
    if audio.ndim == 1:
        return cast(np.ndarray, sosfilt(sos, audio).astype(audio.dtype))
    else:
        return cast(
            np.ndarray,
            np.stack(
                [
                    sosfilt(sos, audio[:, ch]).astype(audio.dtype)
                    for ch in range(audio.shape[1])
                ],
                axis=1,
            ),
        )


def _apply_noise_reduction(
    audio: np.ndarray,
    sr: int,
    stationary: bool = True,
    prop_decrease: float = 0.8,
) -> np.ndarray:
    """Reduce background noise via spectral gating.

    Uses ``noisereduce`` if available; otherwise falls back to a simple
    spectral-gating implementation using ``librosa``.
    """
    if _HAS_NOISEREDUCE:
        return cast(
            np.ndarray,
            nr.reduce_noise(
                y=audio,
                sr=sr,
                stationary=stationary,
                prop_decrease=prop_decrease,
            ),
        )

    if _HAS_LIBROSA:
        return _spectral_gate_librosa(audio, sr, prop_decrease)

    warnings.warn(
        "Neither noisereduce nor librosa available — skipping noise "
        "reduction.  Install with: pip install noisereduce"
    )
    return audio


def _spectral_gate_librosa(
    audio: np.ndarray,
    sr: int,
    prop_decrease: float = 0.8,
    n_fft: int = 2048,
    hop_length: int = 512,
    noise_frames: int = 10,
) -> np.ndarray:
    """Simple spectral gating using librosa.

    Estimates noise profile from the quietest ``noise_frames`` STFT
    frames, then soft-masks the signal.
    """
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    stft = librosa.stft(mono, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(stft), np.angle(stft)

    # Estimate noise from quietest frames (by total energy)
    frame_energy = np.sum(mag**2, axis=0)
    quiet_idx = np.argsort(frame_energy)[:noise_frames]
    noise_profile = np.mean(mag[:, quiet_idx], axis=1, keepdims=True)

    # Soft mask: reduce magnitude where it's close to noise floor
    mask = np.clip((mag - noise_profile * prop_decrease) / (mag + 1e-10), 0.0, 1.0)
    cleaned_stft = mag * mask * np.exp(1j * phase)
    cleaned = librosa.istft(cleaned_stft, hop_length=hop_length, length=len(mono))

    # If original was multi-channel, apply gain ratio to all channels
    if audio.ndim > 1:
        gain = np.divide(
            cleaned, mono + 1e-10, out=np.ones_like(mono), where=np.abs(mono) > 1e-10
        )
        return cast(np.ndarray, (audio * gain[:, np.newaxis]).astype(audio.dtype))
    return cast(np.ndarray, cleaned.astype(audio.dtype))


def _apply_loudness_norm(
    audio: np.ndarray,
    sr: int,
    target_lufs: float = -23.0,
    auto: bool = True,
    tolerance_db: float = 3.0,
    max_gain_db: float = 20.0,
) -> np.ndarray:
    """Normalise integrated loudness to ``target_lufs`` (EBU R 128).

    When ``auto=True``, skips normalisation if the signal is already
    within ``tolerance_db`` of the target.

    ``max_gain_db`` caps the maximum gain applied. If the audio is very
    quiet and would require more gain than this threshold to reach the
    target, a partial normalization is applied instead (capping gain to
    ``max_gain_db`` to avoid peak-limiter distortion). A warning is printed.
    """
    if not _HAS_PYLOUDNORM:
        # Fallback: simple RMS normalisation
        return _rms_normalise(audio, target_db=target_lufs, max_gain_db=max_gain_db)

    meter = pyln.Meter(sr)
    # pyloudnorm expects float64 and shape (samples,) or (samples, channels)
    audio_64 = audio.astype(np.float64)
    try:
        current_lufs = meter.integrated_loudness(audio_64)
    except Exception:
        return audio  # very quiet / silence — skip

    if np.isinf(current_lufs) or np.isnan(current_lufs):
        return audio

    if auto and abs(current_lufs - target_lufs) <= tolerance_db:
        return audio  # already within tolerance

    gain_required = target_lufs - current_lufs
    actual_target = target_lufs

    if gain_required > max_gain_db:
        # Cap the gain to prevent distortion
        capped_gain = max_gain_db
        actual_target = current_lufs + capped_gain
        warnings.warn(
            f"Loudness normalization: required gain {gain_required:.1f} dB "
            f"(current {current_lufs:.1f} LUFS → target {target_lufs:.1f} LUFS) "
            f"exceeds max_gain_db={max_gain_db:.0f} dB. "
            f"Applying capped gain of {capped_gain:.1f} dB to reach {actual_target:.1f} LUFS instead. "
            f"Recording levels are very low — consider checking input source."
        )

    return cast(
        np.ndarray,
        pyln.normalize.loudness(audio_64, current_lufs, actual_target).astype(
            audio.dtype
        ),
    )


def _rms_normalise(
    audio: np.ndarray, target_db: float = -23.0, max_gain_db: float = 20.0
) -> np.ndarray:
    """Simple RMS-based gain normalisation (fallback when pyloudnorm is missing).

    When required gain exceeds max_gain_db, applies capped gain up to max_gain_db
    instead of skipping normalization entirely.
    """
    rms = np.sqrt(np.mean(audio**2))
    if rms < 1e-10:
        return audio
    current_db = 20.0 * np.log10(rms)
    gain_db = target_db - current_db

    actual_gain_db = gain_db
    if gain_db > max_gain_db:
        actual_gain_db = max_gain_db
        actual_target_db = current_db + max_gain_db
        warnings.warn(
            f"RMS normalisation: required gain {gain_db:.1f} dB exceeds max_gain_db={max_gain_db:.0f} dB. "
            f"Applying capped gain of {actual_gain_db:.1f} dB to reach {actual_target_db:.1f} dB instead."
        )

    gain = 10.0 ** (actual_gain_db / 20.0)
    return cast(np.ndarray, (audio * gain).astype(audio.dtype))


def _apply_peak_limit(audio: np.ndarray, ceiling: float = 0.95) -> np.ndarray:
    """Hard-clip to ``ceiling`` to prevent digital clipping."""
    return np.clip(audio, -ceiling, ceiling)


def _resample(audio: np.ndarray, sr_orig: int, sr_target: int) -> np.ndarray:
    """Resample audio to ``sr_target`` Hz."""
    if sr_orig == sr_target:
        return audio
    if _HAS_LIBROSA:
        if audio.ndim == 1:
            return librosa.resample(audio, orig_sr=sr_orig, target_sr=sr_target)
        else:
            channels = [
                librosa.resample(audio[:, ch], orig_sr=sr_orig, target_sr=sr_target)
                for ch in range(audio.shape[1])
            ]
            return np.stack(channels, axis=1)
    # Fallback: scipy.signal.resample
    if _HAS_SCIPY:
        from scipy.signal import resample as scipy_resample

        n_samples = int(len(audio) * sr_target / sr_orig)
        if audio.ndim == 1:
            return cast(
                np.ndarray, scipy_resample(audio, n_samples).astype(audio.dtype)
            )
        else:
            return np.stack(
                [
                    scipy_resample(audio[:, ch], n_samples).astype(audio.dtype)
                    for ch in range(audio.shape[1])
                ],
                axis=1,
            )
    warnings.warn("Neither librosa nor scipy available for resampling — skipping.")
    return audio


# ===================================================================
# Diagnostics — measure input audio properties
# ===================================================================


def analyse_audio(audio: np.ndarray, sr: int) -> Dict[str, Any]:
    """Compute diagnostic statistics for an audio signal.

    Returned dict contains:
      - ``duration_sec``
      - ``sample_rate``
      - ``n_samples`` — number of audio samples
      - ``rms_db``, ``peak_db``
      - ``lufs`` (if pyloudnorm available)
      - ``dc_offset``
      - ``clipping_ratio`` (fraction of samples at ±1.0)
      - ``snr_estimate_db`` (rough estimate from quiet frames)
    """
    duration = len(audio) / sr
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)

    rms = np.sqrt(np.mean(mono**2))
    peak: float = np.max(np.abs(mono))
    dc = np.mean(mono)
    clipping = np.mean(np.abs(mono) >= 0.999)

    stats: Dict[str, Any] = {
        "duration_sec": round(duration, 2),
        "sample_rate": sr,
        "n_samples": len(audio),
        "rms_db": round(20 * np.log10(rms + 1e-10), 1),
        "peak_db": round(20 * np.log10(peak + 1e-10), 1),
        "dc_offset": round(float(dc), 6),
        "clipping_ratio": round(float(clipping), 6),
    }

    if _HAS_PYLOUDNORM:
        try:
            meter = pyln.Meter(sr)
            lufs = meter.integrated_loudness(audio.astype(np.float64))
            stats["lufs"] = round(lufs, 1) if np.isfinite(lufs) else None
        except Exception:
            stats["lufs"] = None

    # Rough SNR estimate: ratio of overall RMS to RMS of quietest 10% of frames
    if _HAS_LIBROSA or _HAS_SCIPY:
        frame_len = int(0.025 * sr)  # 25 ms frames
        hop = int(0.010 * sr)  # 10 ms hop
        n_frames = max(1, (len(mono) - frame_len) // hop)
        frame_rms = np.array(
            [
                np.sqrt(np.mean(mono[i * hop : i * hop + frame_len] ** 2))
                for i in range(n_frames)
            ]
        )
        quiet_10 = np.sort(frame_rms)[: max(1, n_frames // 10)]
        noise_rms = np.mean(quiet_10)
        if noise_rms > 1e-10 and rms > 1e-10:
            snr = 20 * np.log10(rms / noise_rms)
            stats["snr_estimate_db"] = round(snr, 1)

    return stats


# ===================================================================
# Main entry point
# ===================================================================


def preprocess_audio(
    audio_path: str,
    output_dir: str,
    config: Optional[PreprocessConfig] = None,
    *,
    verbose: bool = True,
) -> str:
    """Apply audio preprocessing to a single file.

    Args:
        audio_path: Path to the input audio file.
        output_dir: Pipeline output directory (preprocessed file is
            written to ``output_dir / config.output_subdir``).
        config: Preprocessing configuration.  ``None`` uses defaults.
        verbose: Print progress and diagnostics.

    Returns:
        Path to the preprocessed audio file.  If ``config.enabled`` is
        False, returns the original ``audio_path`` unchanged.
    """
    if config is None:
        config = PreprocessConfig()

    if not config.enabled:
        return audio_path

    # ---- Load (supports WAV, FLAC, OGG, MP3, M4A, etc.) ----
    audio, sr = load_audio(audio_path)
    original_sr = sr

    if verbose:
        stats = analyse_audio(audio, sr)
        print(f"  Audio diagnostics ({os.path.basename(audio_path)}):")
        print(f"    Duration       : {stats['duration_sec']}s")
        print(f"    Sample rate    : {stats['sample_rate']} Hz")
        print(f"    RMS / Peak     : {stats['rms_db']} / {stats['peak_db']} dB")
        print(f"    LUFS           : {stats.get('lufs', 'N/A')}")
        print(f"    DC offset      : {stats['dc_offset']}")
        print(f"    Clipping ratio : {stats['clipping_ratio']}")
        print(f"    Est. SNR       : {stats.get('snr_estimate_db', 'N/A')} dB")

    steps_applied = []

    # ---- 1. Resample (if requested) ----
    if config.sample_rate is not None and sr != config.sample_rate:
        audio = _resample(audio, sr, config.sample_rate)
        sr = config.sample_rate
        steps_applied.append(f"resample {original_sr}→{sr} Hz")

    # ---- 2. High-pass filter ----
    if config.highpass:
        audio = _apply_highpass(audio, sr, cutoff_hz=config.highpass_freq)
        steps_applied.append(f"highpass {config.highpass_freq} Hz")

    # ---- 3. Noise reduction ----
    if config.noise_reduce:
        audio = _apply_noise_reduction(
            audio,
            sr,
            stationary=config.noise_reduce_stationary,
            prop_decrease=config.noise_reduce_prop_decrease,
        )
        steps_applied.append("noise_reduce")

    # ---- 4. Loudness normalisation ----
    if config.loudness_norm:
        audio = _apply_loudness_norm(
            audio,
            sr,
            target_lufs=config.target_lufs,
            auto=config.auto_loudness,
            tolerance_db=config.loudness_tolerance_db,
            max_gain_db=config.max_loudness_gain_db,
        )
        steps_applied.append(f"loudness_norm → {config.target_lufs} LUFS")

    # ---- 5. Peak limiter ----
    if config.peak_limit:
        audio = _apply_peak_limit(audio, ceiling=config.peak_ceiling)
        steps_applied.append(f"peak_limit {config.peak_ceiling}")

    # ---- Write output ----
    stem = Path(audio_path).stem
    suffix = config.output_suffix
    out_subdir = (
        os.path.join(output_dir, config.output_subdir)
        if config.output_subdir
        else output_dir
    )
    os.makedirs(out_subdir, exist_ok=True)
    out_path = os.path.join(out_subdir, f"{stem}{suffix}.wav")
    sf.write(out_path, audio, sr, subtype="PCM_16")

    if verbose:
        print(f"  Steps applied    : {', '.join(steps_applied) or 'none'}")
        after_stats = analyse_audio(audio, sr)
        print(
            f"  After: RMS={after_stats['rms_db']} dB  Peak={after_stats['peak_db']} dB"
            f"  LUFS={after_stats.get('lufs', 'N/A')}  SNR={after_stats.get('snr_estimate_db', 'N/A')} dB"
        )
        print(f"  Saved to: {out_path}")

    return out_path


def preprocess_speakers_audio(
    speakers_audio: Dict[str, str],
    output_dir: str,
    config: Optional[PreprocessConfig] = None,
    *,
    verbose: bool = True,
) -> Dict[str, str]:
    """Preprocess all speaker audio files.

    Args:
        speakers_audio: Mapping of speaker name → audio file path.
        output_dir: Pipeline output directory.
        config: Preprocessing configuration.

    Returns:
        New mapping of speaker name → preprocessed audio file path.
    """
    if config is None:
        config = PreprocessConfig()

    if not config.enabled:
        return dict(speakers_audio)

    print("\n0. Audio Preprocessing / Speech Enhancement")
    result = {}
    for speaker, path in speakers_audio.items():
        if verbose:
            print(f"\n  [{speaker}] {os.path.basename(path)}")
        result[speaker] = preprocess_audio(
            path, output_dir, config=config, verbose=verbose
        )

    return result
