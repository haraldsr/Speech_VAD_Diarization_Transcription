"""
Example usage of the conversation VAD labeling pipeline.

This script demonstrates different configurations for processing conversation
recordings. Modify the example functions or create your own based on these
templates.

Usage:
    python conversation_pipeline.py
"""

from __future__ import annotations

import os
import time

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, rely on system environment variables

from speech_vad_diarization_transcription import (
    process_conversation,
)

# Optional: CarbonTracker for energy monitoring
try:
    from carbontracker.tracker import CarbonTracker

    CARBONTRACKER_AVAILABLE = True
except ImportError:
    CARBONTRACKER_AVAILABLE = False


# ============================================================================
# CONFIGURATION
# ============================================================================

ENABLE_CARBON_TRACKING = False  # Set to True to enable energy/emissions tracking


# ============================================================================
# EXAMPLE CONFIGURATIONS
# ============================================================================


def example_dyad() -> dict:
    """
    Dyad example: Two speakers with separate audio files.

    Use this when each speaker has their own microphone/recording.
    VAD options: 'silero' (neural, recommended) or 'rvad' (energy-based, faster)
    """
    return {
        "speakers_audio": {
            "P1": "path/to/speaker1.wav",
            "P2": "path/to/speaker2.wav",
        },
        "output_dir": "outputs/dyad",
        "vad_type": "rvad",
    }


def example_triad() -> dict:
    """
    Triad example: Three speakers with separate audio files.

    Same as dyad but with three participants.
    """
    return {
        "speakers_audio": {
            "P1": "path/to/speaker1.wav",
            "P2": "path/to/speaker2.wav",
            "P3": "path/to/speaker3.wav",
        },
        "output_dir": "outputs/triad",
        "vad_type": "rvad",
    }


def example_diarization() -> dict:
    """
    Diarization example: Single mixed audio file with multiple speakers.

    Use this when you have one recording with all speakers mixed together.
    Requires pyannote and a HuggingFace token with access to pyannote models.
    """
    return {
        "speakers_audio": "examples/coral/conv_0cbf895a2078529eb4a9d8b212e710c9.wav",
        "output_dir": "outputs/diarized",
        "vad_type": "pyannote",
        "auth_token": os.environ.get("HF_TOKEN"),
        "skip_vad_if_exists": True,
    }


def example_custom_whisper() -> dict:
    """
    Example with custom Whisper model and language settings.

    Demonstrates using a fine-tuned Whisper model for specific languages.
    """
    return {
        "speakers_audio": {
            "P1": "path/to/speaker1.wav",
            "P2": "path/to/speaker2.wav",
        },
        "output_dir": "outputs/custom",
        "vad_type": "silero",
        # Custom transcription settings
        "transcription_model_name": "openai/whisper-large-v3",  # or a fine-tuned model
        "whisper_language": "da",  # Danish
        "whisper_device": "cuda",  # Force GPU
    }


def example_cpu_only() -> dict:
    """
    CPU-only example for systems without GPU.

    Uses smaller batch sizes and CPU device for transcription.
    """
    return {
        "speakers_audio": {
            "P1": "path/to/speaker1.wav",
            "P2": "path/to/speaker2.wav",
        },
        "output_dir": "outputs/cpu",
        "vad_type": "rvad",  # rvad is fastest on CPU
        "whisper_device": "cpu",
        "batch_size": 15.0,  # Smaller batches for CPU
        "whisper_model_batch_size": 16,
    }


def example_full_options() -> dict:
    """
    Example showing all available options with their defaults.

    Modify these values based on your needs.
    """
    return {
        # Input/Output
        "speakers_audio": {
            "P1": "path/to/speaker1.wav",
            "P2": "path/to/speaker2.wav",
        },
        "output_dir": "outputs/full",
        # VAD settings
        "vad_type": "silero",  # 'rvad', 'silero', or 'pyannote'
        "auth_token": None,  # Required for pyannote
        "vad_min_duration": 0.07,  # Minimum segment duration (seconds)
        # Energy filtering
        "energy_margin_db": 10.0,  # dB threshold for filtering low-energy segments
        "interactive_energy_filter": False,  # Set True to manually adjust threshold
        # Turn merging
        "gap_thresh": 0.5,  # Max gap to merge segments from same speaker
        "short_utt_thresh": 1.0,  # Threshold for short utterances
        "window_sec": 3.0,  # Look-ahead window for merging
        "merge_short_after_long": True,
        "merge_long_after_short": True,
        "long_merge_enabled": True,
        "merge_max_dur": 60.0,  # Maximum merged turn duration
        "bridge_short_opponent": True,  # Bridge over short opponent utterances
        # Transcription
        "transcription_model_name": "openai/whisper-large-v3",
        "whisper_device": "auto",  # 'auto', 'cuda', or 'cpu'
        "whisper_language": "da",
        "whisper_model_batch_size": 100,
        "batch_size": 30.0,  # Seconds per batch
        # Classification
        "entropy_threshold": 1.5,  # Threshold for backchannel vs turn
        "max_backchannel_dur": 1.0,
        "max_gap_sec": 3.0,
        # Caching
        "skip_vad_if_exists": True,
        "skip_transcription_if_exists": True,
        # Export
        "export_elan": True,  # Export tab-delimited file for annotation software
    }


def example_with_preprocessing() -> dict:
    """
    Example with audio preprocessing for noisy recordings.

    Demonstrates using PreprocessConfig for single and dual-mode preprocessing.
    Useful for improving transcription accuracy on poor-quality audio.

    Preprocessing stages:
    - High-pass filter: Removes DC offset and low-frequency rumble (<60 Hz)
    - Noise reduction: Reduces stationary background noise
    - Loudness normalization: Standardizes loudness (EBU R 128 standard)
    - Peak limiter: Prevents digital clipping

    For dual-mode preprocessing (recommended):
    - Mild profile: High-pass filter only, preserves speaker timbre for VAD/diarization
    - Strong profile: Full processing (HPF + NR + loudness norm + peak limit) for ASR
    """
    from speech_vad_diarization_transcription import PreprocessConfig

    # Option 1: Dual-mode preprocessing (recommended for most cases)
    # Mild version preserves voice characteristics for speaker separation
    preprocess_config_mild = PreprocessConfig(
        enabled=True,
        highpass=True,
        highpass_freq=60.0,
        noise_reduce=False,  # Don't reduce noise for VAD — preserves speaker timbre
        loudness_norm=False,  # Don't normalize loudness for VAD
        peak_limit=False,
    )

    # Strong version maximizes clarity for transcription
    preprocess_config_strong = PreprocessConfig(
        enabled=True,
        highpass=True,
        highpass_freq=60.0,
        noise_reduce=True,
        noise_reduce_stationary=True,
        noise_reduce_prop_decrease=0.8,  # 80% reduction strength
        loudness_norm=True,
        target_lufs=-23.0,  # EBU R 128 standard
        auto_loudness=True,  # Skip if already within tolerance
        loudness_tolerance_db=3.0,
        peak_limit=True,
        peak_ceiling=0.95,  # Prevent clipping
    )

    return {
        # Input/Output
        "speakers_audio": {
            "P1": "path/to/speaker1.wav",
            "P2": "path/to/speaker2.wav",
        },
        "output_dir": "outputs/with_preprocessing",
        # VAD settings
        "vad_type": "silero",
        # Preprocessing (dual-mode)
        "preprocess_config_mild": preprocess_config_mild,
        "preprocess_config_strong": preprocess_config_strong,
        # Transcription
        "transcription_model_name": "openai/whisper-large-v3",
        "whisper_device": "cuda",
        "whisper_language": "da",
    }


def example_single_preprocessing() -> dict:
    """
    Example with single-mode preprocessing for VAD or balanced processing.

    Use this when you want the same preprocessing applied to all stages
    (VAD/diarization AND transcription), rather than separate mild/strong versions.

    Choose preset profiles by setting only the needed fields:
    - 'vad': High-pass filter only (minimal processing)
    - 'clean': Clean audio — HPF + mild loudness norm + peak limit
    - 'moderate': Typical audio — HPF + noise reduction + loudness norm + peak limit
    - 'noisy': Noisy audio — aggressive HPF + strong NR + loudness norm + peak limit
    """
    from speech_vad_diarization_transcription import PreprocessConfig

    # Custom single-mode preprocessing for moderate noise levels
    preprocess_config = PreprocessConfig(
        enabled=True,
        highpass=True,
        highpass_freq=60.0,
        noise_reduce=True,
        noise_reduce_stationary=True,
        noise_reduce_prop_decrease=0.7,  # 70% reduction (moderate)
        loudness_norm=True,
        target_lufs=-23.0,
        auto_loudness=True,
        loudness_tolerance_db=3.0,
        peak_limit=True,
        peak_ceiling=0.95,
    )

    return {
        "speakers_audio": {
            "P1": "path/to/speaker1.wav",
            "P2": "path/to/speaker2.wav",
        },
        "output_dir": "outputs/single_preprocessing",
        "vad_type": "silero",
        # Single-mode preprocessing (same for VAD and ASR)
        "preprocess_config": preprocess_config,
        "transcription_model_name": "openai/whisper-large-v3",
        "whisper_device": "cuda",
        "whisper_language": "da",
    }


# ============================================================================
# CARBON TRACKING HELPERS
# ============================================================================


def create_carbon_tracker() -> CarbonTracker | None:
    """Create and configure CarbonTracker if enabled and available."""
    if not ENABLE_CARBON_TRACKING:
        return None

    if not CARBONTRACKER_AVAILABLE:
        print("CarbonTracker not installed. Run: pip install carbontracker")
        return None

    api_key = os.environ.get("ELECTRICITYMAPS_API_KEY")
    tracker_kwargs = {
        "epochs": 1,
        "monitor_epochs": 1,
        "log_dir": "logs/carbon",
        "decimal_precision": 3,
        "ignore_errors": True,
        # Simulation settings for systems without direct power measurement
        # Adjust these values for your hardware
        "sim_cpu": "AMD EPYC 7302",
        "sim_cpu_tdp": 20,  # Estimated TDP in Watts
        "sim_cpu_util": 0.2,  # Estimated utilization (0-1)
    }

    if api_key:
        tracker_kwargs["api_keys"] = {"electricitymaps": api_key}
    else:
        print(
            "Note: ELECTRICITYMAPS_API_KEY not set. "
            "Carbon tracking will run without CO2 intensity data."
        )

    return CarbonTracker(**tracker_kwargs)


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    """Run the pipeline with the selected example configuration."""
    # -------------------------------------------------------------------------
    # SELECT YOUR EXAMPLE HERE
    # -------------------------------------------------------------------------
    config = example_dyad()
    # config = example_triad()
    # config = example_diarization()
    # config = example_custom_whisper()
    # config = example_cpu_only()
    # config = example_full_options()
    # config = example_with_preprocessing()  # Dual-mode preprocessing (recommended for noisy audio)
    # config = example_single_preprocessing()  # Single-mode preprocessing

    # -------------------------------------------------------------------------
    # RUN PIPELINE
    # -------------------------------------------------------------------------
    tracker = create_carbon_tracker()
    if tracker:
        tracker.epoch_start()

    start_time = time.time()

    results = process_conversation(**config)

    elapsed = time.time() - start_time

    if tracker:
        tracker.epoch_end()
        tracker.stop()

    print(f"\nProcessing completed in {elapsed:.2f} seconds")
    print(f"Output saved to: {results['output_dir']}")

    # -------------------------------------------------------------------------
    # COMPARE TO MANUAL ANNOTATIONS IF AVAILABLE
    # -------------------------------------------------------------------------
    # if len(label_dir) > 0:
    #     compute_and_print_errors(label_dir, conv_id, annotator_id=annotator_id)


if __name__ == "__main__":
    main()
