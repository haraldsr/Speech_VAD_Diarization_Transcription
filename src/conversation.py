"""
High-level conversation processing pipeline utilities.

Based on Conversational_speech_labeling_pipeline by Hanlu He.

https://github.com/hanlululu/Conversational_speech_labeling_pipeline

"""

from __future__ import annotations

import os
import shutil
from dataclasses import replace as _dc_replace
from typing import Dict, List, Mapping, Optional, Tuple, Union

import pandas as pd

from .audio_preprocessing import (
    _PROFILES,
    PreprocessConfig,
    preprocess_audio,
    preprocess_speakers_audio,
)
from .evaluation import (
    evaluate_pipeline,
    load_reference,
    print_evaluation_summary,
)
from .evaluation_plots import plot_evaluation_results
from .labeling import classify_transcriptions, merge_turns_with_context
from .merge_turns import create_turns_df_windowed
from .postprocess_vad import filter_low_energy_segments
from .transcription import load_whisper_model, transcribe_segments
from .vad import SpeechActivityDetector

EnergyMargin = Union[float, List[float], Tuple[float, ...]]


def _normalise_margins(margins: EnergyMargin, speakers: List[str]) -> List[float]:
    if isinstance(margins, (list, tuple)):
        if len(margins) != len(speakers):
            raise ValueError(
                "energy_margin_db list length does not match number of speakers"
            )
        return [float(value) for value in margins]
    return [float(margins)] * len(speakers)


def _normalise_turn_type(value: object) -> str:
    """Normalise turn labels to canonical values used across outputs."""
    if pd.isna(value):
        return "turn"

    text = str(value).strip().lower()
    if text in {"t", "turn"}:
        return "turn"
    if text in {"b", "backchannel"}:
        return "backchannel"
    if text == "overlapped_turn":
        return "overlapped_turn"
    return text if text else "turn"


def _read_vad_segments(vad_path: str, speaker: str) -> pd.DataFrame:
    """Read one VAD txt file and return standardised segment rows."""
    try:
        df = pd.read_csv(vad_path, sep="\t", comment="#", header=None)
    except Exception:
        return pd.DataFrame(
            columns=[
                "speaker",
                "start_sec",
                "end_sec",
                "duration_sec",
                "type",
                "transcription",
            ]
        )

    if df.shape[1] < 2:
        return pd.DataFrame(
            columns=[
                "speaker",
                "start_sec",
                "end_sec",
                "duration_sec",
                "type",
                "transcription",
            ]
        )

    starts = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    ends = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    if df.shape[1] >= 3:
        raw_labels = df.iloc[:, 2].astype(str)
    else:
        raw_labels = pd.Series(["T"] * len(df))

    valid = starts.notna() & ends.notna() & (ends > starts)
    if not valid.any():
        return pd.DataFrame(
            columns=[
                "speaker",
                "start_sec",
                "end_sec",
                "duration_sec",
                "type",
                "transcription",
            ]
        )

    starts = starts[valid].astype(float)
    ends = ends[valid].astype(float)
    labels = raw_labels[valid].apply(_normalise_turn_type)

    out = pd.DataFrame(
        {
            "speaker": speaker,
            "start_sec": starts.values,
            "end_sec": ends.values,
            "duration_sec": (ends - starts).values,
            "type": labels.values,
            "transcription": "",
        }
    )

    # VAD output is speech activity: keep "turn"-like rows only.
    return out[out["type"] == "turn"].reset_index(drop=True)


def _write_standard_segments(df: pd.DataFrame, output_path: str) -> None:
    """Write canonical segment table with consistent headers and labels."""
    cols = ["speaker", "start_sec", "end_sec", "duration_sec", "type", "transcription"]

    out = df.copy()
    if "start" in out.columns and "start_sec" not in out.columns:
        out["start_sec"] = out["start"]
    if "end" in out.columns and "end_sec" not in out.columns:
        out["end_sec"] = out["end"]
    if "duration" in out.columns and "duration_sec" not in out.columns:
        out["duration_sec"] = out["duration"]
    if "turn_type" in out.columns and "type" not in out.columns:
        out["type"] = out["turn_type"]

    if "type" not in out.columns:
        out["type"] = "turn"
    out["type"] = out["type"].apply(_normalise_turn_type)

    if "transcription" not in out.columns:
        out["transcription"] = ""

    for col in cols:
        if col not in out.columns:
            out[col] = ""

    out = out[cols].copy()
    out["start_sec"] = pd.to_numeric(out["start_sec"], errors="coerce")
    out["end_sec"] = pd.to_numeric(out["end_sec"], errors="coerce")
    out = out.dropna(subset=["start_sec", "end_sec"])
    out = out[out["end_sec"] > out["start_sec"]].copy()
    out["duration_sec"] = out["end_sec"] - out["start_sec"]
    out = out.sort_values(["start_sec", "end_sec", "speaker"]).reset_index(drop=True)

    out.to_csv(output_path, sep="\t", index=False)


def _attach_segment_types(
    df: pd.DataFrame,
    reference_segments: pd.DataFrame,
) -> pd.DataFrame:
    """Ensure a transcription table has a canonical `type` column."""
    out = df.copy()
    if "type" in out.columns and out["type"].notna().any():
        out["type"] = out["type"].apply(_normalise_turn_type)
        return out

    ref = reference_segments[["speaker", "start_sec", "end_sec", "type"]].copy()
    ref["type"] = ref["type"].apply(_normalise_turn_type)
    out = out.merge(ref, on=["speaker", "start_sec", "end_sec"], how="left")
    out["type"] = out["type"].fillna("turn").apply(_normalise_turn_type)
    return out


def _export_to_elan_format(df: pd.DataFrame, output_path: str) -> None:
    """
    Export DataFrame to ELAN-compatible tab-delimited format.

    ELAN import format: tier \t begin_ms \t end_ms \t annotation
    Tier names combine speaker and type: P1_turn, P1_backchannel,
        P1_overlapped_turn, etc.

    Args:
        df: DataFrame with speaker, start_sec, end_sec, transcription, type columns
        output_path: Path to save the ELAN-compatible file
    """
    output_data = []

    for _, row in df.iterrows():
        speaker = row["speaker"]
        start_ms = int(float(row["start_sec"]) * 1000)
        end_ms = int(float(row["end_sec"]) * 1000)
        transcription = str(row.get("transcription", "")).replace("\t", " ")
        utt_type = row.get("type", "turn")

        # Tier name combines speaker and type
        tier_name = f"{speaker}_{utt_type}"
        output_data.append(
            {
                "tier": tier_name,
                "begin": start_ms,
                "end": end_ms,
                "annotation": transcription,
            }
        )

    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_path, sep="\t", index=False)

    # Report tiers
    tier_names = sorted(output_df["tier"].unique())
    print(f"  Created {len(output_data)} annotations across {len(tier_names)} tiers")
    print(f"  Tiers: {tier_names}")


def process_conversation(
    speakers_audio: Mapping[str, str] | str,
    output_dir: str = "outputs",
    vad_type: str = "rvad",
    rvad_threshold: float = 0.4,
    auth_token: str | None = None,
    vad_min_duration: float = 0.07,
    energy_margin_db: EnergyMargin = 20.0,
    gap_thresh: float = 0.5,
    short_utt_thresh: float = 1.0,
    window_sec: float = 3.0,
    merge_short_after_long: bool = True,
    merge_long_after_short: bool = True,
    long_merge_enabled: bool = True,
    merge_max_dur: float = 60.0,
    bridge_short_opponent: bool = True,
    transcription_model_name: str = "large-v3",
    whisper_device: str = "auto",
    whisper_language: str = "da",
    whisper_model_batch_size: int = 100,
    transcription_padding_sec: float = 0.2,
    entropy_threshold: float = 1.5,
    max_backchannel_dur: float = 1.0,
    max_gap_sec: float = 3.0,
    batch_size: float | None = 30.0,
    interactive_energy_filter: bool = False,
    skip_vad_if_exists: bool = False,
    skip_transcription_if_exists: bool = False,
    persist_transcription_artifacts: bool = False,
    cleanup_speaker_folders: bool = True,
    min_duration_samples: float = 1600,  # float('inf'): skips transcription
    export_elan: bool = True,
    preprocess_audio_enabled: bool = False,
    preprocess_config: Optional[PreprocessConfig] = None,
    preprocess_config_mild: Optional[PreprocessConfig] = None,
    preprocess_config_strong: Optional[PreprocessConfig] = None,
    evaluate_ref_path: str | None = None,
    evaluate_stages: List[str] | None = None,
    evaluate_collar: float = 0.25,
    evaluate_plot: bool = False,
    evaluate_plot_format: str = "pdf",
    evaluate_plot_dpi: int | None = None,
) -> Dict[str, object]:
    """
    Run the complete VAD→transcription→labeling pipeline for a conversation.

    Args:
        speakers_audio: Mapping of speaker names to audio file paths,
            or single path for diarization.
        output_dir: Directory to save output files.
        vad_type: Type of VAD to use ('silero', 'rvad', 'whisper', 'pyannote', 'nemo').
        auth_token: HuggingFace auth token (required for pyannote).
        vad_min_duration: Minimum duration (in seconds) for VAD segments.
        energy_margin_db: Energy margin (in dB) for filtering low-energy segments.
        gap_thresh: Maximum gap (in seconds) to merge segments from same speaker.
        short_utt_thresh: Threshold (in seconds) to classify utterances as short.
        window_sec: Time window (in seconds) to look ahead for merging.
        merge_short_after_long: Whether to merge short utterances after long ones.
        merge_long_after_short: Whether to merge long utterances after short ones.
        long_merge_enabled: Whether to merge two consecutive long utterances.
        merge_max_dur: Maximum duration (in seconds) for merged turns.
        bridge_short_opponent: Whether to bridge over short opponent utterances.
        transcription_model_name: Name of the Transcription model to use.
        whisper_device: Device to run Whisper on ('auto', 'cpu', 'cuda').
        whisper_language: Language code for transcription.
        whisper_model_batch_size: Batch size for Whisper transcription.
        transcription_padding_sec: Extra context (seconds) added on both
            sides of each transcription segment when slicing audio for ASR.
            Does not change stored segment timestamps.
        entropy_threshold: Threshold for classifying backchannels vs turns.
        max_backchannel_dur: Maximum duration for backchannel merging.
        max_gap_sec: Maximum gap for merging with context.
        batch_size: Batch size (in seconds) for processing segments.
        interactive_energy_filter: If True, interactively adjust energy
            threshold.
        skip_vad_if_exists: Whether to skip VAD/diarization if existing
            output files are found.
        skip_transcription_if_exists: If True, skip transcription and
            classification if classified_transcriptions.txt exists.
        persist_transcription_artifacts: If True, keep per-segment WAV and
            per-segment transcript cache files in speaker folders for reuse.
            If False (default), these intermediate files are removed after
            transcription to reduce disk usage.
        cleanup_speaker_folders: If True, remove per-speaker subdirectories
            (segment WAVs/cached transcription files and speaker-level
            intermediates) at the end of the run to save disk space.
        min_duration_samples: Minimum duration (in samples) for segments
            to be transcribed.
        export_elan: If True, export final labels to ELAN-compatible
            tab-delimited format (default: True).
        preprocess_audio_enabled: If True, apply audio preprocessing
            (high-pass filter, noise reduction, loudness normalisation)
            before VAD and transcription.  The preprocessed files are saved
            to ``output_dir/preprocessed/`` and used for all downstream
            stages.  Default: False.
        preprocess_config: Fine-grained control over preprocessing steps.
            When provided, ``preprocess_audio_enabled`` is ignored and the
            ``config.enabled`` flag is used instead.  See
            ``PreprocessConfig`` for available options.
        evaluate_ref_path: If provided, run stage-wise evaluation against
            this ground-truth file after the pipeline finishes.
        evaluate_stages: Which stages to evaluate (default: all).
            Options: ``"vad"``, ``"diarization"``, ``"transcription"``,
            ``"label_type"``.
        evaluate_collar: Collar in seconds for evaluation boundary
            tolerance (default: 0.25).
        evaluate_plot: If True, generate plot artifacts for evaluation metrics.
        evaluate_plot_format: Plot file format ("png", "pdf", "svg").
        evaluate_plot_dpi: Optional DPI used for PNG plot outputs.
            Ignored for vector formats (pdf/svg).
        preprocess_config_mild: When set alongside ``preprocess_config_strong``,
            enables **dual preprocessing mode**.  This config is applied before
            VAD/diarization (gentler settings preserve speaker-separation cues).
        preprocess_config_strong: Applied before ASR transcription in dual
            preprocessing mode (more aggressive enhancement maximises ASR clarity).
            Ignored unless ``preprocess_config_mild`` is also provided.

    Returns:
        Dictionary with paths to output files and processed DataFrames.
    """

    print("Starting conversation processing pipeline...")
    os.makedirs(output_dir, exist_ok=True)

    # ---- Audio preprocessing / speech enhancement ----
    # speakers_audio_asr holds ASR audio paths when dual mode is active; None otherwise.
    speakers_audio_asr: Optional[Union[Dict[str, str], str]] = None
    _strong_path_for_diar: Optional[str] = (
        None  # strong path for single-file diarization
    )

    _dual_mode = (
        preprocess_config_mild is not None or preprocess_config_strong is not None
    )

    if _dual_mode:
        # Dual preprocessing: mild version for VAD, strong version for ASR.
        # Default mild to moderate profile; default strong to noisy profile.
        _orig = speakers_audio
        if preprocess_config_mild is None:
            preprocess_config_mild = PreprocessConfig(
                **{**_PROFILES.get("moderate", {}), "output_suffix": "_mild"}
            )
        elif preprocess_config_mild.output_suffix == "":
            preprocess_config_mild = _dc_replace(
                preprocess_config_mild, output_suffix="_mild"
            )
        if preprocess_config_strong is None:
            preprocess_config_strong = PreprocessConfig(
                **{**_PROFILES.get("noisy", {}), "output_suffix": "_strong"}
            )
        elif preprocess_config_strong.output_suffix == "":
            preprocess_config_strong = _dc_replace(
                preprocess_config_strong, output_suffix="_strong"
            )

        # Apply mild preprocessing (used for VAD/diarization).
        if preprocess_config_mild.enabled:
            if isinstance(_orig, str):
                speakers_audio = preprocess_audio(
                    _orig, output_dir, config=preprocess_config_mild
                )
            else:
                speakers_audio = preprocess_speakers_audio(
                    dict(_orig), output_dir, config=preprocess_config_mild
                )

        # Apply strong preprocessing (will be used for transcription).
        if preprocess_config_strong.enabled:
            if isinstance(_orig, str):
                _strong_path_for_diar = preprocess_audio(
                    _orig, output_dir, config=preprocess_config_strong
                )
            else:
                speakers_audio_asr = preprocess_speakers_audio(
                    dict(_orig), output_dir, config=preprocess_config_strong
                )
    else:
        if preprocess_config is not None:
            pp_config = preprocess_config
        elif preprocess_audio_enabled:
            pp_config = PreprocessConfig(enabled=True)
        else:
            pp_config = PreprocessConfig(enabled=False)

        if pp_config.enabled:
            if isinstance(speakers_audio, str):
                speakers_audio = preprocess_audio(
                    speakers_audio, output_dir, config=pp_config
                )
            else:
                speakers_audio = preprocess_speakers_audio(
                    dict(speakers_audio), output_dir, config=pp_config
                )

    vad_paths: Dict[str, str] = {}
    speaker_dirs: Dict[str, str] = {}

    if isinstance(speakers_audio, str):
        # Single file input - Diarization Mode
        if vad_type not in {"pyannote", "nemo"}:
            raise ValueError(
                "Single file input requires vad_type='pyannote' or "
                "'nemo' for diarization."
            )

        audio_path = speakers_audio
        print(f"Processing single audio file: {audio_path}")
        print(f"Output directory: {output_dir}")

        # Check if diarization already done (check for combined_vad.txt which persists after cleanup)
        combined_vad_path = os.path.join(output_dir, "combined_vad.txt")
        if skip_vad_if_exists and os.path.exists(combined_vad_path):
            print(
                "VAD files already exist (combined_vad.txt found), skipping diarization."
            )
            # Load combined_vad to determine speakers
            df_combined = pd.read_csv(combined_vad_path, sep="\t")
            speakers = sorted(df_combined["speaker"].unique().tolist())
            print(f"  Found speakers: {speakers}")
            vad_paths = {}  # We'll reconstruct paths from combined_vad as needed
            speakers_audio = {speaker: audio_path for speaker in speakers}
        else:
            # Run diarization
            vad = SpeechActivityDetector(vad_type=vad_type, auth_token=auth_token)
            print("\n1. Running Voice Activity Detection (Diarization)...")
            vad_paths = vad.run_diarization(
                audio_path, output_dir, min_duration=vad_min_duration
            )
            speakers_audio = {speaker: audio_path for speaker in vad_paths.keys()}

        speakers = list(speakers_audio.keys())

        # In dual mode, map every speaker to the strong-preprocessed file for ASR.
        if _dual_mode and _strong_path_for_diar:
            speakers_audio_asr = {
                speaker: _strong_path_for_diar for speaker in speakers
            }

        # Create speaker dirs
        for speaker in speakers:
            speaker_dir = os.path.join(output_dir, speaker)
            os.makedirs(speaker_dir, exist_ok=True)
            speaker_dirs[speaker] = speaker_dir

        print(f"✓ Diarization completed. Found speakers: {speakers}")

    else:
        # Multiple files input - VAD Mode
        speakers = list(speakers_audio.keys())
        if not speakers:
            raise ValueError("speakers_audio must contain at least one entry")

        for speaker in speakers:
            speaker_dir = os.path.join(output_dir, speaker)
            os.makedirs(speaker_dir, exist_ok=True)
            speaker_dirs[speaker] = speaker_dir

        # Check if VAD files already exist (check for combined_vad.txt which persists after cleanup)
        combined_vad_path = os.path.join(output_dir, "combined_vad.txt")
        if skip_vad_if_exists and os.path.exists(combined_vad_path):
            print(
                "All VAD files already exist (combined_vad.txt found), skipping VAD step."
            )
            vad_paths = {}  # Empty — we'll load combined_vad.txt directly
        else:
            vad = SpeechActivityDetector(
                vad_type=vad_type, auth_token=auth_token, rvad_threshold=rvad_threshold
            )
            print("\n1. Running Voice Activity Detection...")
            for speaker, path in speakers_audio.items():
                vad_path = os.path.join(
                    speaker_dirs[speaker],
                    f"{os.path.splitext(os.path.basename(path))[0]}_vad.txt",
                )
                vad.run_vad(path, vad_path, min_duration=vad_min_duration)
                vad_paths[speaker] = vad_path
            print("✓ VAD completed")

    # Common pipeline continues...
    energy_margins = _normalise_margins(energy_margin_db, speakers)

    print("\n2. Combining VAD outputs...")
    combined_vad_path = os.path.join(output_dir, "combined_vad.txt")

    # If vad_paths is empty, VAD was skipped; load existing combined_vad.txt instead of combining.
    if vad_paths:
        vad_tables: List[pd.DataFrame] = []
        for speaker in speakers:
            vad_tables.append(_read_vad_segments(vad_paths[speaker], speaker))

        if vad_tables:
            combined_vad_df = pd.concat(vad_tables, ignore_index=True)
        else:
            combined_vad_df = pd.DataFrame(
                columns=[
                    "speaker",
                    "start_sec",
                    "end_sec",
                    "duration_sec",
                    "type",
                    "transcription",
                ]
            )

        _write_standard_segments(combined_vad_df, combined_vad_path)
    else:
        # VAD was skipped; load existing combined_vad.txt
        combined_vad_df = pd.read_csv(combined_vad_path, sep="\t")

    print(
        f"✓ Combined VAD output: {combined_vad_path} ({len(combined_vad_df)} segments)"
    )

    print("\n3. Loading and filtering VAD segments...")
    filtered_segments: List[pd.DataFrame] = []
    for idx, speaker in enumerate(speakers):
        audio_path = speakers_audio[speaker]

        speaker_vad = combined_vad_df[combined_vad_df["speaker"] == speaker].copy()
        df = speaker_vad.rename(
            columns={"start_sec": "start", "end_sec": "end", "duration_sec": "duration"}
        )[["start", "end", "duration"]]
        df["label"] = "T"
        df.reset_index(drop=True, inplace=True)

        margin_db = energy_margins[idx]
        filt_df = filter_low_energy_segments(
            df,
            audio_path,
            energy_margin_db=margin_db,
            interactive_threshold=interactive_energy_filter,
        )
        filt_df["speaker"] = speaker
        filtered_segments.append(filt_df)

    if filtered_segments:
        combined = (
            pd.concat(filtered_segments).sort_values(by="start").reset_index(drop=True)
        )
    else:
        combined = pd.DataFrame(columns=["start", "end", "duration", "speaker"])

    filtered_segments_path = os.path.join(output_dir, "filtered_segments.txt")
    filtered_for_output = combined.rename(
        columns={"start": "start_sec", "end": "end_sec", "duration": "duration_sec"}
    )
    filtered_for_output["type"] = "turn"
    _write_standard_segments(filtered_for_output, filtered_segments_path)

    speaker_counts = {
        speaker: int(count)
        for speaker, count in combined["speaker"].value_counts().items()
    }
    print(f"✓ Filtered segments: {speaker_counts}")
    print(f"✓ Filtered segments output: {filtered_segments_path}")

    print("\n4. Merging turns...")
    merged_turns_path = os.path.join(output_dir, "merged_turns.txt")
    turns_df = create_turns_df_windowed(
        df=combined,
        gap_thresh=gap_thresh,
        short_utt_thresh=short_utt_thresh,
        window_sec=window_sec,
        merge_short_after_long=merge_short_after_long,
        merge_long_after_short=merge_long_after_short,
        long_merge_enabled=long_merge_enabled,
        merge_max_dur=merge_max_dur,
        bridge_short_opponent=bridge_short_opponent,
    )
    turns_for_output = turns_df.copy()
    if "turn_type" in turns_for_output.columns:
        turn_source = turns_for_output["turn_type"]
    else:
        turn_source = pd.Series(
            ["turn"] * len(turns_for_output), index=turns_for_output.index
        )
    turns_for_output["type"] = turn_source.apply(_normalise_turn_type)
    _write_standard_segments(turns_for_output, merged_turns_path)

    # Use standardized merged turns downstream.
    turns_df = pd.read_csv(merged_turns_path, sep="\t")
    print(f"✓ Merged into {len(turns_df)} turns")

    print("\n5. Preparing segments for transcription...")
    segments_by_speaker: Dict[str, pd.DataFrame] = {}
    for speaker in speakers:
        segments_by_speaker[speaker] = turns_df[turns_df["speaker"] == speaker][
            ["start_sec", "end_sec", "duration_sec", "speaker", "type"]
        ]

    raw_transcriptions_path = os.path.join(output_dir, "raw_transcriptions.txt")
    classified_path = os.path.join(output_dir, "classified_transcriptions.txt")

    df_all: pd.DataFrame | None = None
    if skip_transcription_if_exists and os.path.exists(raw_transcriptions_path):
        print("Raw transcriptions already exist, skipping Whisper transcription.")
        df_all = pd.read_csv(raw_transcriptions_path, sep="\t")
        df_all = _attach_segment_types(df_all, turns_df)
        df_all.to_csv(raw_transcriptions_path, sep="\t", index=False)

    else:
        print("\n6. Loading Whisper model and transcribing...")
        model = load_whisper_model(
            transcription_model_name=transcription_model_name,
            device=whisper_device,
            language=whisper_language,
            model_batch_size=whisper_model_batch_size,
        )
        print("✓ Model loaded")

        all_results: List[Dict[str, object]] = []
        # In dual mode, use the strong-preprocessed audio for ASR; otherwise fall back to speakers_audio.
        _asr_audio = (
            speakers_audio_asr if speakers_audio_asr is not None else speakers_audio
        )
        if isinstance(_asr_audio, dict):
            for speaker, audio_path in _asr_audio.items():
                print(f"Transcribing {speaker} segments...")
                speaker_segments = segments_by_speaker[speaker]
                results = transcribe_segments(
                    model=model,
                    segments=speaker_segments.reset_index(drop=True),
                    audio_path=audio_path,
                    output_dir=speaker_dirs[speaker],
                    speaker=speaker,
                    cache=persist_transcription_artifacts,
                    batch_size=batch_size,
                    compress=True,
                    min_duration_samples=int(min_duration_samples),
                    segment_padding_sec=transcription_padding_sec,
                )
                all_results.extend(results)

        print(f"✓ Transcription completed: {len(all_results)} total segments")
        df_all = pd.DataFrame(all_results)
        df_all = _attach_segment_types(df_all, turns_df)
        df_all.to_csv(raw_transcriptions_path, sep="\t", index=False)

    print("\n7. Classifying transcriptions and merging with context...")

    df_class = classify_transcriptions(df_all, threshold=entropy_threshold)
    df_class.to_csv(classified_path, sep="\t", index=False)

    df_merged_context = merge_turns_with_context(
        df_class,
        max_backchannel_dur=max_backchannel_dur,
        max_gap_sec=max_gap_sec,
    )
    final_labels_path = os.path.join(output_dir, "final_labels.txt")
    df_merged_context.to_csv(final_labels_path, sep="\t", index=False)

    print(f"✓ Final processing completed: {len(df_merged_context)} total segments")

    # Export to ELAN format if requested
    elan_export_path = None
    if export_elan:
        print("\n8. Exporting to ELAN format...")
        elan_export_path = os.path.join(output_dir, "final_labels_elan.txt")
        _export_to_elan_format(df_merged_context, elan_export_path)
        print(f"✓ ELAN export: {elan_export_path}")

    print("\n" + "=" * 60)
    print("✅ Pipeline completed successfully!")
    print("=" * 60)
    print(f"Output files saved in: {output_dir}")
    print(f"- VAD results: {list(vad_paths.values())}")
    print(f"- Combined VAD: {combined_vad_path}")
    print(f"- Filtered segments: {filtered_segments_path}")
    print(f"- Merged turns: {merged_turns_path}")
    print(f"- Raw transcriptions: {raw_transcriptions_path}")
    print(f"- Classified transcriptions: {classified_path}")
    print(f"- Final labels: {final_labels_path}")
    if elan_export_path:
        print(f"- ELAN export: {elan_export_path}")

    # Stage-wise evaluation (optional)
    eval_results = None
    eval_plot_paths: Dict[str, str] | None = None
    if evaluate_ref_path:
        eval_step = "9" if export_elan else "8"
        print(f"\n{eval_step}. Running stage-wise evaluation...")
        eval_output = os.path.join(output_dir, "evaluation_metrics.json")
        try:
            eval_results = evaluate_pipeline(
                ref_path=evaluate_ref_path,
                hyp_path=final_labels_path,
                stages=evaluate_stages,
                collar=evaluate_collar,
                output_path=eval_output,
            )
            print_evaluation_summary(eval_results)
            print(f"✓ Evaluation metrics: {eval_output}")

            if evaluate_plot:
                eval_plot_paths = plot_evaluation_results(
                    eval_results,
                    output_dir=output_dir,
                    file_prefix="evaluation",
                    image_format=evaluate_plot_format,
                    dpi=evaluate_plot_dpi,
                )
                if eval_plot_paths:
                    print("✓ Evaluation plots:")
                    for _, plot_path in eval_plot_paths.items():
                        print(f"  - {plot_path}")
                else:
                    print("ℹ No plottable metrics found for evaluation plots.")
        except Exception as exc:
            print(f"⚠ Evaluation failed: {exc}")
            eval_results = {"error": str(exc)}

    cleaned_speaker_dirs: List[str] = []
    if cleanup_speaker_folders:
        print("\nCleaning speaker folders to save disk space...")
        for speaker_dir in speaker_dirs.values():
            if os.path.isdir(speaker_dir):
                try:
                    shutil.rmtree(speaker_dir)
                    cleaned_speaker_dirs.append(speaker_dir)
                except Exception as exc:
                    print(f"⚠ Failed to remove {speaker_dir}: {exc}")
        if cleaned_speaker_dirs:
            print(f"✓ Removed {len(cleaned_speaker_dirs)} speaker folder(s)")
        else:
            print("ℹ No speaker folders found to remove")

    return {
        "output_dir": output_dir,
        "vad_paths": vad_paths,
        "combined_vad": combined_vad_path,
        "filtered_segments": filtered_segments_path,
        "merged_turns": merged_turns_path,
        "raw_transcriptions": raw_transcriptions_path,
        "classified": classified_path,
        "final_labels": final_labels_path,
        "elan_export": elan_export_path,
        "turns_df": turns_df,
        "classified_df": df_class,
        "final_df": df_merged_context,
        "evaluation": eval_results,
        "evaluation_plots": eval_plot_paths,
        "cleaned_speaker_dirs": cleaned_speaker_dirs,
    }


# ============================================================================
# "Continue" pipeline – add missing steps to an existing labels file
# ============================================================================


def _detect_needed_steps(df: pd.DataFrame) -> Tuple[bool, bool]:
    """Return ``(needs_transcription, needs_labels)``.

    ``needs_transcription`` is True when most rows have empty transcription.
    ``needs_labels`` is True when the file has no backchannel/overlapped_turn
    labels (all segments are "turn"), meaning entropy-based classification
    has not been run yet.
    """
    transcription_filled = (df["transcription"].str.strip().str.len() > 0).mean() > 0.5
    needs_transcription = not transcription_filled

    non_turn_types = set(df["type"].str.lower().unique()) - {"turn", "t", ""}
    needs_labels = len(non_turn_types) <= 0

    return needs_transcription, needs_labels


def continue_conversation(
    existing_path: str,
    speakers_audio: Mapping[str, str],
    output_dir: str = "outputs",
    transcription_model_name: str = "large-v3",
    whisper_device: str = "auto",
    whisper_language: str = "da",
    whisper_model_batch_size: int = 100,
    transcription_padding_sec: float = 0.2,
    entropy_threshold: float = 1.5,
    max_backchannel_dur: float = 1.0,
    max_gap_sec: float = 3.0,
    batch_size: Optional[float] = 30.0,
    persist_transcription_artifacts: bool = False,
    cleanup_speaker_folders: bool = True,
    min_duration_samples: float = 1600,
    export_elan: bool = True,
    evaluate_ref_path: Optional[str] = None,
    evaluate_stages: Optional[List[str]] = None,
    evaluate_collar: float = 0.25,
    evaluate_plot: bool = False,
    evaluate_plot_format: str = "pdf",
    evaluate_plot_dpi: Optional[int] = None,
) -> Dict[str, object]:
    """Add missing pipeline steps to an existing labels file.

    Reads an existing annotation file (tab-separated ``.txt`` or Aegisub
    ``.ass``), detects which steps are still needed, runs only those, and
    writes the updated output while **preserving** already-present labels.

    Detection logic:

    - **needs_transcription**: more than half of segments have an empty
      ``transcription`` column.
    - **needs_labels**: all ``type`` values are ``"turn"`` (no backchannel
      or overlapped_turn has been assigned), meaning entropy-based
      classification has not been applied yet.

    If existing turn/backchannel labels are present they are **preserved**:
    only the ``transcription`` column is filled in; ``classify_transcriptions``
    is *not* called.

    Args:
        existing_path: Path to the existing labels file (.txt or .ass).
        speakers_audio: Mapping of speaker name → audio file path.
            For diarized recordings where both speakers are in one file,
            map every speaker to the same path, e.g.
            ``{"A": "conv.wav", "B": "conv.wav"}``.
        output_dir: Directory where updated output files are written.
        transcription_model_name: Whisper model variant.
        whisper_device: ``"auto"``, ``"cpu"``, or ``"cuda"``.
        whisper_language: Language code for Whisper.
        whisper_model_batch_size: Batch size passed to Whisper.
        transcription_padding_sec: Extra context added on both sides of
            each segment when slicing audio for ASR.
        entropy_threshold: Entropy cut-off for backchannel classification
            (only used when ``needs_labels`` is True).
        max_backchannel_dur: Maximum duration for backchannel merging.
        max_gap_sec: Maximum gap for context merging.
        batch_size: Audio batch size in seconds (None to disable batching).
        persist_transcription_artifacts: Keep per-segment WAV files.
        cleanup_speaker_folders: Remove per-speaker working directories
            after the run.
        min_duration_samples: Minimum segment length (samples) for
            transcription.
        export_elan: Write an ELAN-compatible tab-delimited export.
        evaluate_ref_path: Optional ground-truth file for evaluation.
        evaluate_stages: Stages to evaluate (default: all).
        evaluate_collar: Collar in seconds for evaluation tolerance.
        evaluate_plot: Generate metric plots.
        evaluate_plot_format: Plot file format (``"pdf"``, ``"png"``,
            ``"svg"``).
        evaluate_plot_dpi: DPI for raster plot outputs.

    Returns:
        Dictionary with paths to output files and processed DataFrames,
        matching the structure returned by :func:`process_conversation`.
    """
    print(f"Continuing pipeline from: {existing_path}")
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Parse existing file ──────────────────────────────────────────────
    df_existing = load_reference(existing_path)
    print(
        f"  Loaded {len(df_existing)} segments from {os.path.basename(existing_path)}"
    )

    # ── 2. Detect what is needed ────────────────────────────────────────────
    needs_transcription, needs_labels = _detect_needed_steps(df_existing)
    print(
        f"  Needs transcription    : {'yes' if needs_transcription else 'no (already present)'}"
    )
    print(
        f"  Needs label assignment : {'yes' if needs_labels else 'no (turn/backchannel found)'}"
    )

    if not needs_transcription and not needs_labels:
        print("  Nothing to add — file appears complete.")
        final_labels_path = os.path.join(output_dir, "final_labels.txt")
        _write_standard_segments(df_existing, final_labels_path)
        return {
            "output_dir": output_dir,
            "final_labels": final_labels_path,
            "final_df": df_existing,
            "evaluation": None,
            "evaluation_plots": None,
            "cleaned_speaker_dirs": [],
        }

    speakers = list(speakers_audio.keys())

    # ── 3. Create per-speaker working directories ───────────────────────────
    speaker_dirs: Dict[str, str] = {}
    for speaker in speakers:
        sd = os.path.join(output_dir, speaker)
        os.makedirs(sd, exist_ok=True)
        speaker_dirs[speaker] = sd

    # ── 4. Write existing data as merged_turns baseline ────────────────────
    merged_turns_path = os.path.join(output_dir, "merged_turns.txt")
    _write_standard_segments(df_existing, merged_turns_path)

    df_work = df_existing.copy()
    raw_transcriptions_path = os.path.join(output_dir, "raw_transcriptions.txt")

    # ── 5. Transcription (if needed) ────────────────────────────────────────
    if needs_transcription:
        print("\nRunning transcription on existing segments...")
        model = load_whisper_model(
            transcription_model_name=transcription_model_name,
            device=whisper_device,
            language=whisper_language,
            model_batch_size=whisper_model_batch_size,
        )
        print("✓ Model loaded")

        all_results: List[Dict[str, object]] = []
        for speaker in speakers:
            if speaker not in speakers_audio:
                print(f"⚠ No audio path for speaker {speaker!r}, skipping")
                continue
            audio_path = speakers_audio[speaker]
            speaker_segs = df_existing[df_existing["speaker"] == speaker][
                ["start_sec", "end_sec", "duration_sec", "speaker", "type"]
            ]
            if speaker_segs.empty:
                print(f"  No segments found for speaker {speaker!r}")
                continue
            print(f"  Transcribing {speaker} ({len(speaker_segs)} segments)…")
            results = transcribe_segments(
                model=model,
                segments=speaker_segs.reset_index(drop=True),
                audio_path=audio_path,
                output_dir=speaker_dirs[speaker],
                speaker=speaker,
                cache=persist_transcription_artifacts,
                batch_size=batch_size,
                compress=True,
                min_duration_samples=int(min_duration_samples),
                segment_padding_sec=transcription_padding_sec,
            )
            all_results.extend(results)

        print(f"✓ Transcription completed: {len(all_results)} segments")
        df_transcribed = pd.DataFrame(all_results)

        # Build lookup (speaker, start_sec, end_sec) → transcription text
        key_to_text: Dict[tuple, str] = {}
        if "transcription" in df_transcribed.columns:
            for _, row in df_transcribed.iterrows():
                key = (
                    str(row.get("speaker", "")),
                    round(float(row.get("start_sec", 0)), 3),
                    round(float(row.get("end_sec", 0)), 3),
                )
                key_to_text[key] = str(row.get("transcription", ""))

        def _fill_transcription(r: "pd.Series") -> str:
            key = (
                str(r["speaker"]),
                round(float(r["start_sec"]), 3),
                round(float(r["end_sec"]), 3),
            )
            return key_to_text.get(key, "")

        df_work = df_existing.copy()
        df_work["transcription"] = df_work.apply(_fill_transcription, axis=1)
        df_work.to_csv(raw_transcriptions_path, sep="\t", index=False)

    # ── 6. Label classification (if needed) ─────────────────────────────────
    classified_path = os.path.join(output_dir, "classified_transcriptions.txt")
    if needs_labels:
        print("\nRunning label classification…")
        df_class = classify_transcriptions(df_work, threshold=entropy_threshold)
        df_class.to_csv(classified_path, sep="\t", index=False)
        df_merged = merge_turns_with_context(
            df_class, max_backchannel_dur=max_backchannel_dur, max_gap_sec=max_gap_sec
        )
    else:
        # Preserve existing labels – write as-is
        print("Preserving existing turn/backchannel labels.")
        df_work.to_csv(classified_path, sep="\t", index=False)
        df_merged = df_work.copy()

    # ── 7. Write final output ───────────────────────────────────────────────
    final_labels_path = os.path.join(output_dir, "final_labels.txt")
    _write_standard_segments(df_merged, final_labels_path)
    print(f"\n✅ Continue pipeline completed! ({len(df_merged)} segments)")
    print(f"   Output: {final_labels_path}")

    # ── 8. ELAN export ───────────────────────────────────────────────────────
    elan_export_path = None
    if export_elan:
        elan_export_path = os.path.join(output_dir, "final_labels_elan.txt")
        _export_to_elan_format(df_merged, elan_export_path)
        print(f"   ELAN   : {elan_export_path}")

    # ── 9. Cleanup ───────────────────────────────────────────────────────────
    cleaned_speaker_dirs: List[str] = []
    if cleanup_speaker_folders:
        for sd in speaker_dirs.values():
            if os.path.isdir(sd):
                try:
                    shutil.rmtree(sd)
                    cleaned_speaker_dirs.append(sd)
                except Exception as exc:
                    print(f"⚠ Failed to remove {sd}: {exc}")

    # ── 10. Evaluation ───────────────────────────────────────────────────────
    eval_results = None
    eval_plot_paths: Optional[Dict[str, str]] = None
    if evaluate_ref_path:
        print("\nRunning stage-wise evaluation…")
        eval_output = os.path.join(output_dir, "evaluation_metrics.json")
        try:
            eval_results = evaluate_pipeline(
                ref_path=evaluate_ref_path,
                hyp_path=final_labels_path,
                stages=evaluate_stages,
                collar=evaluate_collar,
                output_path=eval_output,
            )
            print_evaluation_summary(eval_results)
            if evaluate_plot:
                eval_plot_paths = plot_evaluation_results(
                    eval_results,
                    output_dir=output_dir,
                    file_prefix="evaluation",
                    image_format=evaluate_plot_format,
                    dpi=evaluate_plot_dpi,
                )
        except Exception as exc:
            print(f"⚠ Evaluation failed: {exc}")
            eval_results = {"error": str(exc)}

    return {
        "output_dir": output_dir,
        "merged_turns": merged_turns_path,
        "raw_transcriptions": raw_transcriptions_path if needs_transcription else None,
        "classified": classified_path,
        "final_labels": final_labels_path,
        "elan_export": elan_export_path,
        "final_df": df_merged,
        "evaluation": eval_results,
        "evaluation_plots": eval_plot_paths,
        "cleaned_speaker_dirs": cleaned_speaker_dirs,
    }
