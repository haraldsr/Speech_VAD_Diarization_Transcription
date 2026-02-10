"""
High-level conversation processing pipeline utilities.

Based on Conversational_speech_labeling_pipeline by Hanlu He.

https://github.com/hanlululu/Conversational_speech_labeling_pipeline

"""

from __future__ import annotations

import os
from typing import Dict, List, Mapping, Tuple, Union

import pandas as pd

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


def _export_to_elan_format(df: pd.DataFrame, output_path: str) -> None:
    """
    Export DataFrame to ELAN-compatible tab-delimited format.

    ELAN import format: tier \t begin_ms \t end_ms \t annotation
    Tier names combine speaker and type: P1_turn, P1_backchannel, etc.

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
    auth_token: str | None = None,
    vad_min_duration: float = 0.07,
    energy_margin_db: EnergyMargin = 10.0,
    gap_thresh: float = 0.5,
    short_utt_thresh: float = 1.0,
    window_sec: float = 3.0,
    merge_short_after_long: bool = True,
    merge_long_after_short: bool = True,
    long_merge_enabled: bool = True,
    merge_max_dur: float = 60.0,
    bridge_short_opponent: bool = True,
    transcription_model_name: str = "openai/whisper-large-v3",
    whisper_device: str = "auto",
    whisper_language: str = "da",
    whisper_transformers_batch_size: int = 100,
    entropy_threshold: float = 1.5,
    max_backchannel_dur: float = 1.0,
    max_gap_sec: float = 3.0,
    batch_size: float | None = 30.0,
    interactive_energy_filter: bool = False,
    skip_vad_if_exists: bool = True,
    skip_transcription_if_exists: bool = True,
    min_duration_samples: float = 1600, # float('inf'): skips transcription, default: 1600
    export_elan: bool = True,
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
        keep_bridged_segments: If True, preserve segments bridged over as separate
            entries. If False, only keep the merged turns.
        transcription_model_name: Name of the Transcription model to use.
        whisper_device: Device to run Whisper on ('auto', 'cpu', 'cuda').
        whisper_language: Language code for transcription.
        whisper_transformers_batch_size: Batch size for Whisper transcription.
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
        min_duration_samples: Minimum duration (in seconds) for segments to be transcribed.
        export_elan: If True, export final labels to ELAN-compatible
            tab-delimited format (default: True).

    Returns:
        Dictionary with paths to output files and processed DataFrames.
    """

    print("Starting conversation processing pipeline...")
    os.makedirs(output_dir, exist_ok=True)

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

        # Check if diarization already done
        if skip_vad_if_exists and os.path.exists(output_dir):
            import re

            speaker_dirs_found = [
                d
                for d in os.listdir(output_dir)
                if os.path.isdir(os.path.join(output_dir, d))
                and re.match(r"^(SPEAKER_\d+|P\d+)$", d)
            ]
            vad_paths = {}
            speakers = []
            for speaker_dir in speaker_dirs_found:
                audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
                vad_file = os.path.join(
                    output_dir,
                    speaker_dir,
                    f"{audio_basename}_{speaker_dir}_vad.txt",
                )
                if os.path.exists(vad_file):
                    speakers.append(speaker_dir)
                    vad_paths[speaker_dir] = vad_file
            if speakers:
                print("VAD files already exist, skipping diarization.")
                # Construct speakers_audio
                speakers_audio = {speaker: audio_path for speaker in speakers}
            else:
                # Run diarization
                vad = SpeechActivityDetector(vad_type=vad_type, auth_token=auth_token)
                print("\n1. Running Voice Activity Detection (Diarization)...")
                vad_paths = vad.run_diarization(
                    audio_path, output_dir, min_duration=vad_min_duration
                )
                speakers_audio = {speaker: audio_path for speaker in vad_paths.keys()}
        else:
            vad = SpeechActivityDetector(vad_type=vad_type, auth_token=auth_token)
            print("\n1. Running Voice Activity Detection (Diarization)...")
            vad_paths = vad.run_diarization(
                audio_path, output_dir, min_duration=vad_min_duration
            )
            speakers_audio = {speaker: audio_path for speaker in vad_paths.keys()}

        speakers = list(speakers_audio.keys())

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

        # Compute expected vad_paths
        expected_vad_paths = {}
        for speaker, path in speakers_audio.items():
            vad_path = os.path.join(
                speaker_dirs[speaker],
                f"{os.path.splitext(os.path.basename(path))[0]}_vad.txt",
            )
            expected_vad_paths[speaker] = vad_path

        # Check if all exist
        all_exist = all(os.path.exists(p) for p in expected_vad_paths.values())
        if all_exist and skip_vad_if_exists:
            print("All VAD files already exist, skipping VAD step.")
            vad_paths = expected_vad_paths
        else:
            vad = SpeechActivityDetector(vad_type=vad_type, auth_token=auth_token)
            print("\n1. Running Voice Activity Detection...")
            for speaker, path in speakers_audio.items():
                vad_path = expected_vad_paths[speaker]
                vad.run_vad(path, vad_path, min_duration=vad_min_duration)
                vad_paths[speaker] = vad_path
            print("✓ VAD completed")

    # Common pipeline continues...
    energy_margins = _normalise_margins(energy_margin_db, speakers)
    # exit(0)
    print("\n2. Loading and filtering VAD segments...")
    filtered_segments: List[pd.DataFrame] = []
    for idx, speaker in enumerate(speakers):
        audio_path = speakers_audio[speaker]
        vad_path = vad_paths[speaker]

        df = pd.read_csv(
            vad_path, sep="\t", skiprows=1, names=["start", "end", "label"]
        )
        df = df[df["label"] == "T"].copy()
        df["duration"] = df["end"] - df["start"]
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

    combined = (
        pd.concat(filtered_segments).sort_values(by="start").reset_index(drop=True)
    )
    speaker_counts = {
        speaker: int(count)
        for speaker, count in combined["speaker"].value_counts().items()
    }
    print(f"✓ Filtered segments: {speaker_counts}")

    print("\n3. Merging turns...")
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
    turns_df.to_csv(merged_turns_path, sep="\t", index=False)
    print(f"✓ Merged into {len(turns_df)} turns")

    print("\n4. Preparing segments for transcription...")
    segments_by_speaker: Dict[str, pd.DataFrame] = {}
    for speaker in speakers:
        segments_by_speaker[speaker] = turns_df[turns_df["speaker"] == speaker][
            ["start_sec", "end_sec", "duration_sec", "speaker", "turn_type"]
        ]

    raw_transcriptions_path = os.path.join(output_dir, "raw_transcriptions.txt")
    classified_path = os.path.join(output_dir, "classified_transcriptions.txt")

    df_all: pd.DataFrame | None = None
    if skip_transcription_if_exists and os.path.exists(raw_transcriptions_path):
        print("Raw transcriptions already exist, skipping Whisper transcription.")
        df_all = pd.read_csv(raw_transcriptions_path, sep="\t")
        
    else:
        print("\n5. Loading Whisper model and transcribing...")
        model = load_whisper_model(
            transcription_model_name=transcription_model_name,
            device=whisper_device,
            language=whisper_language,
            transformers_batch_size=whisper_transformers_batch_size,
        )
        print("✓ Model loaded")

        all_results: List[Dict[str, object]] = []
        for speaker, audio_path in speakers_audio.items():
            print(f"Transcribing {speaker} segments...")
            speaker_segments = segments_by_speaker[speaker]
            results = transcribe_segments(
                model=model,
                segments=speaker_segments.reset_index(drop=True),
                audio_path=audio_path,
                output_dir=speaker_dirs[speaker],
                speaker=speaker,
                cache=True,
                batch_size=batch_size,
                compress=True,
                min_duration_samples=min_duration_samples
            )
            all_results.extend(results)

        print(f"✓ Transcription completed: {len(all_results)} total segments")
        df_all = pd.DataFrame(all_results)
        df_all.to_csv(raw_transcriptions_path, sep="\t", index=False)

    print("\n6. Classifying transcriptions and merging with context...")

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
        print("\n7. Exporting to ELAN format...")
        elan_export_path = os.path.join(output_dir, "final_labels_elan.txt")
        _export_to_elan_format(df_merged_context, elan_export_path)
        print(f"✓ ELAN export: {elan_export_path}")

    print("\n" + "=" * 60)
    print("✅ Pipeline completed successfully!")
    print("=" * 60)
    print(f"Output files saved in: {output_dir}")
    print(f"- VAD results: {list(vad_paths.values())}")
    print(f"- Merged turns: {merged_turns_path}")
    print(f"- Raw transcriptions: {raw_transcriptions_path}")
    print(f"- Classified transcriptions: {classified_path}")
    print(f"- Final labels: {final_labels_path}")
    if elan_export_path:
        print(f"- ELAN export: {elan_export_path}")

    return {
        "output_dir": output_dir,
        "vad_paths": vad_paths,
        "merged_turns": merged_turns_path,
        "raw_transcriptions": raw_transcriptions_path,
        "classified": classified_path,
        "final_labels": final_labels_path,
        "elan_export": elan_export_path,
        "turns_df": turns_df,
        "classified_df": df_class,
        "final_df": df_merged_context,
    }
