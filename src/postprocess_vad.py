"""
Based on Conversational_speech_labeling_pipeline by Hanlu He.

https://github.com/hanlululu/Conversational_speech_labeling_pipeline
"""

import os
import shutil

import numpy as np
import pandas as pd
import soundfile as sf


def remove_short_segments(df: pd.DataFrame, min_duration: float = 0.07) -> pd.DataFrame:
    """
    Remove segments that are shorter than the minimum duration.

    Args:
        df: DataFrame with 'start' and 'end' columns.
        min_duration: Minimum duration in seconds for a segment to be kept.

    Returns:
        Filtered DataFrame with short segments removed.
    """
    if df.empty:
        return df
    return df[(df["end"] - df["start"]) >= min_duration].reset_index(drop=True)


def compute_rms(audio: np.ndarray, sr: int, start_sec: float, end_sec: float) -> float:
    """
    Compute the Root Mean Square (RMS) energy of an audio segment.

    Args:
        audio: Audio data as numpy array.
        sr: Sample rate of the audio.
        start_sec: Start time in seconds.
        end_sec: End time in seconds.

    Returns:
        RMS value of the segment, or 0.0 if segment is empty.
    """
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    seg = audio[start_sample:end_sample]
    return 0.0 if len(seg) == 0 else (seg.astype(float) ** 2).mean() ** 0.5


def filter_low_energy_segments(
    df: pd.DataFrame,
    audio_path: str,
    energy_margin_db: float = 10.0,
    interactive_threshold: bool = False,
) -> pd.DataFrame:
    """
    Filter out low-energy segments based on RMS energy relative to the loudest segment.

    Segments with energy below (max_energy_db - energy_margin_db) are removed.

    Args:
        df: DataFrame with 'start' and 'end' columns.
        audio_path: Path to the audio file.
        energy_margin_db: Margin in dB below the max energy to filter.
        interactive_threshold: If True, interactively adjust threshold with examples.

    Returns:
        Filtered DataFrame with added 'energy' and 'distance_to_threshold' columns.
    """
    if df.empty:
        return df.copy()

    # Load audio
    audio, sr = sf.read(audio_path)
    segment_stats = []

    # Compute energy for each segment
    for _, row in df.iterrows():
        rms = compute_rms(audio, sr, row["start"], row["end"])
        rms_db = 20 * np.log10(rms + 1e-8)
        segment_stats.append({"row": row, "energy": rms, "energy_db": rms_db})

    # Calculate and display filtering stats
    if segment_stats:
        reference_db = max(stat["energy_db"] for stat in segment_stats)
        threshold_db = reference_db - energy_margin_db
        kept_count = sum(
            1 for stat in segment_stats if stat["energy_db"] >= threshold_db
        )
        cut_count = len(segment_stats) - kept_count
        percentage_cut = (cut_count / len(segment_stats)) * 100
        print(
            f"Energy filtering: Max {reference_db:.1f} dB, \
                threshold {threshold_db:.1f} dB (margin {energy_margin_db:.1f} dB)"
        )
        print(
            f"Segments: {len(segment_stats)} total, {kept_count} kept, \
                {cut_count} cut ({percentage_cut:.1f}%)"
        )

    # Interactive threshold adjustment
    if interactive_threshold:
        return _interactive_energy_filtering(df, audio, sr, segment_stats, audio_path)

    # Non-interactive filtering
    return _apply_energy_filtering(segment_stats, energy_margin_db)


def _apply_energy_filtering(
    segment_stats: list, energy_margin_db: float
) -> pd.DataFrame:
    """Apply energy filtering with given margin."""
    # Determine threshold based on max energy
    reference_db = max(stat["energy_db"] for stat in segment_stats)
    threshold_db = reference_db - energy_margin_db

    # Keep segments above threshold
    kept_rows = []
    for stat in segment_stats:
        if stat["energy_db"] >= threshold_db:
            row_copy = stat["row"].copy()
            row_copy["energy"] = stat["energy"]
            row_copy["distance_to_threshold"] = stat["energy_db"] - threshold_db
            kept_rows.append(row_copy)

    return pd.DataFrame(kept_rows).reset_index(drop=True)


def _interactive_energy_filtering(
    df: pd.DataFrame, audio: np.ndarray, sr: int, segment_stats: list, audio_path: str
) -> pd.DataFrame:
    """Interactive energy filtering with audio examples."""
    # Create interim folder
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    interim_dir = os.path.join("interim", f"energy_filtering_{base_name}")
    os.makedirs(interim_dir, exist_ok=True)

    # Calculate max energy
    max_energy_db = max(stat["energy_db"] for stat in segment_stats)
    current_threshold = max_energy_db - 10.0  # Start with default 10dB margin

    while True:
        # Calculate margin for current threshold
        current_margin = max_energy_db - current_threshold

        # Count segments
        kept_count = sum(
            1 for stat in segment_stats if stat["energy_db"] >= current_threshold
        )
        cut_count = len(segment_stats) - kept_count

        # Print current status
        print(f"\nMax energy: {max_energy_db:.1f} dB")
        print(
            f"Current threshold: {current_threshold:.1f} dB \
                (margin: {current_margin:.1f} dB)"
        )
        print(f"Segments kept: {kept_count}, cut: {cut_count}")

        # Find examples for boundary segments
        kept_stats = [
            stat for stat in segment_stats if stat["energy_db"] >= current_threshold
        ]
        cut_stats = [
            stat for stat in segment_stats if stat["energy_db"] < current_threshold
        ]

        if kept_stats and cut_stats:
            # Find loudest cut and quietest kept
            loudest_cut = max(cut_stats, key=lambda x: x["energy_db"])
            quietest_kept = min(kept_stats, key=lambda x: x["energy_db"])

            # Save example clips
            loudest_cut_path = os.path.join(interim_dir, "loudest_cut.wav")
            quietest_kept_path = os.path.join(interim_dir, "quietest_kept.wav")

            # Extract and save clips
            start_sample = int(loudest_cut["row"]["start"] * sr)
            end_sample = int(loudest_cut["row"]["end"] * sr)
            sf.write(loudest_cut_path, audio[start_sample:end_sample], sr)

            start_sample = int(quietest_kept["row"]["start"] * sr)
            end_sample = int(quietest_kept["row"]["end"] * sr)
            sf.write(quietest_kept_path, audio[start_sample:end_sample], sr)

            print(f"\nExample clips saved in: {interim_dir}")
            print(
                f"- Loudest cut segment: {loudest_cut_path} \
                    ({loudest_cut['energy_db']:.1f} dB)"
            )
            print(
                f"- Quietest kept segment: {quietest_kept_path} \
                    ({quietest_kept['energy_db']:.1f} dB)"
            )
            print(
                f"Energy difference: \
                    {quietest_kept['energy_db'] - loudest_cut['energy_db']:.1f} dB"
            )

        # Ask user for new threshold
        while True:
            response = input(
                "\nEnter new enrgy margin in dB (or 'k' to keep current): "
            ).strip()
            if response.lower() == "k":
                print(f"Keeping threshold of {current_threshold:.1f} dB")
                break
            try:
                new_threshold = max_energy_db - float(response)
                current_threshold = new_threshold
                break
            except ValueError:
                print("Please enter a valid number or 'k' to keep")

        if response.lower() == "k":
            break

    # Clean up example clips after accepting threshold
    if os.path.exists(interim_dir):
        print("Cleaning up example clips...")
        shutil.rmtree(interim_dir)

    # Apply final filtering
    return _apply_energy_filtering(segment_stats, current_margin)
