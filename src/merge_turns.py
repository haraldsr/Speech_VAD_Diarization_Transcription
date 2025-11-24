"""
Based on Conversational_speech_labeling_pipeline by Hanlu He.

https://github.com/hanlululu/Conversational_speech_labeling_pipeline
"""

import numpy as np
import pandas as pd


def _extend_segment(segment: pd.Series, new_end: float) -> None:
    """
    Update a segment to end at new_end and refresh its duration.

    Args:
        segment: The segment series to update (modified in-place).
        new_end: The new end time for the segment.
    """
    segment["end"] = max(segment["end"], new_end)
    segment["duration"] = segment["end"] - segment["start"]


def create_turns_df_windowed(
    df: pd.DataFrame,
    gap_thresh: float = 0.2,
    short_utt_thresh: float = 0.7,
    window_sec: float = 2.0,
    merge_short_after_long: bool = True,
    merge_long_after_short: bool = True,
    long_merge_enabled: bool = True,
    # long_merge_min_dur: float | None = None,
    merge_max_dur: float | None = None,
    bridge_short_opponent: bool = True,
) -> pd.DataFrame:
    """
    Merge VAD segments into speaker turns by scanning forward within a time window.

    Args:
        df: DataFrame with columns 'speaker', 'start', 'end', 'duration'.
        gap_thresh: Maximum gap (in seconds) to merge segments from same speaker.
        short_utt_thresh: Threshold (in seconds) to classify utterances as short.
        window_sec: Time window (in seconds) to look ahead for merging.
        merge_short_after_long: Whether to merge short utterances after long ones.
        merge_long_after_short: Whether to merge long utterances after short ones.
        long_merge_enabled: Whether to merge two consecutive long utterances.
        merge_max_dur: Maximum duration (in seconds) for merged turns.
            If None, no limit is applied.
        bridge_short_opponent: Whether to bridge over short opponent utterances.

    Returns:
        DataFrame with merged turns and optionally preserved bridged segments,
        columns: Speaker, Start_Sec, End_Sec, Duration_Sec, Turn_Type.
    """
    if df.empty:
        return pd.DataFrame(
            columns=["Speaker", "Start_Sec", "End_Sec", "Duration_Sec", "Turn_Type"]
        )

    # Sort segments by start time and ensure duration column exists
    segments = df.sort_values("start").reset_index(drop=True).copy()

    if "duration" not in segments.columns:
        segments["duration"] = segments["end"] - segments["start"]

    # Set default for merge_max_dur if not provided
    if merge_max_dur is None:
        merge_max_dur = np.inf

    turns: list[dict[str, float | str]] = []
    n = len(segments)
    i = 0

    # Track which segments were consumed during merging
    consumed_indices = set()

    while i < n:
        if i in consumed_indices:
            i += 1
            continue

        current = segments.iloc[i].copy()
        speaker = current["speaker"]
        j = i + 1
        merged_indices = [i]  # Track which segments went into this turn

        while True:
            # Find segments within the time window
            window_mask = (segments.index >= j) & (
                segments["start"] <= current["end"] + window_sec
            )
            window = segments.loc[window_mask]
            if window.empty:
                break

            candidate = window.iloc[0]
            candidate_idx = candidate.name
            gap_from_current = max(0.0, candidate["start"] - current["end"])

            if candidate["speaker"] == speaker:
                # Merge if gap is within threshold
                if gap_from_current <= gap_thresh:
                    _extend_segment(current, candidate["end"])
                    merged_indices.append(candidate_idx)
                    j = candidate_idx + 1
                    continue

                # Merge short utterance after long one
                if (
                    merge_short_after_long
                    and current["duration"] >= short_utt_thresh
                    and candidate["duration"] < short_utt_thresh
                    and current["duration"] + candidate["duration"] >= gap_from_current
                    and current["duration"] + candidate["duration"] < merge_max_dur
                ):
                    _extend_segment(current, candidate["end"])
                    merged_indices.append(candidate_idx)
                    j = candidate_idx + 1
                    continue

                # Merge long utterance after short one
                if (
                    merge_long_after_short
                    and current["duration"] < short_utt_thresh
                    and candidate["duration"] >= short_utt_thresh
                    and current["duration"] + candidate["duration"] >= gap_from_current
                    and current["duration"] + candidate["duration"] < merge_max_dur
                ):
                    _extend_segment(current, candidate["end"])
                    merged_indices.append(candidate_idx)
                    j = candidate_idx + 1
                    continue

                # Merge two long utterances
                if (
                    long_merge_enabled
                    and current["duration"] >= short_utt_thresh
                    and candidate["duration"] >= short_utt_thresh
                    and current["duration"] + candidate["duration"] >= gap_from_current
                    and current["duration"] + candidate["duration"] < merge_max_dur
                ):
                    _extend_segment(current, candidate["end"])
                    merged_indices.append(candidate_idx)
                    j = candidate_idx + 1
                    continue
                break

            # Bridge short opponent utterances if enabled
            if not bridge_short_opponent:
                break

            # Look for same speaker segments within short utterance threshold
            sub_window_mask = (segments.index >= j) & (
                segments["start"] <= current["end"] + short_utt_thresh
            )
            sub_window = segments.loc[sub_window_mask]

            if sub_window.empty:
                break

            same_speaker_rows = sub_window[sub_window["speaker"] == speaker]
            if same_speaker_rows.empty:
                break

            target = same_speaker_rows.iloc[-1]

            # Collect all opponent segments that will be bridged
            bridged_opponent_indices = []
            for idx in range(j, target.name + 1):
                if idx in segments.index:
                    seg = segments.loc[idx]
                    if seg["speaker"] != speaker:
                        bridged_opponent_indices.append(idx)
                    else:
                        merged_indices.append(idx)

            _extend_segment(current, target["end"])
            j = target.name + 1

            # Add bridged opponent segments as separate entries if requested
            # for bridged_idx in bridged_opponent_indices:
            #     bridged_seg = segments.loc[bridged_idx]
            #     turns.append(
            #         {
            #             "Speaker": bridged_seg["speaker"],
            #             "Start_Sec": bridged_seg["start"],
            #             "End_Sec": bridged_seg["end"],
            #             "Duration_Sec": bridged_seg["duration"],
            #             "Turn_Type": "BC",  # Mark as bridged/backchannel
            #         }
            #     )

            continue  # Continue looking for more merges after bridging

        # Add the completed turn to the list
        turns.append(
            {
                "Speaker": speaker,
                "Start_Sec": current["start"],
                "End_Sec": current["end"],
                "Duration_Sec": current["duration"],
                "Turn_Type": "T",
            }
        )

        # Mark merged indices as consumed
        consumed_indices.update(merged_indices)
        i = j

    return pd.DataFrame(turns).sort_values(by="Start_Sec").reset_index(drop=True)
