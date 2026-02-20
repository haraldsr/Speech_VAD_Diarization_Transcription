"""
Based on Conversational_speech_labeling_pipeline by Hanlu He.

https://github.com/hanlululu/Conversational_speech_labeling_pipeline
"""

import re
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import entropy


def _refresh_duration_sec(row: pd.Series) -> None:
    """Update duration_sec in-place when the column exists."""
    if "duration_sec" in row.index:
        row["duration_sec"] = row["end_sec"] - row["start_sec"]


def compute_entropy(text: str, by: str = "word") -> float:
    """
    Compute the entropy of a text string.

    Entropy measures the diversity of tokens in the text.

    Args:
        text: The input text string.
        by: Tokenization method, 'word' or 'char'.

    Returns:
        Entropy value in bits, or 0.0 for empty or single-token text.
    """
    text = str(text).strip().lower()
    if not text:
        return 0.0

    if by == "word":
        # [^\W_] matches Unicode letters/digits (word chars) excluding underscore.
        tokens = re.findall(r"[^\W_]+(?:[-'][^\W_]+)*'?", text, flags=re.UNICODE)
    else:
        tokens = list(text)

    if len(tokens) <= 1:
        return 0.0

    counter = Counter(tokens)
    probs = np.array([v / len(tokens) for v in counter.values()])
    return float(entropy(probs, base=2))


def classify_transcriptions(df: pd.DataFrame, threshold: float = 1.5) -> pd.DataFrame:
    """
    Classify transcriptions as 'backchannel', 'turn', or 'overlapped_turn'
        based on entropy.

    Low entropy indicates repetitive/short responses (backchannels),
    high entropy indicates more diverse content (turns).

    Additionally detects overlapped turns: when one speaker's turn is completely
    enveloped (temporally contained) within another speaker's turn, the contained
    turn is classified as 'overlapped_turn'.

    Args:
        df: DataFrame with 'transcription', 'speaker', 'start_sec', 'end_sec' columns.
        threshold: Entropy threshold for classification (default: 1.5).

    Returns:
        DataFrame with added 'entropy' and 'type' columns.
        'type' will be one of: 'backchannel', 'turn', or 'overlapped_turn'.
    """
    df = df.copy()
    df["entropy"] = df["transcription"].apply(lambda x: compute_entropy(x, "word"))
    df["type"] = df["entropy"].apply(
        lambda x: "backchannel" if x < threshold else "turn"
    )

    # Mark turns fully contained within any other speaker's turn.
    turns = df[df["type"] == "turn"]
    overlapped_indices = []

    for idx, row in turns.iterrows():
        contained_by_other_turn = (
            (turns.index != idx)
            & (turns["speaker"] != row["speaker"])
            & (turns["start_sec"] <= row["start_sec"])
            & (turns["end_sec"] >= row["end_sec"])
        ).any()

        if contained_by_other_turn:
            overlapped_indices.append(idx)

    df.loc[overlapped_indices, "type"] = "overlapped_turn"

    return df


def merge_turns_with_context(
    df: pd.DataFrame, max_backchannel_dur: float = 1.0, max_gap_sec: float = 3.0
) -> pd.DataFrame:
    """
    Merge same-speaker turns separated only by short backchannels.

    Simple post-processing step after windowed turn merging and transcription.
    Merges turns that are separated by backchannels (from any speaker) that are
    short enough, without re-applying duration/gap rules from windowed merging.

    Works with any number of speakers.

    Args:
        df: DataFrame with 'start_sec', 'end_sec', 'speaker', 'type',
            'transcription' columns.
            'type' should be 'backchannel', 'turn', or
                'overlapped_turn' (from classify_transcriptions).
        max_backchannel_dur: Maximum duration (seconds) for backchannels
        to allow merging across.
        max_gap_sec: Maximum time gap (seconds) between turns to consider merging.

    Returns:
        DataFrame with merged turns. Backchannels are preserved as separate entries.
    """
    if df.empty:
        return pd.DataFrame(
            columns=["speaker", "start_sec", "end_sec", "transcription", "type"]
        )

    df = df.sort_values("start_sec").reset_index(drop=True)
    
    # Ensure transcription column exists (fill with empty strings if missing)
    if "transcription" not in df.columns:
        df["transcription"] = ""
    
    merged = []
    processed = set()

    for i in range(len(df)):
        if i in processed:
            continue

        current = df.iloc[i].copy()
        _refresh_duration_sec(current)

        if current["type"] == "backchannel" or current["type"] == "overlapped_turn":
            merged.append(current)
            processed.add(i)
            continue

        # Current is a turn - find all consecutive same-speaker turns to merge,
        # skipping short backchannels/overlapped turns in between,
        # without re-applying windowed merging rules.
        speaker = current["speaker"]
        j = i + 1

        while j < len(df):
            next_segment = df.iloc[j]

            # Skip segments that are fully overlapped (end before/at current turn ends)
            if next_segment["end_sec"] <= current["end_sec"]:
                j += 1
                continue

            if next_segment["speaker"] == speaker:
                # Check if can merge

                if next_segment["start_sec"] - current["end_sec"] > max_gap_sec:
                    break

                # Only check segments that extend beyond current turn
                #  (not fully overlapped)
                between = df.iloc[i + 1 : j]
                between = between[between["end_sec"] > current["end_sec"]]
                can_merge = len(between) == 0 or all(
                    (
                        (between["type"] == "backchannel")
                        | (between["type"] == "overlapped_turn")
                    )
                    & (
                        (between["end_sec"] - between["start_sec"])
                        <= max_backchannel_dur
                    )
                )

                if can_merge:
                    # Merge
                    current["end_sec"] = next_segment["end_sec"]
                    # Handle transcription merging safely (convert to string, handle NaN)
                    if "transcription" in current.index and "transcription" in next_segment.index:
                        curr_text = str(current["transcription"]).strip() if pd.notna(current["transcription"]) else ""
                        next_text = str(next_segment["transcription"]).strip() if pd.notna(next_segment["transcription"]) else ""
                        current["transcription"] = (curr_text + " " + next_text).strip()
                    _refresh_duration_sec(current)
                    # Mark the merged turn as processed
                    processed.add(j)
                    j += 1
                    continue
                else:
                    break
            elif (
                next_segment["type"] == "backchannel"
                or next_segment["type"] == "overlapped_turn"
            ):
                # Skip short backchannels from other speakers
                if (
                    next_segment["end_sec"] - next_segment["start_sec"]
                ) <= max_backchannel_dur:
                    j += 1
                    continue
                else:
                    break
            else:
                # Different speaker turn (not overlapping)
                break

        # Append the merged turn
        merged.append(current)
        # Mark i as processed
        processed.add(i)

    out = pd.DataFrame(merged).reset_index(drop=True)
    if "duration_sec" in out.columns:
        out["duration_sec"] = out["end_sec"] - out["start_sec"]
    return out
