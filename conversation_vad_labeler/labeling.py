import numpy as np
import pandas as pd
import re
from collections import Counter
from scipy.stats import entropy

def compute_entropy(text: str, by: str = 'word') -> float:
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

    if by == 'word':
        # [^\W_] matches Unicode letters/digits (word chars) excluding underscore.
        tokens = re.findall(r"[^\W_]+(?:[-'][^\W_]+)*'?", text, flags=re.UNICODE)
    else:
        tokens = list(text)

    if len(tokens) <= 1:
        return 0.0

    counter = Counter(tokens)
    probs = np.array([v / len(tokens) for v in counter.values()])
    return entropy(probs, base=2)


def classify_transcriptions(df: pd.DataFrame, threshold: float = 1.5) -> pd.DataFrame:
    """
    Classify transcriptions as 'backchannel' or 'turn' based on entropy.

    Low entropy indicates repetitive/short responses (backchannels),
    high entropy indicates more diverse content (turns).

    Args:
        df: DataFrame with 'transcription' column.
        threshold: Entropy threshold for classification.

    Returns:
        DataFrame with added 'entropy' and 'type' columns.
    """
    df = df.copy()
    df['entropy'] = df['transcription'].apply(lambda x: compute_entropy(x, 'word'))
    df['type'] = df['entropy'].apply(lambda x: 'backchannel' if x < threshold else 'turn')
    return df


def merge_turns_with_context(
    df: pd.DataFrame,
    max_backchannel_dur: float = 1.0,
    max_gap_sec: float = 2.0
) -> pd.DataFrame:
    """
    Merge same-speaker turns separated only by short backchannels.

    Simple post-processing step after windowed turn merging and transcription.
    Merges turns that are separated by backchannels (from any speaker) that are
    short enough, without re-applying duration/gap rules from windowed merging.

    Works with any number of speakers.

    Args:
        df: DataFrame with 'start_sec', 'end_sec', 'speaker', 'type', 'transcription' columns.
            'type' should be 'backchannel' or 'turn' (from classify_transcriptions).
        max_backchannel_dur: Maximum duration (seconds) for backchannels to allow merging across.
        max_gap_sec: Maximum time gap (seconds) between turns to consider merging.

    Returns:
        DataFrame with merged turns. Backchannels are preserved as separate entries.
    """
    if df.empty:
        return pd.DataFrame(columns=['speaker', 'start_sec', 'end_sec', 'transcription', 'type'])

    df = df.sort_values('start_sec').reset_index(drop=True)
    merged = []
    i = 0

    while i < len(df):
        current = df.iloc[i].copy()

        # Backchannels are never merged, just kept as-is
        if current['type'] == 'backchannel':
            merged.append(current)
            i += 1
            continue

        # Current is a turn - look for same-speaker turns to merge with
        speaker = current['speaker']
        j = i + 1

        # Look ahead for same-speaker turns
        while j < len(df):
            next_segment = df.iloc[j]

            # Can't merge past turns from other speakers
            if next_segment['type'] == 'turn' and next_segment['speaker'] != speaker:
                break

            # Skip backchannels (allow merging across them if short)
            if next_segment['type'] != 'turn':
                j += 1
                continue

            # Now it's a same-speaker turn - check if we can merge
            gap = next_segment['start_sec'] - current['end_sec']
            if gap > max_gap_sec:
                break  # Gap too large, stop looking

            # Check what's between current and next_segment
            between = df[
                (df.index > i) &
                (df.index < j) &
                (df['start_sec'] >= current['end_sec']) &
                (df['end_sec'] <= next_segment['start_sec'])
            ]

            # Can merge if:
            # 1. Nothing in between, OR
            # 2. Only short backchannels in between (from any speaker)
            can_merge = (
                len(between) == 0 or
                all(
                    (between['type'] == 'backchannel') &
                    ((between['end_sec'] - between['start_sec']) <= max_backchannel_dur)
                )
            )

            if can_merge:
                # Merge next_segment into current
                current['end_sec'] = next_segment['end_sec']
                current['transcription'] = (
                    current['transcription'].strip() + " " + 
                    next_segment['transcription'].strip()
                )
                j += 1
            else:
                # Can't merge, stop looking
                break

        merged.append(current)
        i = j

    return pd.DataFrame(merged).reset_index(drop=True)