"""
Compute turn errors between ground truth and estimated turns.

Based on MATLAB implementation compute_turn_errors.m.
"""

from __future__ import annotations

import warnings
from typing import Any, Tuple, cast

import numpy as np
import pandas as pd


def _maybe_warn(message: str, suppress_warnings: bool) -> None:
    if not suppress_warnings:
        warnings.warn(message)


def _maybe_print(message: str, suppress_prints: bool) -> None:
    if not suppress_prints:
        print(message)


def compute_overlap_ratio(
    s_ref: Tuple[float, float],
    s_est: Tuple[float, float],
) -> float:
    """
    Compute the overlap ratio between two time segments.

    Args:
        s_ref: (start_time, end_time) for reference signal
        s_est: (start_time, end_time) for estimated signal

    Returns:
        Overlap ratio as intersection / union signal duration.
    """
    # Sort segments to ensure compatibility with negative FTOs
    s_ref = cast(Tuple[float, float], tuple(sorted(s_ref)))
    s_est = cast(Tuple[float, float], tuple(sorted(s_est)))

    # Compute overlap ratio as intersection / union
    intersection_duration = max(0.0, min(s_ref[1], s_est[1]) - max(s_ref[0], s_est[0]))
    union_duration = max(s_ref[1], s_est[1]) - min(s_ref[0], s_est[0])

    overlap_ratio = intersection_duration / union_duration

    return overlap_ratio

def compute_turn_errors(
    df_ref: pd.DataFrame,
    df_est: pd.DataFrame,
    min_overlap_ratio: float | dict[str, float],
    suppress_warnings: bool = False,
    suppress_prints: bool = False,
) -> pd.DataFrame:
    """
    Compute turn errors between ground truth and estimated turns.

    Matches ground truth turns to estimated turns based on overlap ratio.
    Handles multiple matches by keeping the highest overlap ratio match.
    Returns the ground truth table augmented with detection status and
    timing differences, plus any false positive (unmatched estimated) turns.

    Args:
        df_ref: Ground truth turns DataFrame with columns:
            - speaker: Speaker identifier
            - start_sec: Turn start time (seconds)
            - end_sec: Turn end time (seconds)
            - duration_sec: Turn duration (seconds)
            - type: Turn type identifier
        df_est: Estimated turns DataFrame with same columns as df_ref.
                min_overlap_ratio: Minimum overlap ratio required for matching turns.
                        - float: same threshold for all types
                        - dict: per-type thresholds, e.g.
                            {"turn": 0.2, "backchannel": 0.05, "default": 0.1}
        suppress_warnings: If True, suppresses warning messages. Defaults to False.
        suppress_prints: If True, suppresses print messages. Defaults to False.

    Returns:
        DataFrame with ground truth turns augmented with:
            - detected: Boolean indicating if turn was detected
            - delta_t: Duration difference (ref - est) for matched turns, NaN otherwise
        Plus rows for false positives (unmatched estimated turns) with
        detected=NaN and delta_t=NaN.
    """
    df_out = df_ref.copy()
    n_ref = len(df_ref)
    n_est = len(df_est)

    # Initialize overlap ratio matrix
    overlap_matrix = np.full((n_est, n_ref), np.nan)

    # Precompute normalized speaker+type keys to avoid O(n_ref * n_est) Python loops
    ref_speakers_sorted = np.array(["".join(sorted(str(s))) for s in df_ref["speaker"]], dtype=object)
    est_speakers_sorted = np.array(["".join(sorted(str(s))) for s in df_est["speaker"]], dtype=object)
    ref_keys = np.char.add(df_ref["type"].astype(str).to_numpy(), "||")
    ref_keys = np.char.add(ref_keys, ref_speakers_sorted)
    est_keys = np.char.add(df_est["type"].astype(str).to_numpy(), "||")
    est_keys = np.char.add(est_keys, est_speakers_sorted)

    ref_starts = df_ref["start_sec"].to_numpy(dtype=float)
    ref_ends = df_ref["end_sec"].to_numpy(dtype=float)
    est_starts = df_est["start_sec"].to_numpy(dtype=float)
    est_ends = df_est["end_sec"].to_numpy(dtype=float)

    ref_group_indices = {
        key: np.flatnonzero(ref_keys == key)
        for key in np.unique(ref_keys)
    }

    for key in np.unique(est_keys):
        if key not in ref_group_indices:
            continue

        est_idx = np.flatnonzero(est_keys == key)
        ref_idx = ref_group_indices[key]

        if len(est_idx) == 0 or len(ref_idx) == 0:
            continue

        ref_start_block = ref_starts[ref_idx][np.newaxis, :]
        ref_end_block = ref_ends[ref_idx][np.newaxis, :]
        est_start_block = est_starts[est_idx][:, np.newaxis]
        est_end_block = est_ends[est_idx][:, np.newaxis]

        intersection_duration = np.maximum(
            0.0,
            np.minimum(ref_end_block, est_end_block) - np.maximum(ref_start_block, est_start_block),
        )
        union_duration = np.maximum(ref_end_block, est_end_block) - np.minimum(ref_start_block, est_start_block)

        overlap_block = np.divide(
            intersection_duration,
            union_duration,
            out=np.full_like(intersection_duration, np.nan),
            where=union_duration > 0,
        )

        overlap_matrix[np.ix_(est_idx, ref_idx)] = overlap_block

    # Compute duration differences
    duration_delta = (
        df_ref["duration_sec"].values[np.newaxis, :]
        - df_est["duration_sec"].values[:, np.newaxis]
    )

    start_delta = (
        df_ref["start_sec"].values[np.newaxis, :]
        - df_est["start_sec"].values[:, np.newaxis]
    )

    end_delta = (
        df_ref["end_sec"].values[np.newaxis, :]
        - df_est["end_sec"].values[:, np.newaxis]
    )

    # Build match matrix (supports global or per-type overlap thresholds)
    if isinstance(min_overlap_ratio, dict):
        type_thresholds = {str(key): float(value) for key, value in min_overlap_ratio.items()}
        default_threshold = float(type_thresholds.get("default", 0.0))
    else:
        default_threshold = float(min_overlap_ratio)
        type_thresholds = {}

    match_matrix = np.zeros_like(overlap_matrix, dtype=bool)
    ref_types = df_ref["type"].astype(str).to_numpy()
    est_types = df_est["type"].astype(str).to_numpy()
    all_types = set(ref_types.tolist()) | set(est_types.tolist())

    for type_name in all_types:
        est_mask = est_types == type_name
        ref_mask = ref_types == type_name
        if not np.any(est_mask) or not np.any(ref_mask):
            continue

        threshold = float(type_thresholds.get(type_name, default_threshold))
        match_matrix[np.ix_(est_mask, ref_mask)] = (
            overlap_matrix[np.ix_(est_mask, ref_mask)] >= threshold
        )

    # Count matches per reference and estimated turn
    matches_to_ref = np.sum(match_matrix, axis=1)
    matches_to_est = np.sum(match_matrix, axis=0)

    # Resolve multiple matches to ground truth events
    if not np.all(matches_to_ref <= 1):
        _maybe_warn("Multiple matches found for a single ground truth event.", suppress_warnings)

        for i_est in range(n_est):
            if matches_to_ref[i_est] > 1:
                _maybe_print(
                    f"Ground truth event {i_est} has "
                    f"{int(matches_to_ref[i_est])} matches.",
                    suppress_prints,
                )
                _maybe_print(
                    "Keeping only the closest match in terms of overlapping ratio.",
                    suppress_prints,
                )

                # Find best match (highest overlap ratio among matches)
                masked_overlap = np.where(
                    match_matrix[i_est, :], overlap_matrix[i_est, :], -np.inf
                )
                closest_match = np.argmax(masked_overlap)
                match_matrix[i_est, :] = False
                match_matrix[i_est, closest_match] = True

        _maybe_print("----------------------------", suppress_prints)

    # Resolve multiple matches to estimated events
    if not np.all(matches_to_est <= 1):
        _maybe_warn("Multiple matches found for a single estimated event.", suppress_warnings)

        for i_ref in range(n_ref):
            if matches_to_est[i_ref] > 1:
                _maybe_print(
                    f"Estimated event {i_ref} has "
                    f"{int(matches_to_est[i_ref])} matches.",
                    suppress_prints,
                )
                _maybe_print(
                    "Keeping only the closest match in terms of overlapping ratio.",
                    suppress_prints,
                )

                # Find best match (highest overlap ratio among matches)
                masked_overlap = np.where(
                    match_matrix[:, i_ref], overlap_matrix[:, i_ref], -np.inf
                )
                closest_match = np.argmax(masked_overlap)
                match_matrix[:, i_ref] = False
                match_matrix[closest_match, i_ref] = True

        _maybe_print("----------------------------", suppress_prints)

    # Build error matrix with duration differences for matched turns
    duration_delta_matrix = np.full_like(duration_delta, np.nan)
    duration_delta_matrix[match_matrix] = duration_delta[match_matrix]

    start_delta_matrix = np.full_like(start_delta, np.nan)
    start_delta_matrix[match_matrix] = start_delta[match_matrix]

    end_delta_matrix = np.full_like(end_delta, np.nan)
    end_delta_matrix[match_matrix] = end_delta[match_matrix]

    # Handle true positives and false negatives
    detected_mask = np.any(match_matrix, axis=0)
    duration_delta_sum = np.nansum(duration_delta_matrix, axis=0)
    start_delta_sum = np.nansum(start_delta_matrix, axis=0)
    end_delta_sum = np.nansum(end_delta_matrix, axis=0)

    df_out["detected"] = detected_mask
    df_out["duration_delta"] = np.where(detected_mask, duration_delta_sum, np.nan)
    df_out["start_delta"] = np.where(detected_mask, start_delta_sum, np.nan)
    df_out["end_delta"] = np.where(detected_mask, end_delta_sum, np.nan)

    # Handle false positives (unmatched estimated turns)
    false_positive_indices = np.flatnonzero(~np.any(match_matrix, axis=1))
    if len(false_positive_indices) > 0:
        df_fp = df_est.iloc[false_positive_indices][
            ["speaker", "start_sec", "end_sec", "duration_sec", "type"]
        ].copy()
        df_fp["detected"] = np.nan
        df_fp["duration_delta"] = np.nan
        df_fp["start_delta"] = np.nan
        df_fp["end_delta"] = np.nan
        df_out = pd.concat([df_out, df_fp], ignore_index=True)

    return df_out

def tabulate_floor_transfers(
    df_turns: pd.DataFrame,
    suppress_warnings: bool = False,
    suppress_prints: bool = False,
) -> pd.DataFrame:
    """
    Extract floor transfer events from a turns table.

    Floor transfers are the gaps between consecutive turns from different
    speakers. This function filters for turns only and creates a table of
    floor transfer events.

    Args:
        df_turns: Turns DataFrame with columns:
            - speaker: Speaker identifier
            - start_sec: Turn start time (seconds)
            - end_sec: Turn end time (seconds)
            - duration_sec: Turn duration (seconds)
            - type: Turn type identifier ("turn" for turns)
        suppress_warnings: If True, suppresses warning messages. Defaults to False.
        suppress_prints: If True, suppresses print messages. Defaults to False.

    Returns:
        DataFrame with floor transfer events containing:
            - speaker: Combined speaker identifiers (e.g., "P1-P2")
            - start_sec: Floor transfer start time (end of previous turn)
            - end_sec: Floor transfer end time (start of next turn)
            - duration_sec: Floor transfer duration
            - type: "FTO" (floor transfer offset)
    """
    # Filter for turns only
    turns = df_turns[df_turns["type"] == "turn"].copy()
    turns = turns.sort_values(by="start_sec").reset_index(drop=True)

    n_turns = len(turns)
    floor_transfers = []  # Collect rows in a list instead of repeatedly concatenating

    for i_turn in range(n_turns - 1):
        speaker_current = turns.iloc[i_turn]["speaker"]
        speaker_next = turns.iloc[i_turn + 1]["speaker"]

        # Floor transfers only occur between different speakers
        if speaker_current == speaker_next:
            _maybe_warn(
                f"Consecutive turns by the same speaker found at index {i_turn}. "
                "Expected alternating speakers for floor transfers.",
                suppress_warnings,
            )
        else:
            # Create floor transfer entry
            speakers = f"{speaker_current}-{speaker_next}"
            t_start = turns.iloc[i_turn]["end_sec"]
            t_end = turns.iloc[i_turn + 1]["start_sec"]
            duration = t_end - t_start

            floor_transfers.append({
                "speaker": speakers,
                "start_sec": t_start,
                "end_sec": t_end,
                "duration_sec": duration,
                "type": "FTO",
            })

    # Create DataFrame from collected rows (more efficient and avoids FutureWarning)
    df_fto = pd.DataFrame(floor_transfers) if floor_transfers else pd.DataFrame(columns=["speaker", "start_sec", "end_sec", "duration_sec", "type"])

    return df_fto

def print_error_summary(err: dict, suppress_prints: bool = False) -> None:
    """
    Print a summary of turn error metrics.

    Args:
        err: Dictionary containing error metrics for each turn type.
        suppress_prints: If True, suppresses print messages. Defaults to False.
    """
    for type, metrics in err.items():
        _maybe_print("-" * 60, suppress_prints)
        _maybe_print(f"{type} Precision: {metrics['precision']:.2f}", suppress_prints)
        _maybe_print(f"{type} Recall: {metrics['recall']:.2f}", suppress_prints)
        _maybe_print(
            f"{type} Mean Absolute Duration Delta: "
            f"{metrics['duration_delta'][1]:.2f} seconds [5%: {metrics['duration_delta'][0]:.2f}s, 95%: {metrics['duration_delta'][2]:.2f}s]",
            suppress_prints,
        )
        _maybe_print(
            f"{type} Mean Absolute Start Delta: "
            f"{metrics['start_delta'][1]:.2f} seconds [5%: {metrics['start_delta'][0]:.2f}s, 95%: {metrics['start_delta'][2]:.2f}s]",
            suppress_prints,
        )
        _maybe_print(
            f"{type} Mean Absolute End Delta: "
            f"{metrics['end_delta'][1]:.2f} seconds [5%: {metrics['end_delta'][0]:.2f}s, 95%: {metrics['end_delta'][2]:.2f}s]",
            suppress_prints,
        )
        _maybe_print(
            f"{type} Mean Absolute Start/End Delta: "
            f"{metrics['start_end'][1]:.2f} seconds [5%: {metrics['start_end'][0]:.2f}s, 95%: {metrics['start_end'][2]:.2f}s]",
            suppress_prints,
        )

def compute_all_errors(
    df_ref: pd.DataFrame,
    df_est: pd.DataFrame,
    min_overlap_ratio: float | dict[str, float],
    suppress_warnings: bool = False,
    suppress_prints: bool = False,
) -> Tuple[dict, pd.DataFrame]:
    """
    Compute turn errors for both turns and floor transfers.

    Args:
        df_ref: Ground truth turns DataFrame.
        df_est: Estimated turns DataFrame.
        min_overlap_ratio: Minimum overlap ratio for matching turns.
            - float: same threshold for all types
            - dict: per-type thresholds (e.g., turn/backchannel)
        suppress_warnings: If True, suppresses warning messages. Defaults to False.
        suppress_prints: If True, suppresses print messages. Defaults to False.

    Returns:
        Tuple of (metrics_dict, errors_dataframe) where metrics_dict contains
        precision, recall, and mean timing deltas for each turn type.
    """

    df_fto_ref = tabulate_floor_transfers(df_ref, suppress_warnings, suppress_prints)
    df_fto_est = tabulate_floor_transfers(df_est, suppress_warnings, suppress_prints)

    df_ref_cat = pd.concat([df_ref, df_fto_ref], ignore_index=True)
    df_est_cat = pd.concat([df_est, df_fto_est], ignore_index=True)
                           
    err_df = compute_turn_errors(
        df_ref_cat,
        df_est_cat,
        min_overlap_ratio,
        suppress_warnings,
        suppress_prints,
    )

    # Compute summary metrics for each turn type
    err: dict = {}
    for type in err_df["type"].unique():
        type_df = err_df[err_df["type"] == type]

        # Detection statistics
        tp = (type_df["detected"] == 1).sum()
        fn = (type_df["detected"] == 0).sum()
        fp = type_df["detected"].isna().sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # err.update({type: {"precision": precision, "recall": recall}})
        err[type] = {"precision": precision, "recall": recall}

        # Time error metrics (only for detected turns)
        detected_df = type_df[type_df["detected"] == True] 

        for delta_col in ["duration_delta", "start_delta", "end_delta"]:
            mean_abs = detected_df[delta_col].abs().mean() if len(detected_df) > 0 else float("nan")
            p975_abs = detected_df[delta_col].abs().quantile(0.975) if len(detected_df) > 0 else float("nan")
            p025_abs = detected_df[delta_col].abs().quantile(0.025) if len(detected_df) > 0 else float("nan")
            
            err[type][delta_col] = np.array([p025_abs, mean_abs, p975_abs])

        # Time error metrics for start and end combined
        mean_abs_start_end = np.mean(np.concatenate(
            [detected_df["start_delta"].abs().values, detected_df["end_delta"].abs().values]
        )) if len(detected_df) > 0 else float("nan")

        p975_abs_start_end = np.quantile(np.concatenate(
            [detected_df["start_delta"].abs().values, detected_df["end_delta"].abs().values]), 0.975
        ) if len(detected_df) > 0 else float("nan")

        p025_abs_start_end = np.quantile(np.concatenate(
            [detected_df["start_delta"].abs().values, detected_df["end_delta"].abs().values]), 0.025
        ) if len(detected_df) > 0 else float("nan")

        err[type]["start_end"] = np.array([p025_abs_start_end, mean_abs_start_end, p975_abs_start_end])
    return err, err_df

def replace_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Replace common label variations in the DataFrame."""

    df = df.replace(
        {
            # Common speeling variations for speakers and types
            "speaker": {"Talker1": "P1","p1":"P1", "Talker2": "P2","p2":"P2"},
            "type": {"T": "turn","t": "turn", "B": "backchannel", "b": "backchannel"},
        }
    )

    # Treat overlap labels turns as backchannels
    df = df.replace({"type": {"overlap": "backchannel", "overlapped_turn": "backchannel"}})

    return df

def validate_turns(
    df: pd.DataFrame,
    suppress_warnings: bool = False,
    suppress_prints: bool = False,
) -> pd.DataFrame:
    _ = suppress_warnings
    
    _maybe_print("="*70, suppress_prints)
    _maybe_print("Validating column names, labels, durations, and self-intersecting events.", suppress_prints)
    _maybe_print("="*70, suppress_prints)
    df = check_colnames(df, suppress_prints=suppress_prints)
    df = check_labels(df, suppress_prints=suppress_prints)
    df = check_durations(df, suppress_prints=suppress_prints)
    df = check_self_intersection(df, suppress_prints=suppress_prints)

    return df

def check_colnames(df: pd.DataFrame, suppress_prints: bool = False) -> pd.DataFrame:
    _maybe_print("-"*60 + "\nChecking required columns in the DataFrame...", suppress_prints)
    required_columns = {"speaker", "start_sec", "end_sec", "duration_sec", "type"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    else:
        _maybe_print("All required columns are present.", suppress_prints)
    return df

def check_labels(df: pd.DataFrame, suppress_prints: bool = False) -> pd.DataFrame:
    _maybe_print("-"*60 + "\nChecking for valid speaker and type labels...", suppress_prints)
    allowed_speaker_labels = {"P1", "P2"}
    allowed_type_labels = {"turn", "backchannel"}
    if not set(df["speaker"].unique()).issubset(allowed_speaker_labels):
        _maybe_print("Unexpected speaker labels found:", suppress_prints)
        for label in set(df["speaker"].unique()) - allowed_speaker_labels:
            _maybe_print(f" > {label}", suppress_prints)
        _maybe_print("Attempting replacement with common variations.", suppress_prints)
    if not set(df["type"].unique()).issubset(allowed_type_labels):
        _maybe_print("Unexpected type labels found:", suppress_prints)
        for label in set(df["type"].unique()) - allowed_type_labels:
            _maybe_print(f" > {label}", suppress_prints)
        _maybe_print("Attempting replacement with common variations.", suppress_prints)
    else:
        _maybe_print("All labels are valid. No replacement performed.", suppress_prints)
    
    # Replace common label variations with allowed labels
    df = df.replace(
        {
            "speaker": {"Talker1": "P1",
                        "p1": "P1",
                        "Talker2": "P2",
                        "p2": "P2",
                        "P1_backchannel": "P1",
                        "P2_backchannel": "P2",},
            "type": {"T": "turn",
                     "t": "turn",
                     "B": "backchannel",
                     "b": "backchannel",
                     "OT": "backchannel",
                     "B + L": "backchannel",
                     "overlap": "backchannel",
                     "overlapped_turn": "backchannel"},
        }
    )

    if not set(df["speaker"].unique()).issubset(allowed_speaker_labels):
        raise ValueError(f"Speaker labels could not be replaced. Remaining invalid labels: {set(df['speaker'].unique()) - allowed_speaker_labels}.")
    elif not set(df["type"].unique()).issubset(allowed_type_labels):
        raise ValueError(f"Type labels could not be replaced. Remaining invalid labels: {set(df['type'].unique()) - allowed_type_labels}.")
    else:
        _maybe_print("Speaker and type labels successfully replaced.", suppress_prints)
        
    return df

def check_durations(df: pd.DataFrame, suppress_prints: bool = False) -> pd.DataFrame:
    _maybe_print("-"*60 + "\nChecking duration consistency and positivity...", suppress_prints)
    df["duration_check"] = df["end_sec"] - df["start_sec"] - df["duration_sec"]
    msk_mis = np.abs(df["duration_check"]) >= 1e-3
    if np.any(msk_mis):
        _maybe_print("Duration inconsistencies found (end_sec-start_sec does not match duration_sec):", suppress_prints)
        _maybe_print(df[msk_mis].to_string(), suppress_prints)
        _maybe_print("Replacing duration_sec with end_sec-start_sec for inconsistent entries.", suppress_prints)
        df.loc[msk_mis, "duration_sec"] = df.loc[msk_mis, "end_sec"] - df.loc[msk_mis, "start_sec"]
    else:
        _maybe_print("All durations are consistent.", suppress_prints)

    # Check if all durations are positive
    if np.any(df["duration_sec"] <= 0):
        _maybe_print("\nNon-positive durations found:", suppress_prints)
        _maybe_print(df[df["duration_sec"] <= 0].to_string(), suppress_prints)
        _maybe_print("Removing entries with non-positive durations.", suppress_prints)
        df = df[df["duration_sec"] > 0]
    else:
        _maybe_print("All durations are positive.", suppress_prints)
    
    return df.drop(columns=["duration_check"])

def check_self_intersection(df: pd.DataFrame, suppress_prints: bool = False) -> pd.DataFrame:
    _maybe_print("-"*60 + "\nChecking for self-intersecting events...", suppress_prints)
    df["self_intersect"] = ""
    for i in df.index:
        speaker = df.loc[i, "speaker"]

        msk_spk = df["speaker"] == speaker
        msk_prev = df["start_sec"] < df.loc[i, "start_sec"]

        msk_previous_selfoverlap = find_intersecting_events(df, 
                                               t1 = df.loc[i, "start_sec"], 
                                               t2 = df.loc[i, "end_sec"],
                                               mask = msk_spk & msk_prev)

        if np.any(msk_previous_selfoverlap):
            df.loc[i, "self_intersect"] = "_".join(map(str, df.index[msk_previous_selfoverlap].tolist()))

    if np.any(df["self_intersect"] != ""):
        _maybe_print("\nSelf-intersecting entries found (same speaker with overlapping time intervals):", suppress_prints)
        _maybe_print(df[df["self_intersect"] != ""].to_string(), suppress_prints)
        _maybe_print("Keeping only first entry and removing self-intersecting entries for each speaker.", suppress_prints)
        # Remove self-intersecting entries
        df = df[df["self_intersect"] == ""].drop(columns=["self_intersect"])
    else:
        _maybe_print("No self-intersecting entries found.", suppress_prints)
    
    return df

def find_embedding_events(df: pd.DataFrame,t_start,t_end,mask = None) -> bool:
    # Find events that fully embed the given time interval and match the mask
    if mask is None:
        mask = pd.Series([True] * len(df))

    idx_embedding = (df["start_sec"] <= t_start) & (df["end_sec"] >= t_end) & mask
  
    return idx_embedding

def merge_events(df: pd.DataFrame, idx1, idx2) -> pd.DataFrame:
    # Merge two events in the DataFrame by updating the first event's end time and duration, and marking the second event for deletion
    df.loc[idx1, "end_sec"] = max(df.loc[idx1, "end_sec"], df.loc[idx2, "end_sec"])
    df.loc[idx1, "duration_sec"] = df.loc[idx1, "end_sec"] - df.loc[idx1, "start_sec"]
    df.loc[idx2, "merged"] = True  # Mark the second event for deletion

    # Merge transciption labels if they exist
    if "transcription" in df.columns:
        df.loc[idx1, "transcription"] = f"{df.loc[idx1, 'transcription']} {df.loc[idx2, 'transcription']}"
    return df

def find_intersecting_events(df: pd.DataFrame,t1,t2,mask = None) -> bool:
    # Find events that overlap with the given time interval and match the mask
    if mask is None:
        mask = pd.Series(True, index=df.index)

    t_start = min(t1, t2)
    t_end = max(t1, t2)

    starts = df["start_sec"].to_numpy()
    ends = df["end_sec"].to_numpy()
    mask_values = np.asarray(mask, dtype=bool)

    intersection = np.maximum(0.0, np.minimum(ends, t_end) - np.maximum(starts, t_start))
    idx_intersecting = pd.Series((intersection > 0.0) & mask_values, index=df.index)
  
    return idx_intersecting

def plot_turns(*dfs: pd.DataFrame, titles: list[str] | None = None, xlim: tuple[float, float] | None = None) -> None:
  import matplotlib.pyplot as plt

  # Handle single DataFrame or multiple DataFrames
  if len(dfs) == 1 and isinstance(dfs[0], pd.DataFrame):
    dfs = (dfs[0],)
  
  if titles is None:
    titles = [f"Turns Plot {i+1}" if len(dfs) > 1 else "Turns Plot" for i in range(len(dfs))]
  elif len(titles) != len(dfs):
    raise ValueError(f"Number of titles ({len(titles)}) must match number of DataFrames ({len(dfs)})")

  fig, axes = plt.subplots(len(dfs), 1, figsize=(12, 3 * len(dfs)), sharex=True)
  if len(dfs) == 1:
    axes = [axes]

  for ax, df, title in zip(axes, dfs, titles):
    df = df.sort_values(by="start_sec").reset_index(drop=True)
    speakers = df["speaker"].unique()
    turn_types = df["type"].unique()
    colors = plt.get_cmap("tab10", len(speakers))

    y_pos = 0
    y_ticks = []
    y_labels = []
    
    for i, speaker in enumerate(speakers):
      speaker_df = df[df["speaker"] == speaker]
      for turn_type in sorted(turn_types):
        type_df = speaker_df[speaker_df["type"] == turn_type]
        for _, turn in type_df.iterrows():
          jitter = np.random.uniform(-0.1, 0.1) # Add jitter to show self-intersections
          ax.plot(
            [turn["start_sec"], turn["end_sec"]],
            [y_pos + jitter, y_pos + jitter],
            color=colors(i),
            linewidth=10,
            solid_capstyle="butt",
            linestyle="-"
          )
        y_ticks.append(y_pos)
        y_labels.append(f"{speaker}\n({turn_type})")
        y_pos += 0.5
      y_pos += 1

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_title(title)
    ax.grid(True, axis="x")
    
    if xlim is not None:
      ax.set_xlim(xlim)

  axes[-1].set_xlabel("Time (seconds)")
  plt.tight_layout()
  plt.show()
    
def postprocess_turn_df(
    df: pd.DataFrame,
    max_iter = 20,
    rec_iter = 0,
    suppress_warnings: bool = False,
    suppress_prints: bool = False,
) -> pd.DataFrame:
    speakers = ["P1", "P2"] # TO DO: cover any speaker amount

    def _active_rows_mask(frame: pd.DataFrame) -> pd.Series:
        if "merged" in frame.columns:
            return frame["merged"] != True
        return pd.Series(True, index=frame.index)

    def _merge_same_speaker_segments(
        frame: pd.DataFrame,
        speaker: str,
        segment_type: str,
        blocker_mask: pd.Series,
    ) -> tuple[pd.DataFrame, int]:
        merged_count = 0
        active_mask = _active_rows_mask(frame)
        idx_segments = (
            frame[(frame["speaker"] == speaker) & active_mask & (frame["type"] == segment_type)]
            .sort_values(by="start_sec")
            .index
            .tolist()
        )

        i = 0
        while i < len(idx_segments) - 1:
            idx_anchor = idx_segments[i]
            j = i + 1

            while j < len(idx_segments):
                idx_next = idx_segments[j]
                is_blocked = np.any(
                    find_intersecting_events(
                        frame,
                        t1=frame.loc[idx_anchor, "end_sec"],
                        t2=frame.loc[idx_next, "start_sec"],
                        mask=blocker_mask,
                    )
                )

                if is_blocked:
                    break

                frame = merge_events(frame, idx_anchor, idx_next)
                merged_count += 1
                j += 1

            i = j

        return frame, merged_count

    def _merge_bridge_backchannels_into_turns(
        frame: pd.DataFrame,
        speaker: str,
    ) -> tuple[pd.DataFrame, int]:
        merged_count = 0
        active_mask = _active_rows_mask(frame)
        msk_spk_active = (frame["speaker"] == speaker) & active_mask
        msk_int_any = (frame["speaker"] != speaker) & active_mask

        idx_bc = (
            frame[msk_spk_active & (frame["type"] == "backchannel")]
            .sort_values(by="start_sec")
            .index
            .tolist()
        )
        idx_turn = (
            frame[msk_spk_active & (frame["type"] == "turn")]
            .sort_values(by="start_sec")
            .index
            .tolist()
        )

        if len(idx_bc) == 0 or len(idx_turn) < 2:
            return frame, merged_count

        for bc_idx in idx_bc:
            bc_start = frame.loc[bc_idx, "start_sec"]
            bc_end = frame.loc[bc_idx, "end_sec"]

            prev_turn_candidates = [idx for idx in idx_turn if frame.loc[idx, "end_sec"] <= bc_start]
            next_turn_candidates = [idx for idx in idx_turn if frame.loc[idx, "start_sec"] >= bc_end]
            if not prev_turn_candidates or not next_turn_candidates:
                continue

            prev_turn_idx = prev_turn_candidates[-1]
            next_turn_idx = next_turn_candidates[0]
            if prev_turn_idx == next_turn_idx:
                continue

            if np.any(
                find_intersecting_events(
                    frame,
                    t1=bc_start,
                    t2=bc_end,
                    mask=msk_int_any,
                )
            ):
                continue

            frame.loc[bc_idx, "type"] = "turn"
            frame = merge_events(frame, prev_turn_idx, bc_idx)
            frame = merge_events(frame, prev_turn_idx, next_turn_idx)
            merged_count += 2

            active_mask = _active_rows_mask(frame)
            msk_spk_active = (frame["speaker"] == speaker) & active_mask
            idx_turn = (
                frame[msk_spk_active & (frame["type"] == "turn")]
                .sort_values(by="start_sec")
                .index
                .tolist()
            )

        return frame, merged_count

    # Initialize at first iteration    
    if rec_iter == 0:
        df = replace_labels(df)
        df = df.sort_values(by="start_sec")
        _maybe_print("-"*60, suppress_prints)
    
    _maybe_print(f">>> Starting postprocessing - recursion iteration {rec_iter}", suppress_prints)
    turns_merged = 0

    # 1. Merge consecutive turns by the same speaker if they are not separated by any interlocutor turn or self-backchannel
    for speaker in speakers:
        msk_active = _active_rows_mask(df)
        msk_int = (df["speaker"] == speakers[1 - speakers.index(speaker)]) & msk_active
        msk_turn = df["type"] == "turn"
        msk_spk_bc = (df["speaker"] == speaker) & msk_active & (df["type"] == "backchannel")
        msk_int_turn = msk_int & msk_turn

        df, merged_count = _merge_same_speaker_segments(
            df,
            speaker=speaker,
            segment_type="turn",
            blocker_mask=msk_int_turn | msk_spk_bc,
        )
        turns_merged += merged_count

    # 2. Treat speaker backchannels as turns only if they have no overlap with any other active event
    for speaker in speakers:
        msk_active = _active_rows_mask(df)
        msk_spk = (df["speaker"] == speaker) & msk_active
        msk_bc = df["type"] == "backchannel"

        for idx_bc in df[msk_spk & msk_bc].index:
            msk_other_active = _active_rows_mask(df) & (df.index != idx_bc)
            if not np.any(
                find_intersecting_events(df,
                                         t1=df.loc[idx_bc, "start_sec"],
                                         t2=df.loc[idx_bc, "end_sec"],
                                         mask=msk_other_active)):
                df.loc[idx_bc, "type"] = "turn"
    
    # 3. Treat speaker turns as backchannels if they're embedded in any interlocutor turn or backchannel
    for speaker in speakers:
        msk_active = _active_rows_mask(df)
        msk_spk = (df["speaker"] == speaker) & msk_active
        msk_int = (df["speaker"] == speakers[1 - speakers.index(speaker)]) & msk_active
        msk_turn = df["type"] == "turn"
        
        for idx_turn in df[msk_spk & msk_turn].index:
            if np.any(
                find_embedding_events(df, 
                                      t_start = df.loc[idx_turn, "start_sec"], 
                                      t_end = df.loc[idx_turn, "end_sec"], 
                                      mask = msk_int & msk_turn)):
                df.loc[idx_turn, "type"] = "backchannel"

    # 4. Re-merge after re-labeling (Step 2/3 may have created new merge opportunities)
    for speaker in speakers:
        msk_active = _active_rows_mask(df)
        msk_int = (df["speaker"] == speakers[1 - speakers.index(speaker)]) & msk_active
        msk_turn = df["type"] == "turn"
        msk_spk_bc = (df["speaker"] == speaker) & msk_active & (df["type"] == "backchannel")
        msk_int_turn = msk_int & msk_turn

        # Bridge pattern like t-b-t: merge middle same-speaker backchannel into turns
        df, merged_bridge_count = _merge_bridge_backchannels_into_turns(df, speaker=speaker)
        turns_merged += merged_bridge_count

        # Re-merge turns with same blocker logic as Step 1
        df, merged_turn_count = _merge_same_speaker_segments(
            df,
            speaker=speaker,
            segment_type="turn",
            blocker_mask=msk_int_turn | msk_spk_bc,
        )
        turns_merged += merged_turn_count
    
    # Remove merged rows
    df = df[df["merged"] != True] if "merged" in df.columns else df

    # Mark for recursion if consecutive turns by the same speaker after merging
    recur = False
    idx_turns = df[df["type"] == "turn"].index
    for i in range(len(idx_turns) - 1): 
        current_turn = df.loc[idx_turns[i]]
        next_turn = df.loc[idx_turns[i + 1]]

        if current_turn["speaker"] == next_turn["speaker"]:
            recur = True
            _maybe_warn(
                f"Consecutive turns by the same speaker found at index {idx_turns[i]} and {idx_turns[i + 1]} after merging. Recursing for another iteration.",
                suppress_warnings,
            )
            _maybe_print(
                f"Consecutive turns by the same speaker found at index {idx_turns[i]} and {idx_turns[i + 1]} after merging. Recursing for another iteration.",
                suppress_prints,
            )
            break
        
    _maybe_print(
        f"Post-processing iteration completed with {turns_merged} turns merged after {rec_iter} iterations.\n",
        suppress_prints,
    )

    # Recur unless 1) no consecutive turns found or 2) max iterations reached
    if rec_iter < max_iter:
      if recur:
                return postprocess_turn_df(
                        df,
                        max_iter=max_iter,
                        rec_iter=rec_iter+1,
                        suppress_warnings=suppress_warnings,
                        suppress_prints=suppress_prints,
                )
      else:
                    _maybe_print("No consecutive turns found. Post-processing complete.", suppress_prints)
                    _maybe_print("-"*60, suppress_prints)
    else:        
                _maybe_warn(
                        f"Maximum recursion iterations ({max_iter}) reached. Post-processing terminated.",
                        suppress_warnings,
                )
                _maybe_print(
                        f"Maximum recursion iterations ({max_iter}) reached. Post-processing terminated.",
                        suppress_prints,
                )
                _maybe_print("-"*60, suppress_prints)

    return df


if __name__ == "__main__":
    # Example usage

    # Load example estimated labels
    df_est = pd.read_csv(
        "demo/annotations/demo_labels.txt",
        sep="\t",
        header=None,
        names=["speaker", "foo", "start_sec", "end_sec", "duration_sec", "type"],
    )

    # Validate and post-process estimate
    dfv_est = validate_turns(df_est)
    dfp_est = postprocess_turn_df(dfv_est, max_iter = 20)
    dfp_est.to_csv("outputs/demo/demo_labels_proc.txt", sep="\t", index=True, index_label="index")

    # Plot original, validated, and post-processed estimates
    plot_turns(df_est, dfv_est, dfp_est, titles=["Original turn data - estimate (demo)", "After validation", "After post-processing"])

    # Load example reference labels
    df_ref = pd.read_csv(
        "demo/annotations/F1F2_quiet_food_1m_01_labels_manual_rinor.txt",
        sep="\t",
        header=None,
        names=["speaker", "foo", "start_sec", "end_sec", "duration_sec", "type"],
    ).drop(columns=["foo"])

    # Validate and post-process reference
    dfv_ref = validate_turns(df_ref)
    dfp_ref = postprocess_turn_df(dfv_ref, max_iter = 20)
    dfp_ref.to_csv("outputs/demo/F1F2_quiet_food_1m_01_labels_manual_rinor_proc.txt", sep="\t", index=True, index_label="index")

    # Plot original, validated, and post-processed reference
    plot_turns(df_ref, dfv_ref, dfp_ref, titles=["Original turn data - reference (demo)", "After validation", "After post-processing"])

    # Compute errors and print summary
    err, err_df = compute_all_errors(dfp_ref, dfp_est, min_overlap_ratio=0.1)
    print(err["turn"]["duration_delta"])
    print_error_summary(err)
