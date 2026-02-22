"""
Compute turn errors between ground truth and estimated turns.

Based on MATLAB implementation compute_turn_errors.m.
"""

from __future__ import annotations

import warnings
from typing import Tuple, cast

import numpy as np
import pandas as pd


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
    min_overlap_ratio: float,
    suppress_warnings: bool = False,
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
        suppress_warnings: If True, suppresses warning messages. Defaults to False.

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

    for i_ref in range(n_ref):
        for i_est in range(n_est):
            # Sort speakers to treat "1-2" and "2-1" FTOs together
            speaker_ref = "".join(sorted(str(df_ref.iloc[i_ref]["speaker"])))
            speaker_est = "".join(sorted(str(df_est.iloc[i_est]["speaker"])))

            # Compute overlap ratio only if type and speaker match
            if (
                df_ref.iloc[i_ref]["type"] == df_est.iloc[i_est]["type"]
                and speaker_ref == speaker_est
            ):
                overlap_matrix[i_est, i_ref] = compute_overlap_ratio(
                    (df_ref.iloc[i_ref]["start_sec"], df_ref.iloc[i_ref]["end_sec"]),
                    (df_est.iloc[i_est]["start_sec"], df_est.iloc[i_est]["end_sec"]),
                )

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

    # Build match matrix
    match_matrix = overlap_matrix >= min_overlap_ratio

    # Count matches per reference and estimated turn
    matches_to_ref = np.nansum(match_matrix, axis=1)
    matches_to_est = np.nansum(match_matrix, axis=0)

    # Resolve multiple matches to ground truth events
    if not np.all(matches_to_ref <= 1):
        if not suppress_warnings:
            warnings.warn("Multiple matches found for a single ground truth event.")

        for i_est in range(n_est):
            if matches_to_ref[i_est] > 1:
                if not suppress_warnings:
                    print(
                        f"Ground truth event {i_est} has "
                        f"{int(matches_to_ref[i_est])} matches."
                    )
                    print("Keeping only the closest match in terms of overlapping ratio.")

                # Find best match (highest overlap ratio among matches)
                masked_overlap = np.where(
                    match_matrix[i_est, :], overlap_matrix[i_est, :], -np.inf
                )
                closest_match = np.argmax(masked_overlap)
                match_matrix[i_est, :] = False
                match_matrix[i_est, closest_match] = True

        if not suppress_warnings:
            print("----------------------------")

    # Resolve multiple matches to estimated events
    if not np.all(matches_to_est <= 1):
        if not suppress_warnings:
            warnings.warn("Multiple matches found for a single estimated event.")

        for i_ref in range(n_ref):
            if matches_to_est[i_ref] > 1:
                if not suppress_warnings:
                    print(
                        f"Estimated event {i_ref} has "
                        f"{int(matches_to_est[i_ref])} matches."
                    )
                    print("Keeping only the closest match in terms of overlapping ratio.")

                # Find best match (highest overlap ratio among matches)
                masked_overlap = np.where(
                    match_matrix[:, i_ref], overlap_matrix[:, i_ref], -np.inf
                )
                closest_match = np.argmax(masked_overlap)
                match_matrix[:, i_ref] = False
                match_matrix[closest_match, i_ref] = True

        if not suppress_warnings:
            print("----------------------------")

    # Build error matrix with duration differences for matched turns
    duration_delta_matrix = np.full_like(duration_delta, np.nan)
    duration_delta_matrix[match_matrix] = duration_delta[match_matrix]

    start_delta_matrix = np.full_like(start_delta, np.nan)
    start_delta_matrix[match_matrix] = start_delta[match_matrix]

    end_delta_matrix = np.full_like(end_delta, np.nan)
    end_delta_matrix[match_matrix] = end_delta[match_matrix]

    # Handle true positives and false negatives
    detected_list = []
    duration_delta_list = []
    start_delta_list = []
    end_delta_list = []

    for i_ref in range(n_ref):
        is_detected = np.any(match_matrix[:, i_ref])
        detected_list.append(is_detected)

        if is_detected:
            duration_delta_list.append(
                float(np.nansum(duration_delta_matrix[:, i_ref]))
            )
            start_delta_list.append(float(np.nansum(start_delta_matrix[:, i_ref])))
            end_delta_list.append(float(np.nansum(end_delta_matrix[:, i_ref])))
        else:
            duration_delta_list.append(np.nan)
            start_delta_list.append(np.nan)
            end_delta_list.append(np.nan)

    df_out["detected"] = detected_list
    df_out["duration_delta"] = duration_delta_list
    df_out["start_delta"] = start_delta_list
    df_out["end_delta"] = end_delta_list

    # Handle false positives (unmatched estimated turns)
    false_positives = []
    for i_est in range(n_est):
        if not np.any(match_matrix[i_est, :]):
            false_positives.append(
                {
                    "speaker": df_est.iloc[i_est]["speaker"],
                    "start_sec": df_est.iloc[i_est]["start_sec"],
                    "end_sec": df_est.iloc[i_est]["end_sec"],
                    "duration_sec": df_est.iloc[i_est]["duration_sec"],
                    "type": df_est.iloc[i_est]["type"],
                    "detected": np.nan,
                    "duration_delta": np.nan,
                    "start_delta": np.nan,
                    "end_delta": np.nan,
                }
            )

    if false_positives:
        df_fp = pd.DataFrame(false_positives)
        df_out = pd.concat([df_out, df_fp], ignore_index=True)

    return df_out

def tabulate_floor_transfers(
    df_turns: pd.DataFrame,
    suppress_warnings: bool = False,
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
            if not suppress_warnings:
                warnings.warn(
                    f"Consecutive turns by the same speaker found at index {i_turn}. "
                    "Expected alternating speakers for floor transfers."
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

def print_error_summary(err: dict) -> None:
    """
    Print a summary of turn error metrics.

    Args:
        err: Dictionary containing error metrics for each turn type.
    """
    for type, metrics in err.items():
        print("-" * 60)
        print(f"{type} Precision: {metrics['precision']:.2f}")
        print(f"{type} Recall: {metrics['recall']:.2f}")
        print(
            f"{type} Mean Absolute Duration Delta: "
            f"{metrics['duration_delta'][1]:.2f} seconds [5%: {metrics['duration_delta'][0]:.2f}s, 95%: {metrics['duration_delta'][2]:.2f}s]"
        )
        print(
            f"{type} Mean Absolute Start Delta: "
            f"{metrics['start_delta'][1]:.2f} seconds [5%: {metrics['start_delta'][0]:.2f}s, 95%: {metrics['start_delta'][2]:.2f}s]"
        )
        print(
            f"{type} Mean Absolute End Delta: "
            f"{metrics['end_delta'][1]:.2f} seconds [5%: {metrics['end_delta'][0]:.2f}s, 95%: {metrics['end_delta'][2]:.2f}s]"
        )
        print(
            f"{type} Mean Absolute Start/End Delta: "
            f"{metrics['start_end'][1]:.2f} seconds [5%: {metrics['start_end'][0]:.2f}s, 95%: {metrics['start_end'][2]:.2f}s]"
        )

def compute_all_errors(
    df_ref: pd.DataFrame,
    df_est: pd.DataFrame,
    min_overlap_ratio: float,
    suppress_warnings: bool = False,
) -> Tuple[dict, pd.DataFrame]:
    """
    Compute turn errors for both turns and floor transfers.

    Args:
        df_ref: Ground truth turns DataFrame.
        df_est: Estimated turns DataFrame.
        min_overlap_ratio: Minimum overlap ratio for matching turns.
        suppress_warnings: If True, suppresses warning messages. Defaults to False.

    Returns:
        Tuple of (metrics_dict, errors_dataframe) where metrics_dict contains
        precision, recall, and mean timing deltas for each turn type.
    """

    df_fto_ref = tabulate_floor_transfers(df_ref, suppress_warnings)
    df_fto_est = tabulate_floor_transfers(df_est, suppress_warnings)

    df_ref_cat = pd.concat([df_ref, df_fto_ref], ignore_index=True)
    df_est_cat = pd.concat([df_est, df_fto_est], ignore_index=True)
                           
    err_df = compute_turn_errors(df_ref_cat, df_est_cat, min_overlap_ratio, suppress_warnings)

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

    df =df.replace(
        {
            # Common speeling variations for speakers and types
            "speaker": {"Talker1": "P1","p1":"P1", "Talker2": "P2","p2":"P2"},
            "type": {"T": "turn","t": "turn", "B": "backchannel", "b": "backchannel"},
        }
    )

    # Treat overlap labels turns as backchannels
    df = df.replace({"type": {"overlap": "backchannel", "overlapped_turn": "backchannel"}})

    return df

def validate_turns(df: pd.DataFrame) -> None:
    
    print("="*70)
    print("Validating column names, labels, durations, and self-intersecting events.")
    print("="*70)
    df = check_colnames(df)
    df = check_labels(df)
    df = check_durations(df)
    df = check_self_intersection(df)

    return df

def check_colnames(df: pd.DataFrame) -> pd.DataFrame:
    print("-"*60 + "\nChecking required columns in the DataFrame...")
    required_columns = {"speaker", "start_sec", "end_sec", "duration_sec", "type"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    else:
        print("All required columns are present.")
    return df

def check_labels(df: pd.DataFrame) -> pd.DataFrame:
    print("-"*60 + "\nChecking for valid speaker and type labels...")
    allowed_speaker_labels = {"P1", "P2"}
    allowed_type_labels = {"turn", "backchannel"}
    if not set(df["speaker"].unique()).issubset(allowed_speaker_labels):
        print(f"Unexpected speaker labels found:")
        for label in set(df["speaker"].unique()) - allowed_speaker_labels:
            print(f" > {label}")
        print("Attempting replacement with common variations.")
    if not set(df["type"].unique()).issubset(allowed_type_labels):
        print(f"Unexpected type labels found:")
        for label in set(df["type"].unique()) - allowed_type_labels:
            print(f" > {label}")
        print("Attempting replacement with common variations.")
    else:
        print("All labels are valid. No replacement performed.")
    
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
        print(f"Speaker and type labels successfully replaced.")
        
    return df

def check_durations(df: pd.DataFrame) -> pd.DataFrame:
    print("-"*60 + "\nChecking duration consistency and positivity...")
    df["duration_check"] = df["end_sec"] - df["start_sec"] - df["duration_sec"]
    msk_mis = np.abs(df["duration_check"]) >= 1e-3
    if np.any(msk_mis):
        print("Duration inconsistencies found (end_sec-start_sec does not match duration_sec):")
        print(df[msk_mis])
        print("Replacing duration_sec with end_sec-start_sec for inconsistent entries.")
        df.loc[msk_mis, "duration_sec"] = df.loc[msk_mis, "end_sec"] - df.loc[msk_mis, "start_sec"]
    else:
        print("All durations are consistent.")

    # Check if all durations are positive
    if np.any(df["duration_sec"] <= 0):
        print("\nNon-positive durations found:")
        print(df[df["duration_sec"] <= 0])
        print("Removing entries with non-positive durations.")
        df = df[df["duration_sec"] > 0]
    else:
        print("All durations are positive.") 
    
    return df.drop(columns=["duration_check"])

def check_self_intersection(df: pd.DataFrame) -> pd.DataFrame:
    print("-"*60 + "\nChecking for self-intersecting events...")
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
        print("\nSelf-intersecting entries found (same speaker with overlapping time intervals):")
        print(df[df["self_intersect"] != ""])
        print("Keeping only first entry and removing self-intersecting entries for each speaker.")
        # Remove self-intersecting entries
        df = df[df["self_intersect"] == ""].drop(columns=["self_intersect"])
    else:
        print("No self-intersecting entries found.")
    
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
        mask = pd.Series([True] * len(df))
    
    OR = pd.Series([0.0] * len(df), index=df.index)
    for i, row in df.iterrows():
        OR[i] = compute_overlap_ratio((t1, t2), (row["start_sec"], row["end_sec"]))

    idx_intersecting = (OR > 0.0) & mask
  
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
) -> pd.DataFrame:
    speakers = ["P1", "P2"] # TO DO: cover any speaker amount

    # Initialize at first iteration    
    if rec_iter == 0:
        df = replace_labels(df)
        df = df.sort_values(by="start_sec")
        print("-"*60)
    
    print(f">>> Starting postprocessing - recursion iteration {rec_iter}")
    turns_merged = 0

    # 1. Merge consecutive turns by the same speaker if they are not separated by any interlocutor turn or self-backchannel
    for speaker in speakers:
        msk_spk = df["speaker"] == speaker
        msk_int = df["speaker"] == speakers[1 - speakers.index(speaker)]
        msk_turn = df["type"] == "turn"
        msk_spk_bc = msk_spk & (df["type"] == "backchannel")
        msk_int_turn = msk_int & msk_turn
        
        idx_spk_turn = df[msk_spk & msk_turn].index
        for i in range(len(idx_spk_turn) - 1):
            if not any(find_intersecting_events(df, 
                                                t1 = df.loc[idx_spk_turn[i], "end_sec"],
                                                t2 = df.loc[idx_spk_turn[i+1],"start_sec"],
                                                mask = msk_int_turn | msk_spk_bc)):
                
                df = merge_events(df, idx_spk_turn[i], idx_spk_turn[i + 1])

                # Count merged turns (for printing at the end of the iteration)
                turns_merged = turns_merged + 1

    # 2. Treat speaker backchannels as turns if they're not embedded in any interlocutor turn
    for speaker in speakers:
        
        msk_spk = df["speaker"] == speaker
        msk_int = df["speaker"] == speakers[1 - speakers.index(speaker)]
        msk_bc = df["type"] == "backchannel"
        msk_turn = df["type"] == "turn"

        for idx_bc in df[msk_spk & msk_bc].index:
            if not any(
                find_embedding_events(df,
                                      t_start = df.loc[idx_bc, "start_sec"],
                                      t_end = df.loc[idx_bc, "end_sec"],
                                      mask =  msk_int & msk_turn)):
                df.loc[idx_bc, "type"] = "turn"
    
    # 3. Treat speaker turns as backchannels if they're embedded in any interlocutor turn or backchannel
    for speaker in speakers:
        msk_spk = df["speaker"] == speaker
        msk_int = df["speaker"] == speakers[1 - speakers.index(speaker)]
        msk_turn = df["type"] == "turn"
        
        for idx_turn in df[msk_spk & msk_turn].index:
            if any(
                find_embedding_events(df, 
                                      t_start = df.loc[idx_turn, "start_sec"], 
                                      t_end = df.loc[idx_turn, "end_sec"], 
                                      mask = msk_int & msk_turn)):
                df.loc[idx_turn, "type"] = "backchannel"
    
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
            print(f"Consecutive turns by the same speaker found at index {idx_turns[i]} and {idx_turns[i + 1]} after merging. Recursing for another iteration.")
            break
        
    print(f"Post-processing iteration completed with {turns_merged} turns merged after {rec_iter} iterations.\n")

    # Recur unless 1) no consecutive turns found or 2) max iterations reached
    if rec_iter < max_iter:
      if recur:
        return postprocess_turn_df(df, max_iter=max_iter, rec_iter=rec_iter+1)
      else:
          print("No consecutive turns found. Post-processing complete.")
          print("-"*60)
    else:        
        print(f"Maximum recursion iterations ({max_iter}) reached. Post-processing terminated.")
        print("-"*60)

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
