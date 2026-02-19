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
            f"{type} Mean Duration Delta (abs): "
            f"{metrics['mean_duration_delta']:.2f} ± {metrics['std_duration_delta']:.2f} seconds"
        )
        print(
            f"{type} Mean Start Delta (abs): "
            f"{metrics['mean_start_delta']:.2f} ± {metrics['std_start_delta']:.2f} seconds"
        )
        print(
            f"{type} Mean End Delta (abs): "
            f"{metrics['mean_end_delta']:.2f} ± {metrics['std_end_delta']:.2f} seconds"
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
    # turn_errors_df = compute_turn_errors(df_ref, df_est, min_overlap_ratio)

    # df_fto_ref = tabulate_floor_transfers(df_ref)
    # df_fto_est = tabulate_floor_transfers(df_est)
    # print(df_ref)
    # print(df_fto_ref)
    # fto_errors_df = compute_turn_errors(df_fto_ref, df_fto_est, min_overlap_ratio)

    # turn_errors_df = compute_turn_errors(df_ref, df_est, min_overlap_ratio)

    df_fto_ref = tabulate_floor_transfers(df_ref, suppress_warnings)
    df_fto_est = tabulate_floor_transfers(df_est, suppress_warnings)

    df_ref_cat = pd.concat([df_ref, df_fto_ref], ignore_index=True)
    df_est_cat = pd.concat([df_est, df_fto_est], ignore_index=True)
                           
    err_df = compute_turn_errors(df_ref_cat, df_est_cat, min_overlap_ratio, suppress_warnings)

    # err_df = pd.concat([turn_errors_df, fto_errors_df], ignore_index=True)

    # Compute summary metrics for each turn type
    err: dict = {}
    for type in err_df["type"].unique():
        type_df = err_df[err_df["type"] == type]

        # True positives: detected=True
        tp = (type_df["detected"] == 1).sum()
        # False negatives: detected=False
        fn = (type_df["detected"] == 0).sum()
        # False positives: detected=NaN
        fp = type_df["detected"].isna().sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        detected_df = type_df[type_df["detected"] == True]  # noqa: E712
        mean_duration_delta = (
            detected_df["duration_delta"].abs().mean() if len(detected_df) > 0 else 0.0
        )
        mean_start_delta = (
            detected_df["start_delta"].abs().mean() if len(detected_df) > 0 else 0.0
        )
        mean_end_delta = (
            detected_df["end_delta"].abs().mean() if len(detected_df) > 0 else 0.0
        )
        
        std_duration_delta = (
            detected_df["duration_delta"].abs().std() if len(detected_df) > 0 else 0.0
        )
        std_start_delta = (
            detected_df["start_delta"].abs().std() if len(detected_df) > 0 else 0.0
        )
        std_end_delta = (
            detected_df["end_delta"].abs().std() if len(detected_df) > 0 else 0.0
        )

        err[type] = {
            "precision": precision,
            "recall": recall,
            "mean_duration_delta": mean_duration_delta,
            "std_duration_delta": std_duration_delta,
            "mean_start_delta": mean_start_delta,
            "std_start_delta": std_start_delta,
            "mean_end_delta": mean_end_delta,
            "std_end_delta": std_end_delta,
        }

    return err, err_df

def postprocess_turn_df(
    df: pd.DataFrame,
) -> pd.DataFrame:
    
    # Replace common value variations
    df =df.replace(
        {
            "speaker": {"Talker1": "P1","p1":"P1", "Talker2": "P2","p2":"P2"},
            "type": {"T": "turn","t": "turn", "B": "backchannel", "b": "backchannel"},
        }
    )

    # Treat "overlap" and "overlapped_turn" as "backchannel" for error analysis
    df = df.replace(
        {
            "type": {"overlap": "backchannel", "overlapped_turn": "backchannel"},
        }
    )

    # Treat backchannels as turns if they're not embedded in an interlocutor turn
    speakers = ["P1", "P2"]
    for ip in (0, 1):
        p_bc = speakers[ip]
        p_int = speakers[1 - ip]  # Get the other speaker

        mask_backchannel = (df["type"] == "backchannel") & (df["speaker"] == p_bc)
        mask_interlocutor_turn = (df["type"] == "turn") & (df["speaker"] == p_int)

        for idx_bc in df[mask_backchannel].index:
            t_start_bc = df.loc[idx_bc, "start_sec"]
            t_end_bc = df.loc[idx_bc, "end_sec"]

            # Check if backchannel is fully contained within any interlocutor turn
            if not np.any(
                (mask_interlocutor_turn)
                & (df["start_sec"] <= t_start_bc)
                & (df["end_sec"] >= t_end_bc)
            ):
                df.loc[idx_bc, "type"] = "turn"

    # Combine consecutive turns of the same speaker if no interlocutor turn between them
    for speaker in speakers:
        print(f"Processing speaker {speaker} for turn combination...")
        mask_speaker_turn = (df["type"] == "turn") & (df["speaker"] == speaker)
        print(f"Found {mask_speaker_turn.sum()} turns for speaker {speaker} before combination.")
        df_speaker_turns = df[mask_speaker_turn].sort_values(by="start_sec")

        for i in range(len(df_speaker_turns) - 1):
            current_turn = df_speaker_turns.iloc[i]
            next_turn = df_speaker_turns.iloc[i + 1]

            # Check if there's an interlocutor turn starts and/or ends between current and next turn
            if np.any(
                (df["type"] == "turn")
                & (df["speaker"] != speaker)
                & (df["start_sec"] >= current_turn["end_sec"])
                & (df["start_sec"] <= next_turn["start_sec"])
            ) | np.any(
                (df["type"] == "turn")
                & (df["speaker"] != speaker)
                & (df["end_sec"] >= current_turn["end_sec"])
                & (df["end_sec"] <= next_turn["start_sec"])
            ):
                continue 
            else:

                # Combine turns by updating the end time of the current turn
                df.loc[df_speaker_turns.index[i], "end_sec"] = next_turn["end_sec"]
                df.loc[df_speaker_turns.index[i], "duration_sec"] = (
                    df.loc[df_speaker_turns.index[i], "end_sec"]
                    - df.loc[df_speaker_turns.index[i], "start_sec"]
                )
                # Mark the next turn for deletion
                df.loc[df_speaker_turns.index[i + 1], "type"] = "delete"
    df = df[df["type"] != "delete"].sort_values(by="start_sec").reset_index(drop=True)

    df_turns = df[df["type"] == "turn"].sort_values(by="start_sec").reset_index(drop=True)
    for i in range(len(df_turns) - 1):
        current_turn = df_turns.iloc[i]
        next_turn = df_turns.iloc[i + 1]
        assert current_turn["speaker"] != next_turn["speaker"], (
            f"Consecutive turns by same speaker found at index {i} and {i+1}."
        )
    return df



if __name__ == "__main__":
    # Example usage
    df_ref = pd.read_csv(
        "demo/annotations/F1F2_quiet_food_1m_01_labels_manual_rinor.txt",
        sep="\t",
        header=None,
        names=["speaker", "foo", "start_sec", "end_sec", "duration_sec", "type"],
    ).drop(columns=["foo"])

    df_est = pd.read_csv("outputs/cpu/final_labels.txt", sep="\t")

    df_ref_proc = postprocess_turn_df(df_ref)
    df_est_proc = postprocess_turn_df(df_est)

    err, err_df = compute_all_errors(df_ref_proc, df_est_proc, min_overlap_ratio=0.1)

    print_error_summary(err)