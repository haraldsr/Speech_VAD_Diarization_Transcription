"""
Compute turn errors between ground truth and estimated turns.

Based on MATLAB implementation compute_turn_errors.m.
"""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
import pandas as pd


def compute_overlap_ratio(
    s1: Tuple[float, float],
    s2: Tuple[float, float],
) -> float:
    """
    Compute the overlap ratio between two time segments.

    Args:
        s1: (start_time, end_time) for first segment (ground truth, used as
            reference for ratio).
        s2: (start_time, end_time) for second segment (estimated).

    Returns:
        Overlap ratio as overlapping duration / ground truth duration.
    """
    # Ensure segments are sorted [start, end]
    s1 = tuple(sorted(s1))
    s2 = tuple(sorted(s2))

    # Compute overlap ratio as overlapping duration / true duration
    overlapping_duration = min(s1[1], s2[1]) - max(s1[0], s2[0])
    true_duration = s1[1] - s1[0]

    if true_duration <= 0:
        warnings.warn(
            f"Invalid reference segment duration: {true_duration}. "
            "Overlap ratio set to 0.0."
        )
        return 0.0

    return max(0.0, overlapping_duration) / true_duration


def compute_turn_errors(
    df_ref: pd.DataFrame,
    df_est: pd.DataFrame,
    min_overlap_ratio: float,
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
            - turn_type: Turn type identifier
        df_est: Estimated turns DataFrame with same columns as df_ref.
        min_overlap_ratio: Minimum overlap ratio required for matching turns.

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

            # Compute overlap ratio only if turn_type and speaker match
            if (
                df_ref.iloc[i_ref]["turn_type"] == df_est.iloc[i_est]["turn_type"]
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
        warnings.warn("Multiple matches found for a single ground truth event.")

        for i_est in range(n_est):
            if matches_to_ref[i_est] > 1:
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

        print("----------------------------")

    # Resolve multiple matches to estimated events
    if not np.all(matches_to_est <= 1):
        warnings.warn("Multiple matches found for a single estimated event.")

        for i_ref in range(n_ref):
            if matches_to_est[i_ref] > 1:
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
            duration_delta_list.append(float(np.nansum(duration_delta_matrix[:, i_ref])))
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
                    "turn_type": df_est.iloc[i_est]["turn_type"],
                    "detected": np.nan,
                    "duration_delta": np.nan,
                    "start_delta": np.nan,
                    "end_delta": np.nan}
            )

    if false_positives:
        df_fp = pd.DataFrame(false_positives)
        df_out = pd.concat([df_out, df_fp], ignore_index=True)

    return df_out

def tabulate_floor_transfers(
    df_turns: pd.DataFrame,
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
            - turn_type: Turn type identifier ("T" for turns)

    Returns:
        DataFrame with floor transfer events containing:
            - speaker: Combined speaker identifiers (e.g., "P1-P2")
            - start_sec: Floor transfer start time (end of previous turn)
            - end_sec: Floor transfer end time (start of next turn)
            - duration_sec: Floor transfer duration
            - turn_type: "FTO" (floor transfer offset)
    """
    # Filter for turns only
    turns = df_turns[df_turns["turn_type"] == "T"].copy()
    turns = turns.sort_values(by="start_sec").reset_index(drop=True)

    n_turns = len(turns)
    floor_transfers = []

    for i_turn in range(n_turns - 1):
        speaker_current = turns.iloc[i_turn]["speaker"]
        speaker_next = turns.iloc[i_turn + 1]["speaker"]

        # Floor transfers only occur between different speakers
        # if speaker_current == speaker_next:
        #     raise ValueError(
        #         f"Consecutive turns by the same speaker found at index {i_turn}. "
        #         "Expected alternating speakers for floor transfers."
        #     )

        # Create floor transfer entry
        speakers = f"{speaker_current}-{speaker_next}"
        t_start = turns.iloc[i_turn]["end_sec"]
        t_end = turns.iloc[i_turn + 1]["start_sec"]
        duration = t_end - t_start

        floor_transfers.append(
            {
                "speaker": speakers,
                "start_sec": t_start,
                "end_sec": t_end,
                "duration_sec": duration,
                "turn_type": "FTO",
            }
        )

    df_fto = pd.DataFrame(floor_transfers)

    # Verify floor transfer count
    n_fto = len(df_fto)
    expected_fto = n_turns - 1
    if n_fto != expected_fto:
        raise ValueError(
            f"Expected {expected_fto} floor transfers, but got {n_fto}."
        )

    return df_fto

def compute_all_errors(
    df_ref: pd.DataFrame,
    df_est: pd.DataFrame,
    min_overlap_ratio: float,
) -> Tuple[dict,pd.DataFrame]:
    """
    Compute turn errors for both turns and floor transfers.

    Args:
        df_ref: Ground truth turns DataFrame.
        df_est: Estimated turns DataFrame.
        min_overlap_ratio: Minimum overlap ratio for matching turns.

    Returns:
        Tuple of (turn_errors_df, fto_errors_df) where:
            - turn_errors_df: DataFrame with turn errors.
            - fto_errors_df: DataFrame with floor transfer errors.
    """
    turn_errors_df = compute_turn_errors(df_ref, df_est, min_overlap_ratio)

    df_fto_ref = tabulate_floor_transfers(df_ref)
    df_fto_est = tabulate_floor_transfers(df_est)
    fto_errors_df = compute_turn_errors(df_fto_ref, df_fto_est, min_overlap_ratio)

    err_df = pd.concat([turn_errors_df, fto_errors_df], ignore_index=True)

    # Backchannel error metrics
    BC_FP = err_df[(err_df["turn_type"] == "B") & (err_df["detected"].isna())]
    BC_TP = err_df[(err_df["turn_type"] == "B") & (err_df["detected"] == True)]
    BC_FN = err_df[(err_df["turn_type"] == "B") & (err_df["detected"] == False)]
    
    BC_metrics = {
      "precision": len(BC_TP) / (len(BC_TP) + len(BC_FP)) if (len(BC_TP) + len(BC_FP)) > 0 else 0.0,
      "recall": len(BC_TP) / (len(BC_TP) + len(BC_FN)) if (len(BC_TP) + len(BC_FN)) > 0 else 0.0,
      "mean_duration_delta": BC_TP["duration_delta"].dropna().abs().mean() if not BC_TP.empty else np.nan,
      "mean_start_delta": BC_TP["start_delta"].dropna().abs().mean() if not BC_TP.empty else np.nan,
      "mean_end_delta": BC_TP["end_delta"].dropna().abs().mean() if not BC_TP.empty else np.nan,
    }

    # Turn error metrics
    T_FP = err_df[(err_df["turn_type"] == "T") & (err_df["detected"].isna())]
    T_TP = err_df[(err_df["turn_type"] == "T") & (err_df["detected"] == True)]
    T_FN = err_df[(err_df["turn_type"] == "T") & (err_df["detected"] == False)]

    T_metrics = {
        "precision": len(T_TP) / (len(T_TP) + len(T_FP)) if (len(T_TP) + len(T_FP)) > 0 else 0.0,
        "recall": len(T_TP) / (len(T_TP) + len(T_FN)) if (len(T_TP) + len(T_FN)) > 0 else 0.0,
        "mean_duration_delta": T_TP["duration_delta"].dropna().abs().mean() if not T_TP.empty else np.nan,
        "mean_start_delta": T_TP["start_delta"].dropna().abs().mean() if not T_TP.empty else np.nan,
        "mean_end_delta": T_TP["end_delta"].dropna().abs().mean() if not T_TP.empty else np.nan,
    }

    # Floor transfer error metrics
    FTO_FP = err_df[(err_df["turn_type"] == "FTO") & (err_df["detected"].isna())]
    FTO_TP = err_df[(err_df["turn_type"] == "FTO") & (err_df["detected"] == True)]
    FTO_FN = err_df[(err_df["turn_type"] == "FTO") & (err_df["detected"] == False)]

    FTO_metrics = {
        "precision": len(FTO_TP) / (len(FTO_TP) + len(FTO_FP)) if (len(FTO_TP) + len(FTO_FP)) > 0 else 0.0,
        "recall": len(FTO_TP) / (len(FTO_TP) + len(FTO_FN)) if (len(FTO_TP) + len(FTO_FN)) > 0 else 0.0,
        "mean_duration_delta": FTO_TP["duration_delta"].dropna().abs().mean() if not FTO_TP.empty else np.nan,
        "mean_start_delta": FTO_TP["start_delta"].dropna().abs().mean() if not FTO_TP.empty else np.nan,
        "mean_end_delta": FTO_TP["end_delta"].dropna().abs().mean() if not FTO_TP.empty else np.nan,
    }

    err = {
        "BC": BC_metrics,
        "T": T_metrics,
        "FTO": FTO_metrics,
    }

    return err, err_df



if __name__ == "__main__":
    # Example usage
    df_ref = pd.read_csv("path\\to\\manual_labels.txt", sep="\t",header=None, names=["speaker", "foo", "start_sec", "end_sec", "duration_sec", "turn_type"])
    df_ref = df_ref.replace({"speaker": {"Talker1": "P1", "Talker2": "P2"},"turn_type": {"t": "T", "b": "B"}})
    df_ref = df_ref.drop(columns=["foo"])

    df_est = pd.read_csv("outputs\\dyad\\merged_turns.txt", sep="\t")
    
    err,err_df = compute_all_errors(df_ref, df_est, min_overlap_ratio=0.5)

    print("-"*60)    

    print(f"Backchannel Precision: {err['BC']['precision']:.2f}")
    print(f"Backchannel Recall: {err['BC']['recall']:.2f}")
    print(f"Backchannel Mean Duration Delta (abs): {err['BC']['mean_duration_delta']:.2f} seconds")
    print(f"Backchannel Mean Start Delta (abs): {err['BC']['mean_start_delta']:.2f} seconds")
    print(f"Backchannel Mean End Delta (abs): {err['BC']['mean_end_delta']:.2f} seconds")

    print("-"*60)

    print(f"Turn Precision: {err['T']['precision']:.2f}")
    print(f"Turn Recall: {err['T']['recall']:.2f}")
    print(f"Turn Mean Duration Delta (abs): {err['T']['mean_duration_delta']:.2f} seconds")
    print(f"Turn Mean Start Delta (abs): {err['T']['mean_start_delta']:.2f} seconds")
    print(f"Turn Mean End Delta (abs): {err['T']['mean_end_delta']:.2f} seconds")

    print("-"*60)

    print(f"FTO Precision: {err['FTO']['precision']:.2f}")
    print(f"FTO Recall: {err['FTO']['recall']:.2f}")
    print(f"FTO Mean Duration Delta (abs): {err['FTO']['mean_duration_delta']:.2f} seconds")
    print(f"FTO Mean Start Delta (abs): {err['FTO']['mean_start_delta']:.2f} seconds")
    print(f"FTO Mean End Delta (abs): {err['FTO']['mean_end_delta']:.2f} seconds")  
