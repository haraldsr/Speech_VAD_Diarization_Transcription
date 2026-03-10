"""
Stage-wise evaluation metrics for the speech processing pipeline.

Evaluates each pipeline stage independently:
  1. VAD / Speech Activity Detection  (detection error rate, precision, recall, F1)
  2. Diarization / Speaker Attribution (DER with components, JER)
  3. Transcription quality             (WER, CER — raw and normalised)
  4. Label-type classification          (turn / backchannel / overlapped_turn)

Uses ``pyannote.metrics`` where available; falls back to lightweight
implementations when it is not installed.

Does **not** implement turn-taking dynamics metrics — those remain in
``compute_turn_errors.py``.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

try:
    _HAS_INTERVALTREE = True
except ImportError:  # pragma: no cover
    _HAS_INTERVALTREE = False

# ---------------------------------------------------------------------------
# Optional heavy imports — degrade gracefully
# ---------------------------------------------------------------------------
try:
    from pyannote.core import Annotation, Segment, Timeline
    from pyannote.metrics.detection import (
        DetectionAccuracy,
        DetectionErrorRate,
        DetectionPrecision,
        DetectionRecall,
    )
    from pyannote.metrics.diarization import (
        DiarizationCompleteness,
        DiarizationCoverage,
        DiarizationErrorRate,
        DiarizationHomogeneity,
        DiarizationPurity,
        GreedyDiarizationErrorRate,
        JaccardErrorRate,
    )
    from pyannote.metrics.identification import (
        IdentificationErrorRate,
    )
    from pyannote.metrics.segmentation import (
        SegmentationCoverage,
        SegmentationPrecision,
        SegmentationPurity,
        SegmentationPurityCoverageFMeasure,
        SegmentationRecall,
    )

    _HAS_PYANNOTE_METRICS = True
except ImportError:  # pragma: no cover
    _HAS_PYANNOTE_METRICS = False

try:
    import jiwer

    _HAS_JIWER = True
except ImportError:  # pragma: no cover
    _HAS_JIWER = False

try:
    import sacrebleu as _sacrebleu

    _HAS_SACREBLEU = True
except ImportError:  # pragma: no cover
    _HAS_SACREBLEU = False

try:
    import torch as _torch
    from transformers import AutoModel as _AutoModel
    from transformers import AutoTokenizer as _AutoTokenizer

    _HAS_TRANSFORMERS = True
except ImportError:  # pragma: no cover
    _HAS_TRANSFORMERS = False

# Cache for loaded embedding model (loaded once on first use)
_EMBED_MODEL_CACHE: Dict[str, Any] = {}


# ===================================================================
# 1.  Reference / hypothesis readers  (multi-format)
# ===================================================================


def _detect_format(path: str) -> str:
    """Detect ground-truth file format from content heuristics.

    Returns one of ``"ass"``, ``"elan"``, ``"exp5"``, ``"exp6"``,
    ``"rttm"``, ``"pipeline_output"``.
    """
    if Path(path).suffix.lower() == ".ass":
        return "ass"

    with open(path, encoding="utf-8") as fh:
        first_line = fh.readline().strip()
        peek_lines = [first_line]
        for _ in range(20):
            line = fh.readline()
            if not line:
                break
            peek_lines.append(line.strip())

    if any(
        line.startswith("[Script Info]") or line.startswith("Dialogue:")
        for line in peek_lines
    ):
        return "ass"

    # RTTM lines start with "SPEAKER"
    if first_line.startswith("SPEAKER"):
        return "rttm"

    cols = first_line.split("\t")

    # ELAN header: tier  begin  end  annotation
    if cols[0].lower() == "tier":
        return "elan"

    # Pipeline output header
    if cols[0].lower() == "speaker" and "start_sec" in first_line.lower():
        return "pipeline_output"

    # EXP 5-col vs 6-col (6-col has an empty second column):
    # 5-col: speaker start end dur type
    # 6-col: speaker <empty> start end dur type
    if len(cols) >= 6 and cols[1] == "":
        return "exp6"

    return "exp5"


def load_reference(
    path: str,
    *,
    fmt: Optional[str] = None,
) -> pd.DataFrame:
    """Load a reference (ground-truth) annotation file.

    Returns a normalised DataFrame with columns:
        ``speaker  start_sec  end_sec  duration_sec  type  [transcription]``
    """
    if fmt is None:
        fmt = _detect_format(path)

    loaders = {
        "ass": _load_ass,
        "elan": _load_elan,
        "exp5": _load_exp5,
        "exp6": _load_exp6,
        "rttm": _load_rttm,
        "pipeline_output": _load_pipeline_output,
    }
    if fmt not in loaders:
        raise ValueError(f"Unknown reference format: {fmt!r}")
    return loaders[fmt](path)


# ---- format-specific loaders ----


def _ass_time_to_sec(value: str) -> float:
    """Convert ASS timestamp (H:MM:SS.CS) to seconds."""
    text = str(value).strip()
    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid ASS timestamp: {value!r}")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return (hours * 3600.0) + (minutes * 60.0) + seconds


def _clean_ass_text(text: str) -> str:
    """Remove ASS inline style tags from dialogue text."""
    return re.sub(r"\{[^}]*\}", "", str(text)).strip()


def _load_ass(path: str) -> pd.DataFrame:
    """Load ASS subtitle dialogue lines.

    Uses the ASS `Name` field as speaker label when available.
    """
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line.startswith("Dialogue:"):
                continue

            payload = line[len("Dialogue:") :].lstrip()
            parts = payload.split(",", 9)
            if len(parts) != 10:
                continue

            # ASS event schema:
            # Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
            start_txt = parts[1].strip()
            end_txt = parts[2].strip()
            speaker = parts[4].strip() or "unknown"
            transcription = _clean_ass_text(parts[9])

            try:
                start_sec = _ass_time_to_sec(start_txt)
                end_sec = _ass_time_to_sec(end_txt)
            except Exception:
                continue

            if end_sec <= start_sec:
                continue

            rows.append(
                {
                    "speaker": speaker,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "duration_sec": end_sec - start_sec,
                    "type": "turn",
                    "transcription": transcription,
                }
            )

    return pd.DataFrame(rows)


def _load_elan(path: str) -> pd.DataFrame:
    """Load ELAN tab-delimited format (tier / begin_ms / end_ms / annotation)."""
    df = pd.read_csv(path, sep="\t")
    # Header: tier  begin  end  annotation
    rows: list[dict] = []
    for _, r in df.iterrows():
        tier_str = str(r["tier"]).strip()
        # Skip rows with empty/missing tier
        if not tier_str or tier_str.lower() == "nan":
            continue
        # ELAN tiers can be "A", "B", "P1", or combined "P1_turn"
        parts = tier_str.rsplit("_", 1)
        if len(parts) == 2 and parts[1] in {"turn", "backchannel", "overlapped_turn"}:
            speaker, utt_type = parts
        else:
            speaker = parts[0]
            utt_type = "turn"
        begin_sec = float(r["begin"]) / 1000.0
        end_sec = float(r["end"]) / 1000.0
        if end_sec <= begin_sec:
            continue
        annotation = str(r.get("annotation", ""))
        rows.append(
            {
                "speaker": speaker,
                "start_sec": begin_sec,
                "end_sec": end_sec,
                "duration_sec": end_sec - begin_sec,
                "type": utt_type,
                "transcription": annotation,
            }
        )
    return pd.DataFrame(rows)


def _load_exp5(path: str) -> pd.DataFrame:
    """Load 5-column EXP-style (speaker start end dur type)."""
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["speaker", "start_sec", "end_sec", "duration_sec", "type"],
    )
    df["transcription"] = ""
    return df


def _load_exp6(path: str) -> pd.DataFrame:
    """Load 6-column EXP-style (speaker <blank> start end dur type)."""
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["speaker", "_blank", "start_sec", "end_sec", "duration_sec", "type"],
    )
    df.drop(columns=["_blank"], inplace=True)
    df["transcription"] = ""
    return df


def _load_rttm(path: str) -> pd.DataFrame:
    """Load NIST RTTM format."""
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if not parts or parts[0] != "SPEAKER":
                continue
            # SPEAKER file 1 start dur <NA> <NA> spk <NA> <NA>
            start = float(parts[3])
            dur = float(parts[4])
            speaker = parts[7]
            rows.append(
                {
                    "speaker": speaker,
                    "start_sec": start,
                    "end_sec": start + dur,
                    "duration_sec": dur,
                    "type": "turn",
                    "transcription": "",
                }
            )
    return pd.DataFrame(rows)


def _load_pipeline_output(path: str) -> pd.DataFrame:
    """Load the pipeline's own output (final_labels.txt with header)."""
    df = pd.read_csv(path, sep="\t")
    if "duration_sec" not in df.columns:
        df["duration_sec"] = df["end_sec"] - df["start_sec"]
    if "transcription" not in df.columns:
        df["transcription"] = ""
    return df


# ===================================================================
# 2.  Helpers — build pyannote objects from DataFrames
# ===================================================================


def _df_to_annotation(
    df: pd.DataFrame,
    label_col: str = "speaker",
) -> "Annotation":
    """Convert a DataFrame with start_sec / end_sec into a ``pyannote.core.Annotation``."""
    if not _HAS_PYANNOTE_METRICS:
        raise ImportError(
            "pyannote.metrics is required for this evaluation. "
            "Install with: pip install pyannote.metrics"
        )
    ann = Annotation()
    for _, row in df.iterrows():
        seg = Segment(float(row["start_sec"]), float(row["end_sec"]))
        ann[seg, ann.new_track(seg)] = str(row[label_col])
    return ann


def _df_to_timeline(df: pd.DataFrame) -> "Timeline":
    """Convert a DataFrame into a ``pyannote.core.Timeline`` (speaker-agnostic)."""
    if not _HAS_PYANNOTE_METRICS:
        raise ImportError("pyannote.metrics is required")
    return Timeline(
        [Segment(float(r["start_sec"]), float(r["end_sec"])) for _, r in df.iterrows()]
    )


def _ensure_sorted_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame sorted by temporal boundaries when columns are present."""
    if df.empty:
        return df
    if "start_sec" in df.columns and "end_sec" in df.columns:
        return df.sort_values(["start_sec", "end_sec"]).reset_index(drop=True)
    return df.reset_index(drop=True)


def _build_uem_from_df(ref: pd.DataFrame, hyp: pd.DataFrame) -> Optional["Timeline"]:
    """Build an explicit UEM timeline to avoid pyannote implicit approximation warnings.

    Uses the same default region pyannote would approximate: the union of
    reference and hypothesis extents.
    """
    if not _HAS_PYANNOTE_METRICS:
        return None

    # We collect global min start and max end across BOTH ref+hyp so the
    # scorer evaluates exactly the shared recording extent.
    starts: list[float] = []
    ends: list[float] = []
    for df in (ref, hyp):
        if df.empty:
            continue
        starts.append(float(df["start_sec"].min()))
        ends.append(float(df["end_sec"].max()))

    if not starts:
        return None

    # UEM shape is one continuous segment: [global_start, global_end].
    return Timeline([Segment(min(starts), max(ends))])


def _overlap_edges(
    ref: pd.DataFrame, hyp: pd.DataFrame
) -> List[Tuple[int, int, float]]:
    """Return overlap edges (ref_idx, hyp_idx, overlap_seconds)."""
    if ref.empty or hyp.empty:
        return []

    # 1-D arrays of boundaries used for fast numeric overlap operations.
    # Shapes:
    #   ref_starts/ref_ends -> (n_ref,)
    #   hyp_starts/hyp_ends -> (n_hyp,)
    ref_starts = ref["start_sec"].values.astype(float)
    ref_ends = ref["end_sec"].values.astype(float)
    hyp_starts = hyp["start_sec"].values.astype(float)
    hyp_ends = hyp["end_sec"].values.astype(float)

    edges: List[Tuple[int, int, float]] = []

    # Sweep-line overlap construction:
    #   - iterate ref segments in start-time order
    #   - maintain active hyp segments that might overlap current ref segment
    # This avoids all-pairs scans in common sparse-overlap settings.
    valid_ref = np.where(ref_ends > ref_starts)[0]
    valid_hyp = np.where(hyp_ends > hyp_starts)[0]
    if valid_ref.size == 0 or valid_hyp.size == 0:
        return edges

    ref_order = valid_ref[np.argsort(ref_starts[valid_ref], kind="mergesort")]
    hyp_order = valid_hyp[np.argsort(hyp_starts[valid_hyp], kind="mergesort")]

    hyp_cursor = 0
    active_hyp: List[int] = []

    for i in ref_order:
        rs = ref_starts[i]
        re_ = ref_ends[i]

        # Add hyp intervals whose start is before current ref end.
        while hyp_cursor < hyp_order.size and hyp_starts[hyp_order[hyp_cursor]] < re_:
            active_hyp.append(int(hyp_order[hyp_cursor]))
            hyp_cursor += 1

        # Drop hyp intervals that end at/before current ref start.
        if active_hyp:
            active_hyp = [j for j in active_hyp if hyp_ends[j] > rs]

        # Compute exact positive overlaps for current ref against active hyps.
        for j in active_hyp:
            overlap = min(re_, hyp_ends[j]) - max(rs, hyp_starts[j])
            if overlap > 0:
                edges.append((int(i), int(j), float(overlap)))

    return edges


def _many_to_many_groups_from_edges(
    n_ref: int,
    n_hyp: int,
    edges: List[Tuple[int, int, float]],
) -> List[Tuple[List[int], List[int]]]:
    """Build connected many-to-many groups from overlap edges."""
    if not edges:
        return []

    # Bipartite graph encoding:
    #   ref nodes are [0 .. n_ref-1]
    #   hyp nodes are [n_ref .. n_ref+n_hyp-1]
    # We connect ref_i <-> hyp_j when they overlap.
    adj: Dict[int, set[int]] = defaultdict(set)
    for i, j, _ in edges:
        hnode = n_ref + j
        adj[i].add(hnode)
        adj[hnode].add(i)

    # BFS connected-components over the bipartite graph.
    visited: set[int] = set()
    groups: List[Tuple[List[int], List[int]]] = []
    for start_node in list(adj.keys()):
        if start_node in visited:
            continue
        queue: deque[int] = deque([start_node])
        visited.add(start_node)
        ref_idxs: List[int] = []
        hyp_idxs: List[int] = []
        while queue:
            node = queue.popleft()
            # Decode node id back to side-specific index.
            if node < n_ref:
                ref_idxs.append(node)
            else:
                hyp_idxs.append(node - n_ref)
            for nb in adj[node]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        # Keep only mixed components (both sides present).
        if ref_idxs and hyp_idxs:
            groups.append((ref_idxs, hyp_idxs))

    return groups


# ===================================================================
# 3.  VAD evaluation
# ===================================================================


# Difference: VAD is speaker-agnostic in pooled scoring and adds boundary MAE diagnostics.
def evaluate_vad(
    ref: pd.DataFrame,
    hyp: pd.DataFrame,
    collar: float = 0.25,
    skip_overlap: bool = False,
    per_speaker: bool = True,
) -> Dict[str, Any]:
    """Evaluate Voice Activity Detection quality.

    Computes detection error rate, precision, recall, and F1 using pyannote
    metrics (frame-level scoring with configurable collar).

    Also provides boundary quality statistics (onset/offset MAE) computed
    directly from matched segments. The per-speaker metrics use duration-based
    overlap and are effective for isolating performance on individual speakers,
    avoiding the brittleness of segment-counting approaches.

    Args:
        ref: Reference annotations (normalised DataFrame).
        hyp: Hypothesis annotations (normalised DataFrame).
        collar: Tolerance in seconds around reference boundaries (default 0.25 s).
        skip_overlap: If True, exclude overlapping regions from scoring.
        per_speaker: Compute per-speaker breakdowns as well.

        Returns:
                Dictionary with these keys:

                - ``pooled``: dict with
                    ``detection_error_rate``, ``detection_accuracy``, ``precision``,
                    ``recall``, ``f1``, ``n_ref_segments``, ``n_hyp_segments``,
                    ``onset_mae``, ``offset_mae``, ``onset_median_ae``,
                    ``offset_median_ae``, ``boundary_group_count``.
                - ``per_speaker``: primary per-speaker view (raw or mapped,
                    depending on label overlap policy).
                - ``per_speaker_raw``: optional raw-label per-speaker metrics.
                - ``per_speaker_mapped``: optional mapped-label per-speaker metrics.
                - ``speaker_mapping``: optional ref-speaker -> hyp-speaker mapping.
                - ``per_speaker_strategy``: optional strategy string.
                - ``shared_speaker_labels``: optional shared label list.
    """
    if not _HAS_PYANNOTE_METRICS:
        raise ImportError(
            "pyannote.metrics is required for VAD evaluation. "
            "Install with: pip install pyannote.metrics"
        )

    # Deterministic ordering is important because many helper functions use
    # row indices as stable ids (e.g., overlap edges and mapping matrix fills).
    ref = _ensure_sorted_segments(ref)
    hyp = _ensure_sorted_segments(hyp)

    results: Dict[str, Any] = {}

    # ---- pooled (speaker-agnostic) ----
    # Timeline/Annotation conversion ignores speaker identity for VAD scoring.
    ref_tl = _df_to_timeline(ref)
    hyp_tl = _df_to_timeline(hyp)

    # pyannote detection metrics work on Annotation objects (label is irrelevant)
    ref_ann = Annotation()
    for seg in ref_tl:
        ref_ann[seg] = "speech"
    hyp_ann = Annotation()
    for seg in hyp_tl:
        hyp_ann[seg] = "speech"

    det_er = DetectionErrorRate(collar=collar, skip_overlap=skip_overlap)
    det_prec = DetectionPrecision(collar=collar, skip_overlap=skip_overlap)
    det_rec = DetectionRecall(collar=collar, skip_overlap=skip_overlap)
    det_acc = DetectionAccuracy(collar=collar, skip_overlap=skip_overlap)

    # Explicit UEM avoids implicit pyannote approximation warnings.
    uem = _build_uem_from_df(ref, hyp)
    score_kwargs = {"uem": uem} if uem is not None else {}

    der_val = det_er(ref_ann, hyp_ann, **score_kwargs)
    prec_val = det_prec(ref_ann, hyp_ann, **score_kwargs)
    rec_val = det_rec(ref_ann, hyp_ann, **score_kwargs)
    acc_val = det_acc(ref_ann, hyp_ann, **score_kwargs)
    # Harmonic mean of precision/recall, guarded for zero denominator.
    f1_val = (
        2 * prec_val * rec_val / (prec_val + rec_val)
        if (prec_val + rec_val) > 0
        else 0.0
    )

    # Boundary quality diagnostic (separate from pyannote DER components).
    # Returns onset/offset absolute errors + group count.
    boundary = _compute_boundary_errors(ref, hyp)

    results["pooled"] = {
        "detection_error_rate": round(der_val, 4),
        "detection_accuracy": round(acc_val, 4),
        "precision": round(prec_val, 4),
        "recall": round(rec_val, 4),
        "f1": round(f1_val, 4),
        "n_ref_segments": len(ref),
        "n_hyp_segments": len(hyp),
        **boundary,
    }

    # ---- per-speaker ----
    if per_speaker:
        # Normalize label namespace to strings to avoid mixed int/str issues.
        ref_spk_series = ref["speaker"].astype(str)
        hyp_spk_series = hyp["speaker"].astype(str)
        ref_speakers = sorted(set(ref_spk_series.unique()))
        hyp_speakers = sorted(set(hyp_spk_series.unique()))
        shared_labels = sorted(set(ref_speakers) & set(hyp_speakers))

        strategy = "raw_and_mapped" if shared_labels else "mapped_only_no_label_overlap"
        results["per_speaker_strategy"] = strategy
        results["shared_speaker_labels"] = shared_labels

        # Cache prevents repeated recursive evaluate_vad() calls for the same
        # (ref_label, hyp_label) pair.
        score_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

        def _score_pair(ref_label: str, hyp_label: str) -> Dict[str, Any]:
            key = (ref_label, hyp_label)
            if key in score_cache:
                return score_cache[key]

            ref_spk_df = ref[ref_spk_series == ref_label]
            if ref_spk_df.empty:
                score_cache[key] = {"note": "no reference segments"}
                return score_cache[key]

            hyp_spk_df = hyp[hyp_spk_series == hyp_label]
            # Recursive call with per_speaker=False keeps recursion shallow:
            # it computes only pooled metrics for this label pair.
            score_cache[key] = evaluate_vad(
                ref_spk_df,
                hyp_spk_df,
                collar=collar,
                skip_overlap=skip_overlap,
                per_speaker=False,
            )["pooled"]
            return score_cache[key]

        # Mapping handles label permutation (e.g., ref A/B vs hyp P1/P2).
        mapping = _compute_speaker_overlap_mapping(ref, hyp)
        per_spk_mapped: Dict[str, Any] = {}
        for spk in ref_speakers:
            mapped_hyp_label = mapping.get(spk)
            if mapped_hyp_label is None:
                per_spk_mapped[spk] = {
                    "note": "no mapped hypothesis speaker",
                    "mapped_to": None,
                }
                continue
            per_spk_mapped[spk] = {
                "mapped_to": mapped_hyp_label,
                **_score_pair(spk, mapped_hyp_label),
            }

        if shared_labels:
            speakers = sorted(set(ref_speakers) | set(hyp_speakers))
            per_spk_raw: Dict[str, Any] = {}
            for spk in speakers:
                per_spk_raw[spk] = _score_pair(spk, spk)

            # Backward compatibility: keep historical `per_speaker` as raw view
            # when direct label overlap exists.
            results["per_speaker"] = per_spk_raw
            results["per_speaker_raw"] = per_spk_raw
        else:
            # No shared label namespace: expose mapped diagnostics as primary view
            results["per_speaker"] = per_spk_mapped

        results["per_speaker_mapped"] = per_spk_mapped
        results["speaker_mapping"] = mapping

    return results


def _compute_boundary_errors(ref: pd.DataFrame, hyp: pd.DataFrame) -> Dict[str, float]:
    """Compute onset/offset absolute boundary errors using many-to-many groups.

    For each overlap-connected component (many-to-many group), each reference
    boundary is matched to its nearest hypothesis boundary (ref→hyp only).
    This yields one onset and one offset distance per reference boundary while
    still allowing one hypothesis boundary to be nearest for multiple reference
    boundaries in split/merge cases.
    """
    if hyp.empty or ref.empty:
        return {
            "onset_mae": float("nan"),
            "offset_mae": float("nan"),
            "onset_median_ae": float("nan"),
            "offset_median_ae": float("nan"),
        }

    # Sort first so group ordering and index-based slicing are deterministic.
    ref = _ensure_sorted_segments(ref)
    hyp = _ensure_sorted_segments(hyp)
    groups = _match_segments_many_to_many(ref, hyp)
    ref_onsets_all = ref["start_sec"].to_numpy(dtype=float)
    ref_offsets_all = ref["end_sec"].to_numpy(dtype=float)
    hyp_onsets_all = hyp["start_sec"].to_numpy(dtype=float)
    hyp_offsets_all = hyp["end_sec"].to_numpy(dtype=float)
    onset_errs: list[float] = []
    offset_errs: list[float] = []

    def _nearest_abs_1d(
        query_vals: np.ndarray,
        candidate_vals: np.ndarray,
    ) -> np.ndarray:
        """Vectorized nearest absolute distance in 1-D (query->candidate)."""
        if query_vals.size == 0 or candidate_vals.size == 0:
            return np.array([], dtype=float)

        # Sort once, then binary-search insertion points for all queries.
        candidates_sorted = np.sort(candidate_vals)
        pos = np.searchsorted(candidates_sorted, query_vals, side="left")
        # Clamp indices for left/right neighbours.
        right_idx = np.minimum(pos, candidates_sorted.size - 1)
        left_idx = np.maximum(pos - 1, 0)
        right_dist = np.abs(candidates_sorted[right_idx] - query_vals)
        left_dist = np.abs(query_vals - candidates_sorted[left_idx])
        return cast(np.ndarray, np.minimum(left_dist, right_dist))

    for ref_idxs, hyp_idxs in groups:
        # IMPORTANT: ref_idxs and hyp_idxs are NOT pairwise-aligned arrays.
        # They are two index lists that represent one overlap-connected group:
        #   - ref_idxs: row indices into `ref`
        #   - hyp_idxs: row indices into `hyp`
        # Example many-to-one case (10 -> 1):
        #   ref_idxs = [0,1,2,3,4,5,6,7,8,9]
        #   hyp_idxs = [2]
        # This is valid and common when many short ref turns overlap one long
        # hyp segment. We do NOT force 1:1 boundary alignment here.
        # 1-D vectors of boundary timestamps (seconds).
        # Shapes:
        #   ref_onsets/ref_offsets -> (n_ref_in_group,)
        #   hyp_onsets/hyp_offsets -> (n_hyp_in_group,)
        ref_idx_arr = np.asarray(ref_idxs, dtype=np.int64)
        hyp_idx_arr = np.asarray(hyp_idxs, dtype=np.int64)
        ref_onsets = ref_onsets_all[ref_idx_arr]
        ref_offsets = ref_offsets_all[ref_idx_arr]
        hyp_onsets = hyp_onsets_all[hyp_idx_arr]
        hyp_offsets = hyp_offsets_all[hyp_idx_arr]

        if len(ref_onsets) == 0 or len(hyp_onsets) == 0:
            continue

        # Query is ref->hyp only (one nearest hyp boundary per ref boundary).
        # So if there are many ref boundaries and very few hyp boundaries,
        # multiple ref points can map to the SAME hyp point (reuse allowed).
        # Distances returned are absolute time errors in seconds.
        onset_ref_to_hyp = _nearest_abs_1d(ref_onsets, hyp_onsets)
        offset_ref_to_hyp = _nearest_abs_1d(ref_offsets, hyp_offsets)

        onset_errs.extend(onset_ref_to_hyp.astype(float).tolist())
        offset_errs.extend(offset_ref_to_hyp.astype(float).tolist())

    if not onset_errs:
        return {
            "onset_mae": float("nan"),
            "offset_mae": float("nan"),
            "onset_median_ae": float("nan"),
            "offset_median_ae": float("nan"),
        }

    # Aggregate global diagnostics across all groups.
    return {
        "onset_mae": round(float(np.mean(onset_errs)), 4),
        "offset_mae": round(float(np.mean(offset_errs)), 4),
        "onset_median_ae": round(float(np.median(onset_errs)), 4),
        "offset_median_ae": round(float(np.median(offset_errs)), 4),
        "boundary_group_count": len(groups),
    }


# Difference: speaker mapping is overlap-duration based and
# permutation-safe via Hungarian assignment.
def _compute_speaker_overlap_mapping(
    ref: pd.DataFrame,
    hyp: pd.DataFrame,
) -> Dict[str, str]:
    """Map reference speakers to hypothesis speakers by maximum overlap duration."""
    if ref.empty or hyp.empty:
        return {}
    if "speaker" not in ref.columns or "speaker" not in hyp.columns:
        return {}

    # Stable ordering keeps edge indices aligned with DataFrame row lookups.
    ref = _ensure_sorted_segments(ref)
    hyp = _ensure_sorted_segments(hyp)

    edges = _overlap_edges(ref, hyp)
    if not edges:
        return {}

    ref_labels = sorted({str(s) for s in ref["speaker"].dropna().unique()})
    hyp_labels = sorted({str(s) for s in hyp["speaker"].dropna().unique()})
    if not ref_labels or not hyp_labels:
        return {}

    # Lookup maps speaker label -> matrix index.
    ref_lookup = {lab: idx for idx, lab in enumerate(ref_labels)}
    hyp_lookup = {lab: idx for idx, lab in enumerate(hyp_labels)}
    # overlap_matrix dimensions:
    #   rows = n_ref_speakers
    #   cols = n_hyp_speakers
    # Cell [r, c] stores TOTAL overlapped duration between those speakers.
    overlap_matrix = np.zeros((len(ref_labels), len(hyp_labels)), dtype=float)

    # Per-row speaker labels (string-normalized once).
    ref_speakers = ref["speaker"].astype(str).to_numpy()
    hyp_speakers = hyp["speaker"].astype(str).to_numpy()
    ref_label_idx_per_row = np.array(
        [ref_lookup[label] for label in ref_speakers], dtype=np.int64
    )
    hyp_label_idx_per_row = np.array(
        [hyp_lookup[label] for label in hyp_speakers], dtype=np.int64
    )

    edge_ref_idx = np.fromiter((i for i, _, _ in edges), dtype=np.int64)
    edge_hyp_idx = np.fromiter((j for _, j, _ in edges), dtype=np.int64)
    edge_overlap = np.fromiter((ov for _, _, ov in edges), dtype=float)

    # Vectorized overlap accumulation into speaker-speaker matrix.
    np.add.at(
        overlap_matrix,
        (ref_label_idx_per_row[edge_ref_idx], hyp_label_idx_per_row[edge_hyp_idx]),
        edge_overlap,
    )

    if not np.any(overlap_matrix > 0):
        return {}

    # Hungarian solves a MIN-cost assignment, but we want MAX overlap.
    # Negating overlap converts maximize(sum(overlap)) into minimize(sum(cost)).
    cost = -overlap_matrix
    row_ind, col_ind = linear_sum_assignment(cost)

    # Mapping is partial if counts differ; we keep only positive-overlap pairs.
    mapping: Dict[str, str] = {}
    for r, c in zip(row_ind, col_ind):
        if overlap_matrix[r, c] > 0:
            mapping[ref_labels[r]] = hyp_labels[c]
    return mapping


# ===================================================================
# 4.  Diarization / speaker-attribution evaluation
# ===================================================================


# Difference: diarization keeps speaker identity and reports
# attribution/cluster metrics (DER/JER/IER/etc.).
def evaluate_diarization(
    ref: pd.DataFrame,
    hyp: pd.DataFrame,
    collar: float = 0.25,
    skip_overlap: bool = False,
    include_segment_metrics: bool = False,
) -> Dict[str, Any]:
    """Evaluate speaker diarization / attribution quality.

    Uses ``pyannote.metrics`` to compute a comprehensive set of metrics:

    - **DER** (Diarization Error Rate) with component breakdown
      (confusion, missed speech, false alarm).
    - **Greedy DER** — DER with greedy (not optimal) speaker mapping.
    - **JER** (Jaccard Error Rate) — overlap-aware per-speaker metric.
    - **Purity / Coverage** — cluster quality metrics.
    - **Homogeneity / Completeness** — information-theoretic quality.
    - **IER** (Identification Error Rate) — with components.

    Optional Segment Metrics (disabled by default because they can be brittle
    and misleading when speech is heavily fragmented; typically per-speaker VAD
    and DER are preferred since they are intrinsically duration-weighted):
    - **Speaker detection accuracy** — frame-level ACC.
    - **Speaker identification accuracy** — discrete utterance-level matching.

    Args:
        ref: Reference annotations with ``speaker`` column.
        hyp: Hypothesis annotations with ``speaker`` column.
        collar: Collar in seconds for boundary forgiveness.
        skip_overlap: Whether to exclude overlapping speech from scoring.
        include_segment_metrics: Compute and include brittle segment-counting SID metrics.

        Returns:
                Dictionary with keys:

                - ``diarization_error_rate``, ``greedy_diarization_error_rate``,
                    ``jaccard_error_rate``
                - ``der_miss``, ``der_false_alarm``, ``der_confusion``, ``der_total``
                - ``purity``, ``coverage``, ``homogeneity``, ``completeness``
                - ``identification_error_rate``, ``ier_miss``, ``ier_false_alarm``,
                    ``ier_confusion``
                - ``speaker_detection_accuracy`` (if include_segment_metrics=True)
                - ``speaker_id_accuracy`` (mapped), ``speaker_id_accuracy_raw`` (literal) (if True)
                - ``speaker_id_detail`` (nested mapping/per-speaker SID details) (if True)
                - ``collar``, ``skip_overlap``
    """
    if not _HAS_PYANNOTE_METRICS:
        raise ImportError(
            "pyannote.metrics is required for diarization evaluation. "
            "Install with: pip install pyannote.metrics"
        )

    ref_ann = _df_to_annotation(ref)
    hyp_ann = _df_to_annotation(hyp)

    # --- DER (optimal mapping) ---
    der_metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    uem = _build_uem_from_df(ref, hyp)
    score_kwargs = {"uem": uem} if uem is not None else {}

    der_val = der_metric(ref_ann, hyp_ann, **score_kwargs)
    der_details = der_metric(ref_ann, hyp_ann, detailed=True, **score_kwargs)

    # --- Greedy DER ---
    # Fast local-matching variant of DER (useful contrast against optimal DER).
    greedy_der_metric = GreedyDiarizationErrorRate(
        collar=collar, skip_overlap=skip_overlap
    )
    greedy_der_val = greedy_der_metric(ref_ann, hyp_ann, **score_kwargs)

    # --- JER ---
    jer_metric = JaccardErrorRate(collar=collar, skip_overlap=skip_overlap)
    jer_val = jer_metric(ref_ann, hyp_ann, **score_kwargs)

    # --- Purity / Coverage ---
    purity_metric = DiarizationPurity(collar=collar, skip_overlap=skip_overlap)
    purity_val = purity_metric(ref_ann, hyp_ann, **score_kwargs)

    coverage_metric = DiarizationCoverage(collar=collar, skip_overlap=skip_overlap)
    coverage_val = coverage_metric(ref_ann, hyp_ann, **score_kwargs)

    # --- Homogeneity / Completeness ---
    homogeneity_metric = DiarizationHomogeneity(
        collar=collar, skip_overlap=skip_overlap
    )
    homogeneity_val = homogeneity_metric(ref_ann, hyp_ann, **score_kwargs)

    completeness_metric = DiarizationCompleteness(
        collar=collar, skip_overlap=skip_overlap
    )
    completeness_val = completeness_metric(ref_ann, hyp_ann, **score_kwargs)

    # --- IER (Identification Error Rate) ---
    ier_metric = IdentificationErrorRate(collar=collar, skip_overlap=skip_overlap)
    ier_val = ier_metric(ref_ann, hyp_ann, **score_kwargs)
    ier_details = ier_metric(ref_ann, hyp_ann, detailed=True, **score_kwargs)

    if include_segment_metrics:
        # --- Speaker Detection Accuracy (frame-level) ---
        det_acc_metric = DetectionAccuracy(collar=collar, skip_overlap=skip_overlap)
        # For speaker detection accuracy we use speaker-labelled annotations
        det_acc_val = det_acc_metric(ref_ann, hyp_ann, **score_kwargs)

        # --- Speaker identification accuracy (segment-level) ---
        sid_acc = _compute_speaker_identification_accuracy(ref, hyp)
    else:
        det_acc_val = None
        sid_acc = {}

    return {
        # Core diarization (SUPERB SD)
        "diarization_error_rate": round(der_val, 4),
        "greedy_diarization_error_rate": round(greedy_der_val, 4),
        "jaccard_error_rate": round(jer_val, 4),
        # DER components
        "der_miss": round(float(der_details["missed detection"]), 4),
        "der_false_alarm": round(float(der_details["false alarm"]), 4),
        "der_confusion": round(float(der_details["confusion"]), 4),
        "der_total": round(float(der_details["total"]), 4),
        # Cluster quality
        "purity": round(purity_val, 4),
        "coverage": round(coverage_val, 4),
        # Information-theoretic
        "homogeneity": round(homogeneity_val, 4),
        "completeness": round(completeness_val, 4),
        # Identification
        "identification_error_rate": round(ier_val, 4),
        "ier_miss": round(float(ier_details.get("missed detection", 0)), 4),
        "ier_false_alarm": round(float(ier_details.get("false alarm", 0)), 4),
        "ier_confusion": round(float(ier_details.get("confusion", 0)), 4),
        # Speaker detection accuracy
        "speaker_detection_accuracy": (
            round(det_acc_val, 4) if det_acc_val is not None else None
        ),
        # Segment-level speaker ID accuracy (mapped + raw)
        "speaker_id_accuracy": (
            sid_acc.get("accuracy") if include_segment_metrics else None
        ),
        "speaker_id_accuracy_raw": (
            sid_acc.get("raw_accuracy") if include_segment_metrics else None
        ),
        "speaker_id_detail": sid_acc if include_segment_metrics else None,
        # Config
        "collar": collar,
        "skip_overlap": skip_overlap,
    }


# Difference: SID helper reports both raw label-match accuracy and mapped
# (label-permutation-safe) accuracy.
def _compute_speaker_identification_accuracy(
    ref: pd.DataFrame,
    hyp: pd.DataFrame,
) -> Dict[str, Any]:
    """Compute segment-level speaker identification accuracy (SUPERB SID).

    For each reference segment, finds the best-overlapping hypothesis
    segment and checks whether the speaker label matches.

    Reports **two** accuracy numbers:

    - **mapped** (Hungarian) — optimal mapping of ref labels → hyp labels.
      This is the standard metric for diarization systems that use
      arbitrary cluster IDs.
    - **raw** — literal label string comparison (no mapping).  Useful when
      the pipeline is expected to preserve canonical speaker names.

    Returns:
        Dictionary with keys:

        - ``accuracy``: mapped SID accuracy
        - ``raw_accuracy``: literal-label SID accuracy
        - ``n_matched``: number of matched segments used for scoring
        - ``n_correct``: mapped correct count
        - ``raw_n_correct``: raw correct count
        - ``speaker_mapping``: ref-label -> hyp-label assignment
        - ``per_speaker``: mapped per-ref-speaker stats
        - ``raw_per_speaker``: raw per-ref-speaker stats
    """
    pairs = _match_segments_by_overlap(ref, hyp)
    if not pairs:
        return {"accuracy": float("nan"), "raw_accuracy": float("nan"), "n_matched": 0}

    # Parallel speaker label vectors for matched segments.
    # Both arrays have identical length = total matched pairs.
    ref_speaker_by_row = ref["speaker"].astype(str).to_numpy()
    hyp_speaker_by_row = hyp["speaker"].astype(str).to_numpy()
    pair_ref_idx = np.fromiter((i for i, _ in pairs), dtype=np.int64)
    pair_hyp_idx = np.fromiter((j for _, j in pairs), dtype=np.int64)
    ref_spks = ref_speaker_by_row[pair_ref_idx]
    hyp_spks = hyp_speaker_by_row[pair_hyp_idx]
    total = len(pairs)

    # ---- Raw (literal) accuracy ----
    raw_match_mask = ref_spks == hyp_spks
    raw_correct = int(np.sum(raw_match_mask))
    raw_accuracy = round(raw_correct / total, 4) if total > 0 else 0.0

    # Per-speaker raw accuracy
    raw_per_speaker: Dict[str, Dict[str, Any]] = {}
    ref_labels = sorted({str(s) for s in ref_spks.tolist()})
    ref_lookup = {label: idx for idx, label in enumerate(ref_labels)}
    ref_label_ids = np.fromiter((ref_lookup[s] for s in ref_spks), dtype=np.int64)
    raw_total_by_ref = np.bincount(ref_label_ids, minlength=len(ref_labels))
    raw_correct_by_ref = np.bincount(
        ref_label_ids[raw_match_mask], minlength=len(ref_labels)
    )
    for rl_idx, rl in enumerate(ref_labels):
        n_segments = int(raw_total_by_ref[rl_idx])
        spk_correct = int(raw_correct_by_ref[rl_idx])
        raw_per_speaker[rl] = {
            "accuracy": round(spk_correct / n_segments, 4) if n_segments else 0.0,
            "n_segments": n_segments,
        }

    # ---- Mapped (Hungarian) accuracy ----
    hyp_labels = sorted({str(s) for s in hyp_spks.tolist()})
    hyp_lookup = {label: idx for idx, label in enumerate(hyp_labels)}
    hyp_label_ids = np.fromiter((hyp_lookup[s] for s in hyp_spks), dtype=np.int64)

    # Cost matrix dimensions:
    #   rows = unique ref speaker labels
    #   cols = unique hyp speaker labels
    # Matrix cell counts how often each (ref_label, hyp_label) pair co-occurs.
    # We negate counts because Hungarian minimizes cost.
    counts = np.zeros((len(ref_labels), len(hyp_labels)), dtype=np.int64)
    np.add.at(counts, (ref_label_ids, hyp_label_ids), 1)
    cost = -counts.astype(float)

    row_ind, col_ind = linear_sum_assignment(cost)
    mapping: Dict[str, str] = {}
    mapped_target_by_ref = np.full(len(ref_labels), -1, dtype=np.int64)
    for r, c in zip(row_ind, col_ind):
        mapping[ref_labels[r]] = hyp_labels[c]
        mapped_target_by_ref[r] = c

    # Compute accuracy with optimal mapping
    mapped_match_mask = hyp_label_ids == mapped_target_by_ref[ref_label_ids]
    mapped_correct = int(np.sum(mapped_match_mask))
    mapped_accuracy = round(mapped_correct / total, 4) if total > 0 else 0.0

    # Per-speaker mapped accuracy
    mapped_per_speaker: Dict[str, Dict[str, Any]] = {}
    mapped_correct_by_ref = np.bincount(
        ref_label_ids[mapped_match_mask], minlength=len(ref_labels)
    )
    for rl_idx, rl in enumerate(ref_labels):
        n_segments = int(raw_total_by_ref[rl_idx])
        spk_correct = int(mapped_correct_by_ref[rl_idx])
        mapped_per_speaker[rl] = {
            "accuracy": round(spk_correct / n_segments, 4) if n_segments else 0.0,
            "n_segments": n_segments,
            "mapped_to": mapping.get(rl, "?"),
        }

    return {
        # Mapped (optimal assignment) — primary metric
        "accuracy": mapped_accuracy,
        "n_matched": total,
        "n_correct": mapped_correct,
        "speaker_mapping": mapping,
        "per_speaker": mapped_per_speaker,
        # Raw (literal label match)
        "raw_accuracy": raw_accuracy,
        "raw_n_correct": raw_correct,
        "raw_per_speaker": raw_per_speaker,
    }


# ===================================================================
# 5.  Transcription evaluation (WER / CER)
# ===================================================================

# ---- Danish-aware text normalisation ----

_FILLER_RE = re.compile(
    r"\b(øh|øhm|hmm|mm|mhm|uh|uhm|um|ah|ehm|eh)\b",
    re.IGNORECASE | re.UNICODE,
)
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_MULTI_SPACE = re.compile(r"\s+")
_UF_RE = re.compile(r"\*\*\*UF\*\*\*", re.IGNORECASE)


def normalise_text(text: str, *, strip_fillers: bool = False) -> str:
    """Normalise text for WER/CER computation.

    Applies: Unicode NFKC → lowercase → optional filler removal →
    punctuation stripping → whitespace collapse.
    """
    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower()
    text = _UF_RE.sub("", text)
    if strip_fillers:
        text = _FILLER_RE.sub("", text)
    text = _PUNCT_RE.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text


# Difference: transcription uses many-to-many temporal grouping instead of
# strict 1:1 segment pairing.
def evaluate_transcription(
    ref: pd.DataFrame,
    hyp: pd.DataFrame,
    collar: float = 0.5,
    per_speaker: bool = True,
) -> Dict[str, Any]:
    """Evaluate transcription quality with WER and CER.

    Matching strategy: for each reference segment that has a non-empty
    transcription, the hypothesis segment with the highest temporal
    overlap is selected.  Only matched pairs are scored.

    Reports both **raw** (minimal normalisation) and **normalised**
    (filler/punctuation stripped) scores.

    Args:
        ref: Reference DataFrame with ``transcription`` column.
        hyp: Hypothesis DataFrame with ``transcription`` column.
        collar: Not used for text matching but kept for API consistency.
        per_speaker: Compute per-speaker breakdown.

        Returns:
                Dictionary with either a ``note`` (when not scorable) or:

                - ``pooled``: dict with
                    ``raw`` (WER/CER/MER/WIL/WIP + edit counts + optional BLEU/semantic),
                    ``normalised`` (same metric family after normalisation),
                    ``n_alignment_groups``, ``n_ref_segments_matched``,
                    ``n_ref_with_text``, ``coverage``.
                - ``per_speaker``: optional dict keyed by speaker; each value contains
                    ``raw``, ``normalised``, ``n_scored_segments``.
    """
    if not _HAS_JIWER:
        raise ImportError(
            "jiwer is required for transcription evaluation. "
            "Install with: pip install jiwer"
        )

    def _text_rows(df: pd.DataFrame) -> pd.DataFrame:
        out = df.dropna(subset=["transcription"]).copy()
        out = out[out["transcription"].astype(str).str.strip() != ""]
        out = out[
            ~out["transcription"]
            .astype(str)
            .str.contains(r"^\*\*\*UF\*\*\*$", regex=True)
        ]
        return out

    # Filter to rows with actual text
    ref_text = _text_rows(ref)

    if ref_text.empty:
        # Robust fallback for accidentally swapped CLI inputs:
        # if hyp has text and ref does not, score with swapped text sides.
        hyp_text_probe = _text_rows(hyp)
        if not hyp_text_probe.empty:
            swapped = evaluate_transcription(
                hyp,
                ref,
                collar=collar,
                per_speaker=per_speaker,
            )
            swapped_out = dict(swapped)
            swapped_out["text_reference_swapped"] = True
            swapped_out["text_reference_note"] = (
                "Reference input had no transcriptions; used hypothesis side as "
                "text reference for transcription scoring."
            )
            return swapped_out

        return {
            "note": "no reference transcriptions available",
            "coverage": 0.0,
        }

    hyp_text = _text_rows(hyp)

    # Many-to-many alignment: group overlapping segments, concatenate
    # texts within each group, then score group-by-group.  This avoids
    # the inflated WER that 1:1 matching produces when the pipeline
    # merges multiple short ref segments into a single long hyp segment.
    groups = _match_segments_many_to_many(ref_text, hyp_text)

    if not groups:
        return {
            "note": "no matching segments found between ref and hyp",
            "coverage": 0.0,
            "n_ref_with_text": len(ref_text),
        }

    ref_raw: list[str] = []
    hyp_raw: list[str] = []
    ref_norm: list[str] = []
    hyp_norm: list[str] = []
    speakers: list[str] = []
    n_ref_segments_matched = 0

    ref_transcriptions = ref_text["transcription"].astype(str).to_numpy()
    hyp_transcriptions = hyp_text["transcription"].astype(str).to_numpy()
    if "speaker" in ref_text.columns:
        ref_speakers = ref_text["speaker"].astype(str).to_numpy()
    else:
        ref_speakers = np.full(len(ref_text), "unknown", dtype=object)
    norm_cache: Dict[str, str] = {}

    def _normalise_cached(text: str) -> str:
        cached = norm_cache.get(text)
        if cached is not None:
            return cached
        normalized = normalise_text(text)
        norm_cache[text] = normalized
        return normalized

    for ref_idxs, hyp_idxs in groups:
        ref_idx_arr = np.asarray(ref_idxs, dtype=np.int64)
        hyp_idx_arr = np.asarray(hyp_idxs, dtype=np.int64)

        # Concatenate all ref texts in this alignment group
        r_parts = [text.strip() for text in ref_transcriptions[ref_idx_arr]]
        r_txt = " ".join(p for p in r_parts if p)
        # Concatenate all hyp texts in this alignment group
        h_parts = [text.strip() for text in hyp_transcriptions[hyp_idx_arr]]
        h_txt = " ".join(p for p in h_parts if p)
        if not r_txt:
            continue
        ref_raw.append(r_txt)
        hyp_raw.append(h_txt if h_txt else "")
        ref_norm.append(_normalise_cached(r_txt))
        hyp_norm.append(_normalise_cached(h_txt) if h_txt else "")
        # Speaker: majority speaker in the ref group
        spk_counts: Dict[str, int] = {}
        for s in ref_speakers[ref_idx_arr]:
            s = str(s)
            spk_counts[s] = spk_counts.get(s, 0) + 1
        speakers.append(max(spk_counts, key=spk_counts.get))  # type: ignore[arg-type]
        n_ref_segments_matched += int(ref_idx_arr.size)

    # Filter out empty normalised refs
    valid = [
        (rr, hr, rn, hn, s)
        for rr, hr, rn, hn, s in zip(ref_raw, hyp_raw, ref_norm, hyp_norm, speakers)
        if rn.strip()
    ]
    if not valid:
        return {
            "note": "all reference texts empty after normalisation",
            "coverage": 0.0,
        }

    ref_raw, hyp_raw, ref_norm, hyp_norm, speakers = zip(*valid)  # type: ignore[assignment]
    ref_raw = list(ref_raw)
    hyp_raw = list(hyp_raw)
    ref_norm = list(ref_norm)
    hyp_norm = list(hyp_norm)
    speakers = list(speakers)

    results: Dict[str, Any] = {}

    # ---- pooled scores ----
    results["pooled"] = {
        "raw": _compute_wer_cer(ref_raw, hyp_raw),
        "normalised": _compute_wer_cer(ref_norm, hyp_norm),
        "n_alignment_groups": len(ref_raw),
        "n_ref_segments_matched": n_ref_segments_matched,
        "n_ref_with_text": len(ref_text),
        "coverage": (
            round(n_ref_segments_matched / len(ref_text), 4) if len(ref_text) else 0.0
        ),
    }

    # ---- per-speaker ----
    if per_speaker:
        speaker_buckets: Dict[str, List[int]] = defaultdict(list)
        for idx, spk in enumerate(speakers):
            speaker_buckets[spk].append(idx)

        spk_set = sorted(speaker_buckets.keys())
        per_spk: Dict[str, Any] = {}
        for spk in spk_set:
            idxs = speaker_buckets[spk]
            sr = [ref_raw[i] for i in idxs]
            sh = [hyp_raw[i] for i in idxs]
            sn_r = [ref_norm[i] for i in idxs]
            sn_h = [hyp_norm[i] for i in idxs]
            per_spk[spk] = {
                "raw": _compute_wer_cer(sr, sh),
                "normalised": _compute_wer_cer(sn_r, sn_h),
                "n_scored_segments": len(idxs),
            }
        results["per_speaker"] = per_spk

    return results


def _compute_wer_cer(refs: List[str], hyps: List[str]) -> Dict[str, float]:
    """Compute WER, CER, MER, WIL, BLEU, and semantic similarity."""
    # Ensure jiwer can handle the input
    refs_clean = [r if r.strip() else "<empty>" for r in refs]
    hyps_clean = [h if h.strip() else "<empty>" for h in hyps]

    wer_out = jiwer.process_words(refs_clean, hyps_clean)
    cer_out = jiwer.process_characters(refs_clean, hyps_clean)

    result: Dict[str, Any] = {
        "wer": round(wer_out.wer, 4),
        "mer": round(wer_out.mer, 4),
        "wil": round(wer_out.wil, 4),
        "wip": round(1.0 - wer_out.wil, 4),  # Word Information Preserved
        "cer": round(cer_out.cer, 4),
        # word-level edit counts
        "substitutions": wer_out.substitutions,
        "deletions": wer_out.deletions,
        "insertions": wer_out.insertions,
        "hits": wer_out.hits,
    }

    # BLEU score (corpus-level)
    if _HAS_SACREBLEU:
        try:
            bleu = _sacrebleu.corpus_bleu(hyps_clean, [refs_clean])
            result["bleu"] = round(bleu.score, 2)
            result["bleu_bp"] = round(bleu.bp, 4)  # brevity penalty
        except Exception:
            result["bleu"] = None

    # Sentence-level semantic similarity (embedding cosine distance)
    if _HAS_TRANSFORMERS:
        try:
            sim = _compute_semantic_similarity(refs_clean, hyps_clean)
            result["semantic_similarity"] = round(sim["mean"], 4)
            result["semantic_distance"] = round(1.0 - sim["mean"], 4)
            result["semantic_similarity_std"] = round(sim["std"], 4)
            result["semantic_similarity_min"] = round(sim["min"], 4)
            result["semantic_similarity_max"] = round(sim["max"], 4)
        except Exception as e:
            result["semantic_similarity"] = None
            result["semantic_distance"] = None
            result["_semantic_error"] = str(e)

    return result


def _get_embedding_model(
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
) -> Tuple[Any, Any]:
    """Load (and cache) a sentence embedding model.

    Uses a lightweight multilingual model by default which supports Danish
    and many other languages out of the box.
    """
    if "model" not in _EMBED_MODEL_CACHE:
        tokenizer = _AutoTokenizer.from_pretrained(model_name)
        model = _AutoModel.from_pretrained(model_name)
        model.eval()
        _EMBED_MODEL_CACHE["tokenizer"] = tokenizer
        _EMBED_MODEL_CACHE["model"] = model
        _EMBED_MODEL_CACHE["name"] = model_name
    return _EMBED_MODEL_CACHE["tokenizer"], _EMBED_MODEL_CACHE["model"]


def _mean_pool(model_output: Any, attention_mask: Any) -> Any:
    """Mean pooling over token embeddings, respecting the attention mask."""
    token_embeddings = model_output.last_hidden_state
    input_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * input_mask).sum(1) / input_mask.sum(1).clamp(min=1e-9)


def _compute_semantic_similarity(
    refs: List[str],
    hyps: List[str],
    batch_size: int = 32,
) -> Dict[str, float]:
    """Compute mean cosine similarity between ref/hyp sentence embeddings.

    Uses a multilingual sentence-transformer model to embed each sentence,
    then computes per-pair cosine similarity.

    Returns dict with mean, std, min, max similarity scores (range 0-1).
    """
    tokenizer, model = _get_embedding_model()
    device = next(model.parameters()).device

    def _encode(texts: List[str]) -> np.ndarray:
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            with _torch.no_grad():
                out = model(**encoded)
            embs = _mean_pool(out, encoded["attention_mask"])
            # L2-normalise
            embs = _torch.nn.functional.normalize(embs, p=2, dim=1)
            all_embs.append(embs.cpu().numpy())
        return np.concatenate(all_embs, axis=0)

    ref_embs = _encode(refs)
    hyp_embs = _encode(hyps)

    # Per-pair cosine similarity (already L2-normed, so dot product suffices)
    sims = np.sum(ref_embs * hyp_embs, axis=1)
    # Clamp to [0, 1] (negative similarities → 0)
    sims = np.clip(sims, 0.0, 1.0)

    return {
        "mean": float(np.mean(sims)),
        "std": float(np.std(sims)),
        "min": float(np.min(sims)),
        "max": float(np.max(sims)),
        "per_pair": sims.tolist(),
    }


def _match_segments_by_overlap(
    ref: pd.DataFrame, hyp: pd.DataFrame
) -> List[Tuple[int, int]]:
    """Optimally match ref segments to hyp segments by maximizing temporal overlap.

    Uses the Hungarian algorithm (Linear Sum Assignment) to prevent overlap-stealing.
    Returns list of (ref_idx, hyp_idx) pairs.  Each hyp segment is used
    at most once.
    """
    if ref.empty or hyp.empty:
        return []

    # Keep positional index mapping so returned pairs always reference
    # the caller's original row positions (expected by downstream `.iloc`).
    ref = ref.copy()
    hyp = hyp.copy()
    ref["_pos_idx"] = np.arange(len(ref), dtype=np.int64)
    hyp["_pos_idx"] = np.arange(len(hyp), dtype=np.int64)
    ref = _ensure_sorted_segments(ref)
    hyp = _ensure_sorted_segments(hyp)

    # Candidate pair list with overlap weights.
    pairs = _overlap_edges(ref, hyp)
    if not pairs:
        return []

    pair_ref_idx = np.fromiter((i for i, _, _ in pairs), dtype=np.int64)
    pair_hyp_idx = np.fromiter((j for _, j, _ in pairs), dtype=np.int64)
    pair_overlap = np.fromiter((ov for _, _, ov in pairs), dtype=float)

    overlap_matrix = np.zeros((len(ref), len(hyp)), dtype=float)
    np.add.at(
        overlap_matrix,
        (pair_ref_idx, pair_hyp_idx),
        pair_overlap,
    )

    if not np.any(overlap_matrix > 0):
        return []

    # Hungarian solves a MIN-cost assignment, but we want MAX overlap.
    cost = -overlap_matrix
    row_ind, col_ind = linear_sum_assignment(cost)

    matched: list[tuple[int, int]] = []
    for r, c in zip(row_ind, col_ind):
        if overlap_matrix[r, c] > 0:
            matched.append((int(ref.iloc[r]["_pos_idx"]), int(hyp.iloc[c]["_pos_idx"])))

    return matched


def _match_segments_many_to_many(
    ref: pd.DataFrame, hyp: pd.DataFrame
) -> List[Tuple[List[int], List[int]]]:
    """Build many-to-many alignment groups between ref and hyp segments.

    Instead of a strict 1:1 mapping, this groups together all ref and hyp
    segments that share a connected region of temporal overlap.  Within
    each group the concatenated texts form a fair unit for WER/CER.

    This fixes the inflated WER problem that occurs when the pipeline
    merges several short reference utterances into a single long
    hypothesis segment (or vice versa).

    Algorithm:
        1. Build a bipartite graph: ref_i ↔ hyp_j whenever they overlap.
        2. Find connected components via BFS.
        3. Return each component as ``([ref_indices], [hyp_indices])``
           sorted by start time.

    Returns:
        List of ``(ref_idx_list, hyp_idx_list)`` groups.
    """
    if ref.empty or hyp.empty:
        return []

    ref = _ensure_sorted_segments(ref)
    hyp = _ensure_sorted_segments(hyp)

    # Cached start-time vectors used for deterministic ordering of each group.
    # Shapes: (n_ref,), (n_hyp,)
    ref_starts = ref["start_sec"].values.astype(float)
    hyp_starts = hyp["start_sec"].values.astype(float)

    edges = _overlap_edges(ref, hyp)
    groups = _many_to_many_groups_from_edges(len(ref), len(hyp), edges)

    # Sort members within each component by onset time for stable downstream use.
    for component_ref, component_hyp in groups:
        component_ref.sort(key=lambda i: ref_starts[i])
        component_hyp.sort(key=lambda j: hyp_starts[j])

    # Sort whole group list by first reference onset.
    groups.sort(key=lambda g: ref_starts[g[0][0]])
    return groups


# ===================================================================
# 5b. Segmentation quality evaluation
# ===================================================================


# Difference: segmentation evaluates boundary/partition quality
# independent of speaker-label identity.
def evaluate_segmentation(
    ref: pd.DataFrame,
    hyp: pd.DataFrame,
    collar: float = 0.25,
) -> Dict[str, Any]:
    """Evaluate segmentation quality (boundary placement).

    Uses ``pyannote.metrics.segmentation`` to compute:
    - **Segmentation purity** — proportion of each hyp segment that
      overlaps with a single reference segment.
    - **Segmentation coverage** — proportion of each ref segment that
      overlaps with a single hypothesis segment.
    - **Segmentation precision / recall** — boundary detection metrics.

    These metrics specifically assess how well the system places segment
    boundaries, independent of speaker identity.

    Args:
        ref: Reference annotations (normalised DataFrame).
        hyp: Hypothesis annotations (normalised DataFrame).
        collar: Tolerance around boundaries (seconds).

    Returns:
        Dictionary with keys:

        - ``segmentation_purity``
        - ``segmentation_coverage``
        - ``segmentation_precision``
        - ``segmentation_recall``
        - ``segmentation_f_measure``
    """
    if not _HAS_PYANNOTE_METRICS:
        raise ImportError(
            "pyannote.metrics is required for segmentation evaluation. "
            "Install with: pip install pyannote.metrics"
        )

    ref_ann = _df_to_annotation(ref)
    hyp_ann = _df_to_annotation(hyp)

    # Segmentation purity & coverage
    seg_purity = SegmentationPurity(tolerance=collar)
    seg_coverage = SegmentationCoverage(tolerance=collar)
    uem = _build_uem_from_df(ref, hyp)
    score_kwargs = {"uem": uem} if uem is not None else {}

    purity_val = seg_purity(ref_ann, hyp_ann, **score_kwargs)
    coverage_val = seg_coverage(ref_ann, hyp_ann, **score_kwargs)

    # Segmentation precision & recall (boundary-based)
    seg_prec = SegmentationPrecision(tolerance=collar)
    seg_rec = SegmentationRecall(tolerance=collar)
    prec_val = seg_prec(ref_ann, hyp_ann, **score_kwargs)
    rec_val = seg_rec(ref_ann, hyp_ann, **score_kwargs)

    # F-measure
    seg_f = SegmentationPurityCoverageFMeasure(tolerance=collar)
    f_val = seg_f(ref_ann, hyp_ann, **score_kwargs)

    return {
        "segmentation_purity": round(purity_val, 4),
        "segmentation_coverage": round(coverage_val, 4),
        "segmentation_precision": round(prec_val, 4),
        "segmentation_recall": round(rec_val, 4),
        "segmentation_f_measure": round(f_val, 4),
    }


# ===================================================================
# 6.  Label-type classification evaluation
# ===================================================================


# Difference: label-type evaluates categorical class agreement on overlap-matched segments.
def evaluate_label_type(
    ref: pd.DataFrame,
    hyp: pd.DataFrame,
    overlap_threshold: Union[float, Dict[str, float]] = 0.5,
) -> Dict[str, Any]:
    """Evaluate classification of segment types (turn / backchannel / overlapped_turn).

    Matches segments by temporal overlap, then compares the ``type`` column.
    Enforces a threshold (e.g., minimum overlap/ref_duration) before counting as a match.

    Returns:
        Dictionary with either a ``note`` (when not scorable) or:

        - ``n_matched``
        - ``macro_precision``, ``macro_recall``, ``macro_f1``
        - ``per_class``: per-label precision/recall/f1/support
        - ``confusion_matrix``: nested dict [ref_label][hyp_label] -> count
    """
    if "type" not in ref.columns or "type" not in hyp.columns:
        return {"note": "type column missing in ref or hyp"}

    # Skip when reference has no backchannel/overlapped_turn distinction.
    # This is common for reference files that only record speech segments without
    # turn-type annotations (e.g. diarization output, .ass subtitle files).
    ref_unique_types = {
        str(t).strip().lower() for t in ref["type"].dropna() if str(t).strip()
    }
    non_turn_ref = ref_unique_types - {"turn", "t"}
    if not non_turn_ref:
        return {
            "note": (
                "Label-type evaluation skipped: the reference contains only 'turn' labels "
                "with no backchannel or overlapped_turn distinction. "
                "This typically means the reference file has no turn/backchannel annotations "
                "(e.g. a diarization output or .ass subtitle file). "
                "Label-type metrics cannot be meaningfully computed without multiple "
                "label classes in the reference."
            )
        }

    pairs = _match_segments_by_overlap(ref, hyp)

    matched_ref_idxs = set()
    matched_hyp_idxs = set()
    valid_pairs = []

    for i, j in pairs:
        r_start = float(ref.iloc[i]["start_sec"])
        r_end = float(ref.iloc[i]["end_sec"])
        h_start = float(hyp.iloc[j]["start_sec"])
        h_end = float(hyp.iloc[j]["end_sec"])

        overlap = min(r_end, h_end) - max(r_start, h_start)
        r_dur = r_end - r_start
        if r_dur <= 0:
            continue

        r_type = str(ref.iloc[i]["type"])
        if isinstance(overlap_threshold, dict):
            thresh = overlap_threshold.get(r_type, 0.5)
        else:
            thresh = overlap_threshold

        if (overlap / r_dur) >= thresh:
            valid_pairs.append((i, j))
            matched_ref_idxs.add(i)
            matched_hyp_idxs.add(j)

    ref_types = [str(t) for t in ref["type"].dropna().unique()]
    hyp_types = [str(t) for t in hyp["type"].dropna().unique()]
    all_labels = sorted(set(ref_types) | set(hyp_types))

    if not all_labels:
        return {"note": "no labels found for label-type evaluation"}

    # Build confusion matrix including "None" for unmatched
    confusion: Dict[str, Dict[str, int]] = {
        label: {m: 0 for m in all_labels + ["None"]} for label in all_labels + ["None"]
    }

    for i, j in valid_pairs:
        rt = str(ref.iloc[i]["type"])
        ht = str(hyp.iloc[j]["type"])
        confusion[rt][ht] += 1

    for i in range(len(ref)):
        if i not in matched_ref_idxs:
            rt = str(ref.iloc[i]["type"])
            confusion[rt]["None"] += 1

    for j in range(len(hyp)):
        if j not in matched_hyp_idxs:
            ht = str(hyp.iloc[j]["type"])
            confusion["None"][ht] += 1

    # Per-class metrics
    per_class: Dict[str, Dict[str, float]] = {}
    for label in all_labels:
        tp = confusion.get(label, {}).get(label, 0)
        fp = sum(
            confusion.get(other, {}).get(label, 0)
            for other in all_labels + ["None"]
            if other != label
        )
        fn = sum(
            confusion.get(label, {}).get(other, 0)
            for other in all_labels + ["None"]
            if other != label
        )
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[label] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }

    # Macro averages
    if per_class:
        macro_prec = float(np.mean([v["precision"] for v in per_class.values()]))
        macro_rec = float(np.mean([v["recall"] for v in per_class.values()]))
        macro_f1 = float(np.mean([v["f1"] for v in per_class.values()]))
    else:
        macro_prec, macro_rec, macro_f1 = 0.0, 0.0, 0.0

    return {
        "n_matched": len(valid_pairs),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


# ===================================================================
# 7.  Top-level orchestrator
# ===================================================================


# Difference: pipeline orchestrator composes stage evaluators and isolates per-stage failures.
def evaluate_pipeline(
    ref_path: str,
    hyp_path: str,
    *,
    ref_fmt: Optional[str] = None,
    hyp_fmt: Optional[str] = None,
    stages: Optional[Sequence[str]] = None,
    collar: float = 0.25,
    skip_overlap: bool = False,
    output_path: Optional[str] = None,
    label_overlap_threshold: Union[float, Dict[str, float]] = 0.5,
) -> Dict[str, Any]:
    """Run evaluation for one or more pipeline stages.

    Args:
        ref_path: Path to ground-truth annotation file.
        hyp_path: Path to pipeline hypothesis file (e.g. ``final_labels.txt``).
        ref_fmt: Override auto-detected format for references.
        hyp_fmt: Override auto-detected format for hypotheses.
        stages: Subset of stages to evaluate.  Default is all available:
            ``["vad", "diarization", "segmentation", "transcription", "label_type"]``.
        collar: Collar in seconds for VAD / diarization boundary tolerance.
        skip_overlap: Whether to skip overlapping regions.
        output_path: If given, write JSON results to this path.
        label_overlap_threshold: Minimum overlap proportion for label-type evaluation.

        Returns:
                Dictionary with top-level keys:

                - ``ref_path``, ``hyp_path``, ``collar``
                - stage keys among ``vad``, ``diarization``, ``segmentation``,
                    ``transcription``, ``label_type`` (depending on ``stages``)

                Each stage value is either:
                - the stage result dictionary from the corresponding ``evaluate_*``
                    function, or
                - ``{"error": "..."}`` if that stage failed.
    """
    all_stages = ["vad", "diarization", "segmentation", "transcription", "label_type"]
    if stages is None:
        stages = all_stages
    else:
        unknown = set(stages) - set(all_stages)
        if unknown:
            raise ValueError(f"Unknown evaluation stages: {unknown}")

    ref = (
        load_reference(ref_path, fmt=ref_fmt)
        .sort_values(["start_sec"])
        .reset_index(drop=True)
    )
    hyp = (
        load_reference(hyp_path, fmt=hyp_fmt)
        .sort_values(["start_sec"])
        .reset_index(drop=True)
    )

    results: Dict[str, Any] = {
        "ref_path": ref_path,
        "hyp_path": hyp_path,
        "collar": collar,
    }

    if "vad" in stages:
        try:
            results["vad"] = evaluate_vad(
                ref, hyp, collar=collar, skip_overlap=skip_overlap
            )
        except Exception as exc:
            results["vad"] = {"error": str(exc)}

    if "diarization" in stages:
        try:
            results["diarization"] = evaluate_diarization(
                ref, hyp, collar=collar, skip_overlap=skip_overlap
            )
        except Exception as exc:
            results["diarization"] = {"error": str(exc)}

    if "segmentation" in stages:
        try:
            results["segmentation"] = evaluate_segmentation(ref, hyp, collar=collar)
        except Exception as exc:
            results["segmentation"] = {"error": str(exc)}

    if "transcription" in stages:
        try:
            results["transcription"] = evaluate_transcription(ref, hyp, collar=collar)
        except Exception as exc:
            results["transcription"] = {"error": str(exc)}

    if "label_type" in stages:
        try:
            results["label_type"] = evaluate_label_type(
                ref, hyp, overlap_threshold=label_overlap_threshold
            )
        except Exception as exc:
            results["label_type"] = {"error": str(exc)}

    # Write results
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        print(f"Evaluation results written to {output_path}")

    return results


# Difference: directory orchestrator auto-discovers hypothesis files
# before calling evaluate_pipeline.
def evaluate_pipeline_from_dir(
    ref_path: str,
    output_dir: str,
    *,
    ref_fmt: Optional[str] = None,
    stages: Optional[Sequence[str]] = None,
    collar: float = 0.25,
    skip_overlap: bool = False,
    label_overlap_threshold: Union[float, Dict[str, float]] = 0.5,
) -> Dict[str, Any]:
    """Evaluate a pipeline output directory against a single reference file.

    Automatically locates ``final_labels.txt`` in *output_dir* and runs
    all requested stage evaluations.

    Args:
        ref_path: Path to ground-truth annotation file.
        output_dir: Pipeline output directory (contains ``final_labels.txt``).
        ref_fmt: Override auto-detected format for references.
        stages: Which stages to evaluate (default: all available).
        collar: Boundary tolerance in seconds.
        skip_overlap: Exclude overlapping speech regions.
        label_overlap_threshold: Minimum overlap proportion for label-type evaluation.

    Returns:
        Same schema as :func:`evaluate_pipeline`.

        The function additionally resolves ``hyp_path`` automatically from
        ``output_dir`` and writes ``evaluation_metrics.json`` in that directory.
    """
    # Try to find hypothesis file
    candidates = [
        "final_labels.txt",
        "final_labels_elan.txt",
        "classified_transcriptions.txt",
    ]
    hyp_path: Optional[str] = None
    for c in candidates:
        p = os.path.join(output_dir, c)
        if os.path.exists(p):
            hyp_path = p
            break

    if hyp_path is None:
        raise FileNotFoundError(
            f"No hypothesis file found in {output_dir}. " f"Looked for: {candidates}"
        )

    eval_output = os.path.join(output_dir, "evaluation_metrics.json")

    return evaluate_pipeline(
        ref_path=ref_path,
        hyp_path=hyp_path,
        ref_fmt=ref_fmt,
        stages=stages,
        collar=collar,
        skip_overlap=skip_overlap,
        output_path=eval_output,
        label_overlap_threshold=label_overlap_threshold,
    )


# ===================================================================
# 8.  Pretty-print summary
# ===================================================================


def print_evaluation_summary(results: Dict[str, Any]) -> None:
    """Print a human-readable summary of evaluation results."""
    print("\n" + "=" * 70)
    print("  PIPELINE EVALUATION SUMMARY")
    print("=" * 70)

    if "ref_path" in results:
        print(f"  Reference: {results['ref_path']}")
    if "hyp_path" in results:
        print(f"  Hypothesis: {results['hyp_path']}")
    print()

    # VAD
    if "vad" in results:
        vad = results["vad"]
        if "error" in vad:
            print(f"  [VAD] Error: {vad['error']}")
        else:
            p = vad.get("pooled", vad)
            print("  [VAD — Speech Activity Detection]")
            print(f"    Detection Error Rate : {p.get('detection_error_rate', 'N/A')}")
            print(f"    Detection Accuracy   : {p.get('detection_accuracy', 'N/A')}")
            print(f"    Precision            : {p.get('precision', 'N/A')}")
            print(f"    Recall               : {p.get('recall', 'N/A')}")
            print(f"    F1                   : {p.get('f1', 'N/A')}")
            print(f"    Onset MAE (s)        : {p.get('onset_mae', 'N/A')}")
            print(f"    Offset MAE (s)       : {p.get('offset_mae', 'N/A')}")
            print(
                f"    Segments (ref/hyp)   : {p.get('n_ref_segments', '?')}"
                f" / {p.get('n_hyp_segments', '?')}"
            )
            if "per_speaker_strategy" in vad:
                print(f"    Per-speaker strategy : {vad.get('per_speaker_strategy')}")
            if "shared_speaker_labels" in vad:
                print(f"    Shared labels        : {vad.get('shared_speaker_labels')}")
            if "speaker_mapping" in vad and vad.get("speaker_mapping"):
                print(f"    Speaker mapping      : {vad.get('speaker_mapping')}")
            if "per_speaker_raw" in vad:
                print("    Per-speaker (raw labels):")
                for spk, sv in vad["per_speaker_raw"].items():
                    if "note" in sv:
                        print(f"      {spk}: {sv['note']}")
                    else:
                        print(
                            f"      {spk}: DER={sv.get('detection_error_rate','?')}"
                            f"  P={sv.get('precision','?')}"
                            f"  R={sv.get('recall','?')}"
                            f"  F1={sv.get('f1','?')}"
                        )
            elif (
                "per_speaker" in vad
                and vad.get("per_speaker_strategy") != "mapped_only_no_label_overlap"
            ):
                print("    Per-speaker:")
                for spk, sv in vad["per_speaker"].items():
                    if "note" in sv:
                        print(f"      {spk}: {sv['note']}")
                    else:
                        print(
                            f"      {spk}: DER={sv.get('detection_error_rate','?')}"
                            f"  P={sv.get('precision','?')}"
                            f"  R={sv.get('recall','?')}"
                            f"  F1={sv.get('f1','?')}"
                        )
            if "per_speaker_mapped" in vad:
                print("    Per-speaker (mapped labels):")
                for spk, sv in vad["per_speaker_mapped"].items():
                    if "note" in sv:
                        print(
                            f"      {spk}: {sv['note']}"
                            f" (mapped_to={sv.get('mapped_to')})"
                        )
                    else:
                        print(
                            f"      {spk}: DER={sv.get('detection_error_rate','?')}"
                            f"  P={sv.get('precision','?')}"
                            f"  R={sv.get('recall','?')}"
                            f"  F1={sv.get('f1','?')}"
                            f"  (mapped_to={sv.get('mapped_to')})"
                        )
        print()

    # Diarization
    if "diarization" in results:
        diar = results["diarization"]
        if "error" in diar:
            print(f"  [Diarization] Error: {diar['error']}")
        else:
            print("  [Diarization — Speaker Attribution]")
            print(
                f"    DER                  : {diar.get('diarization_error_rate', 'N/A')}"
            )
            print(
                f"    Greedy DER           : {diar.get('greedy_diarization_error_rate', 'N/A')}"
            )
            print(f"    JER                  : {diar.get('jaccard_error_rate', 'N/A')}")
            print(f"    Missed speech        : {diar.get('der_miss', 'N/A')}")
            print(f"    False alarm          : {diar.get('der_false_alarm', 'N/A')}")
            print(f"    Speaker confusion    : {diar.get('der_confusion', 'N/A')}")
            print(f"    Purity               : {diar.get('purity', 'N/A')}")
            print(f"    Coverage             : {diar.get('coverage', 'N/A')}")
            print(f"    Homogeneity          : {diar.get('homogeneity', 'N/A')}")
            print(f"    Completeness         : {diar.get('completeness', 'N/A')}")
            # IER
            ier = diar.get("identification_error_rate")
            if ier is not None:
                print(f"    IER                  : {ier}")
                print(
                    f"      IER Precision      : {diar.get('identification_precision', 'N/A')}"
                )
                print(
                    f"      IER Recall         : {diar.get('identification_recall', 'N/A')}"
                )
            # Speaker detection accuracy
            sda = diar.get("speaker_detection_accuracy")
            if sda is not None:
                print(f"    Speaker Det. Acc.    : {sda}")
            # Speaker ID accuracy (segment-level)
            sid_mapped = diar.get("speaker_id_accuracy")
            sid_raw = diar.get("speaker_id_accuracy_raw")
            if sid_mapped is not None:
                print(f"    SID Acc. (mapped)    : {sid_mapped}")
            if sid_raw is not None:
                print(f"    SID Acc. (raw)       : {sid_raw}")
            sid_detail = diar.get("speaker_id_detail")
            if not isinstance(sid_detail, dict):
                sid_detail = {}
            mapping = sid_detail.get("speaker_mapping", {})
            if mapping:
                print(f"      Mapping            : {mapping}")
            per_spk = sid_detail.get("per_speaker", {})
            for spk, info in per_spk.items():
                print(
                    f"      {spk:18s}: mapped={info.get('accuracy','?')}"
                    f"  (→{info.get('mapped_to','?')}, n={info.get('n_segments','?')})"
                )
            raw_per_spk = sid_detail.get("raw_per_speaker", {})
            if raw_per_spk:
                for spk, info in raw_per_spk.items():
                    print(
                        f"      {spk:18s}: raw={info.get('accuracy','?')}"
                        f"  (n={info.get('n_segments','?')})"
                    )
        print()

    # Segmentation
    if "segmentation" in results:
        seg = results["segmentation"]
        if "error" in seg:
            print(f"  [Segmentation] Error: {seg['error']}")
        else:
            print("  [Segmentation — Boundary Quality]")
            print(
                f"    Purity               : {seg.get('segmentation_purity', seg.get('purity', 'N/A'))}"
            )
            print(
                f"    Coverage             : {seg.get('segmentation_coverage', seg.get('coverage', 'N/A'))}"
            )
            print(
                f"    Precision            : {seg.get('segmentation_precision', seg.get('precision', 'N/A'))}"
            )
            print(
                f"    Recall               : {seg.get('segmentation_recall', seg.get('recall', 'N/A'))}"
            )
            print(
                f"    F-measure            : {seg.get('segmentation_f_measure', seg.get('f_measure', 'N/A'))}"
            )
        print()

    # Transcription
    if "transcription" in results:
        asr = results["transcription"]
        if "error" in asr:
            print(f"  [Transcription] Error: {asr['error']}")
        elif "note" in asr:
            print(f"  [Transcription] {asr['note']}")
        else:
            pooled = asr.get("pooled", {})
            raw = pooled.get("raw", {})
            norm = pooled.get("normalised", {})
            print("  [Transcription — ASR Quality]")
            n_groups = pooled.get(
                "n_alignment_groups", pooled.get("n_scored_segments", "?")
            )
            n_matched = pooled.get("n_ref_segments_matched", n_groups)
            print(
                f"    Coverage             : {pooled.get('coverage', 'N/A')}"
                f"  ({n_matched}"
                f" / {pooled.get('n_ref_with_text', '?')} ref segs"
                f" in {n_groups} alignment groups)"
            )
            print("    --- Raw ---")
            print(f"    WER                  : {raw.get('wer', 'N/A')}")
            print(f"    CER                  : {raw.get('cer', 'N/A')}")
            print(f"    MER                  : {raw.get('mer', 'N/A')}")
            print(f"    WIL                  : {raw.get('wil', 'N/A')}")
            print(f"    WIP                  : {raw.get('wip', 'N/A')}")
            if raw.get("bleu") is not None:
                print(f"    BLEU                 : {raw.get('bleu')}")
            if raw.get("semantic_distance") is not None:
                print(f"    Semantic Distance    : {raw.get('semantic_distance')}")
                print(
                    f"    Semantic Similarity  : {raw.get('semantic_similarity')}"
                    f"  (std={raw.get('semantic_similarity_std', '?')}"
                    f"  min={raw.get('semantic_similarity_min', '?')}"
                    f"  max={raw.get('semantic_similarity_max', '?')})"
                )
            edits = raw.get("substitutions")
            if edits is not None:
                print(
                    f"    Edits (S/D/I/H)      : {raw.get('substitutions',0)}"
                    f" / {raw.get('deletions',0)}"
                    f" / {raw.get('insertions',0)}"
                    f" / {raw.get('hits',0)}"
                )
            print("    --- Normalised ---")
            print(f"    WER                  : {norm.get('wer', 'N/A')}")
            print(f"    CER                  : {norm.get('cer', 'N/A')}")
            print(f"    MER                  : {norm.get('mer', 'N/A')}")
            print(f"    WIL                  : {norm.get('wil', 'N/A')}")
            print(f"    WIP                  : {norm.get('wip', 'N/A')}")
            if norm.get("bleu") is not None:
                print(f"    BLEU                 : {norm.get('bleu')}")
            if norm.get("semantic_distance") is not None:
                print(f"    Semantic Distance    : {norm.get('semantic_distance')}")
                print(
                    f"    Semantic Similarity  : {norm.get('semantic_similarity')}"
                    f"  (std={norm.get('semantic_similarity_std', '?')}"
                    f"  min={norm.get('semantic_similarity_min', '?')}"
                    f"  max={norm.get('semantic_similarity_max', '?')})"
                )
            if "per_speaker" in asr:
                print("    --- Per Speaker --")
                for spk, sv in asr["per_speaker"].items():
                    sr = sv.get("raw", {})
                    sn = sv.get("normalised", {})
                    print(
                        f"    {spk}: WER(raw)={sr.get('wer','?')}"
                        f"  WER(norm)={sn.get('wer','?')}"
                        f"  CER(norm)={sn.get('cer','?')}"
                        f"  BLEU(norm)={sn.get('bleu','?')}"
                        f"  SemDist={sn.get('semantic_distance','?')}"
                        f"  (n={sv.get('n_scored_segments','?')})"
                    )
        print()

    # Label type
    if "label_type" in results:
        lt = results["label_type"]
        if "error" in lt:
            print(f"  [Label Type] Error: {lt['error']}")
        elif "note" in lt:
            print(f"  [Label Type] {lt['note']}")
        else:
            print("  [Label Type — Classification Quality]")
            print(f"    Macro F1             : {lt.get('macro_f1', 'N/A')}")
            print(f"    Macro Precision      : {lt.get('macro_precision', 'N/A')}")
            print(f"    Macro Recall         : {lt.get('macro_recall', 'N/A')}")
            print(f"    Matched segments     : {lt.get('n_matched', 'N/A')}")
            if "per_class" in lt:
                for cls, cv in lt["per_class"].items():
                    print(
                        f"    {cls:20s}: P={cv.get('precision','?')}"
                        f"  R={cv.get('recall','?')}"
                        f"  F1={cv.get('f1','?')}"
                        f"  (n={cv.get('support','?')})"
                    )
        print()

    print("=" * 70)
