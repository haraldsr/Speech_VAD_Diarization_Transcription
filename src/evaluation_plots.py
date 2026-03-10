from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from typing_extensions import TypeGuard


def _import_plotting() -> tuple[Any, Any]:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        raise ImportError(
            "matplotlib and numpy are required for evaluation plotting. "
            "Install with: pip install matplotlib numpy"
        ) from exc
    return plt, np


def _is_number(value: Any) -> TypeGuard[float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _metric_direction(metric_name: str) -> str:
    lower_tokens = (
        "error",
        "der",
        "jer",
        "wer",
        "cer",
        "mer",
        "wil",
        "mae",
        "distance",
        "dist",  # catches abbreviated display labels such as "Semantic dist"
    )
    name = metric_name.lower()
    if any(token in name for token in lower_tokens):
        return "lower"
    return "higher"


def _metric_bucket(stage: str, metric: str, value: float) -> str:
    """Bucket metric for clearer plotting axes/scales."""
    m = metric.lower()

    # Explicit counts (label-type matched segments, etc.)
    if "matched" in m or "count" in m or m.startswith("n_"):
        return "count"

    # Explicit unbounded error-like metrics (seconds/distances)
    if "mae" in m or "distance" in m:
        return "low_unbounded"

    direction = _metric_direction(metric)
    in_unit_interval = 0.0 <= float(value) <= 1.0

    if direction == "lower":
        return "low_01" if in_unit_interval else "low_unbounded"
    return "high_01" if in_unit_interval else "high_unbounded"


def _fmt_value(metric: str, value: float) -> str:
    m = metric.lower()
    if "matched" in m or "count" in m or m.startswith("n_"):
        return str(int(round(value)))
    return f"{value:.3f}"


def _stage_metrics(results: Mapping[str, Any]) -> Dict[str, List[Tuple[str, float]]]:
    stage_data: Dict[str, List[Tuple[str, float]]] = {}

    vad = results.get("vad")
    if isinstance(vad, Mapping):
        pooled = vad.get("pooled", vad)
        if isinstance(pooled, Mapping):
            metrics = [
                ("DER", pooled.get("detection_error_rate")),
                ("Accuracy", pooled.get("detection_accuracy")),
                ("Precision", pooled.get("precision")),
                ("Recall", pooled.get("recall")),
                ("F1", pooled.get("f1")),
                ("Onset MAE", pooled.get("onset_mae")),
                ("Offset MAE", pooled.get("offset_mae")),
            ]
            stage_data["VAD"] = [(k, float(v)) for k, v in metrics if _is_number(v)]

    diar = results.get("diarization")
    if isinstance(diar, Mapping):
        metrics = [
            ("DER", diar.get("diarization_error_rate")),
            ("JER", diar.get("jaccard_error_rate")),
            ("Purity", diar.get("purity")),
            ("Coverage", diar.get("coverage")),
            ("Homogeneity", diar.get("homogeneity")),
            ("Completeness", diar.get("completeness")),
            ("IER", diar.get("identification_error_rate")),
            ("Speaker Det. Acc", diar.get("speaker_detection_accuracy")),
            ("SID mapped", diar.get("speaker_id_accuracy")),
            ("SID raw", diar.get("speaker_id_accuracy_raw")),
        ]
        stage_data["Diarization"] = [(k, float(v)) for k, v in metrics if _is_number(v)]

    seg = results.get("segmentation")
    if isinstance(seg, Mapping):
        metrics = [
            ("Purity", seg.get("segmentation_purity", seg.get("purity"))),
            ("Coverage", seg.get("segmentation_coverage", seg.get("coverage"))),
            ("Precision", seg.get("segmentation_precision", seg.get("precision"))),
            ("Recall", seg.get("segmentation_recall", seg.get("recall"))),
            ("F-measure", seg.get("segmentation_f_measure", seg.get("f_measure"))),
        ]
        stage_data["Segmentation"] = [
            (k, float(v)) for k, v in metrics if _is_number(v)
        ]

    tr = results.get("transcription")
    if isinstance(tr, Mapping):
        pooled = tr.get("pooled", {}) if isinstance(tr.get("pooled"), Mapping) else {}
        raw = pooled.get("raw", {}) if isinstance(pooled.get("raw"), Mapping) else {}
        norm = (
            pooled.get("normalised", {})
            if isinstance(pooled.get("normalised"), Mapping)
            else {}
        )
        metrics = [
            ("Coverage", pooled.get("coverage")),
            ("WER raw", raw.get("wer")),
            ("CER raw", raw.get("cer")),
            ("MER raw", raw.get("mer")),
            ("WER norm", norm.get("wer")),
            ("CER norm", norm.get("cer")),
            ("MER norm", norm.get("mer")),
            ("Semantic dist", norm.get("semantic_distance")),
        ]
        stage_data["Transcription"] = [
            (k, float(v)) for k, v in metrics if _is_number(v)
        ]

    lt = results.get("label_type")
    if isinstance(lt, Mapping):
        metrics = [
            ("Macro F1", lt.get("macro_f1")),
            ("Macro Precision", lt.get("macro_precision")),
            ("Macro Recall", lt.get("macro_recall")),
            ("Matched", lt.get("n_matched")),
        ]
        stage_data["Label Type"] = [(k, float(v)) for k, v in metrics if _is_number(v)]

    return {k: v for k, v in stage_data.items() if v}


def _plot_kpis(
    stage_data: Mapping[str, Sequence[Tuple[str, float]]],
    out_path: str,
    *,
    dpi: int | None = None,
) -> str:
    plt, np = _import_plotting()

    # Flatten to entries and split by semantics/scale so low-vs-high are separated.
    grouped: Dict[str, List[Tuple[str, str, float]]] = {
        "high_01": [],
        "low_01": [],
        "low_unbounded": [],
        "high_unbounded": [],
    }

    for stage_name, metrics in stage_data.items():
        for metric_name, metric_value in metrics:
            bucket = _metric_bucket(stage_name, metric_name, metric_value)
            # Skip buckets we don't plot (e.g., counts) to avoid KeyError.
            if bucket not in grouped:
                continue
            grouped[bucket].append((stage_name, metric_name, float(metric_value)))

    panel_order = [
        ("high_01", "Higher is better (0-1)"),
        ("low_01", "Lower is better (0-1)"),
        ("low_unbounded", "Lower is better (unbounded)"),
        ("high_unbounded", "Higher is better (unbounded)"),
    ]
    non_empty = [(k, t) for k, t in panel_order if grouped[k]]
    if not non_empty:
        return ""

    # Make figure dimensions proportional to plotted content.
    n_panels = len(non_empty)
    rows_per_panel = [len(grouped[key]) for key, _ in non_empty]
    total_rows = sum(rows_per_panel)
    max_label_len = max(
        (
            len(f"{stage}: {metric}")
            for rows in grouped.values()
            for stage, metric, _ in rows
        ),
        default=24,
    )
    # Height allocation follows panel row counts directly.
    height_ratios = [max(1, r) for r in rows_per_panel]
    # Dynamic sizing (bar-count driven): each plotted row contributes height.
    fig_h = max(8.5, 2.8 + (0.52 * total_rows) + (0.35 * n_panels))
    fig_w = min(22.0, max(14.0, 11.5 + 0.06 * max_label_len))
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(fig_w, fig_h),
        squeeze=False,
        gridspec_kw={"height_ratios": height_ratios},
    )
    axes_flat = axes.flatten()

    stage_colors = {
        "VAD": "#4C78A8",
        "Diarization": "#F58518",
        "Segmentation": "#54A24B",
        "Transcription": "#B279A2",
        "Label Type": "#E45756",
    }

    # Build legend handles for only the stages present in the plotted metrics.
    present_stages = []
    for rows in grouped.values():
        for s, _, _ in rows:
            if s not in present_stages:
                present_stages.append(s)
    Patch = __import__("matplotlib.patches", fromlist=["Patch"]).Patch
    legend_handles = [
        Patch(facecolor=stage_colors.get(s, "#4C78A8"), label=s) for s in present_stages
    ]

    for ax, (bucket, title) in zip(axes_flat, non_empty):
        rows = grouped[bucket]
        labels = [f"{s}: {m}" for s, m, _ in rows]
        values = np.array([v for _, _, v in rows], dtype=float)
        colors = [stage_colors.get(s, "#4C78A8") for s, _, _ in rows]
        y = np.arange(len(labels))

        bars = ax.barh(y, values, color=colors, alpha=0.92)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=12)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.25)
        ax.set_title(title, fontsize=14, weight="bold")
        ax.tick_params(axis="x", labelsize=11)

        max_val = float(np.max(values)) if len(values) else 1.0
        span = max(0.1, max_val * 0.08)
        xlim_max = max_val + span
        ax.set_xlim(0, xlim_max)

        for rect, (_, metric_name, metric_value) in zip(bars, rows):
            x = rect.get_width()
            ax.text(
                x + xlim_max * 0.015,
                rect.get_y() + rect.get_height() / 2,
                _fmt_value(metric_name, metric_value),
                va="center",
                fontsize=13,
                weight="bold",
            )

    fig.suptitle(
        "Evaluation Metrics (split by direction and scale)",
        fontsize=18,
        weight="bold",
        y=0.992,
    )

    # Place legend below title as multi-column map for color groups.
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            # lowered slightly to reduce whitespace between legend and plots
            bbox_to_anchor=(0.5, 0.98),
            ncol=max(2, min(5, len(legend_handles))),
            frameon=False,
            fontsize=14,
        )

    # tighten the reserved top margin so plots sit closer to the legend
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fmt = os.path.splitext(out_path)[1].lower().lstrip(".")
    save_kwargs: Dict[str, Any] = {"bbox_inches": "tight"}
    if fmt == "png" and dpi is not None:
        save_kwargs["dpi"] = dpi
    fig.savefig(out_path, **save_kwargs)
    plt.close(fig)
    return out_path


def _extract_per_speaker_stage_metrics(
    results: Mapping[str, Any],
) -> Dict[str, List[Tuple[str, str, float]]]:
    per_speaker: Dict[str, List[Tuple[str, str, float]]] = {}

    def _add(spk: str, stage: str, metric: str, value: Any) -> None:
        if not _is_number(value):
            return
        per_speaker.setdefault(str(spk), []).append((stage, metric, float(value)))

    vad = results.get("vad")
    if isinstance(vad, Mapping) and isinstance(vad.get("per_speaker"), Mapping):
        for spk, stats in vad["per_speaker"].items():
            if not isinstance(stats, Mapping):
                continue
            _add(str(spk), "VAD", "DER", stats.get("detection_error_rate"))
            _add(str(spk), "VAD", "Accuracy", stats.get("detection_accuracy"))
            _add(str(spk), "VAD", "Precision", stats.get("precision"))
            _add(str(spk), "VAD", "Recall", stats.get("recall"))
            _add(str(spk), "VAD", "F1", stats.get("f1"))
            _add(str(spk), "VAD", "Onset MAE", stats.get("onset_mae"))
            _add(str(spk), "VAD", "Offset MAE", stats.get("offset_mae"))

    diar = results.get("diarization")
    if isinstance(diar, Mapping):
        diar_sid = diar.get("speaker_id_detail")
        if isinstance(diar_sid, Mapping) and isinstance(
            diar_sid.get("per_speaker"), Mapping
        ):
            for spk, stats in diar_sid["per_speaker"].items():
                if isinstance(stats, Mapping):
                    _add(str(spk), "Diarization", "SID mapped", stats.get("accuracy"))
        if isinstance(diar_sid, Mapping) and isinstance(
            diar_sid.get("raw_per_speaker"), Mapping
        ):
            for spk, stats in diar_sid["raw_per_speaker"].items():
                if isinstance(stats, Mapping):
                    _add(str(spk), "Diarization", "SID raw", stats.get("accuracy"))

    tr = results.get("transcription")
    if isinstance(tr, Mapping) and isinstance(tr.get("per_speaker"), Mapping):
        for spk, stats in tr["per_speaker"].items():
            if not isinstance(stats, Mapping):
                continue
            raw = stats.get("raw", {}) if isinstance(stats.get("raw"), Mapping) else {}
            norm = (
                stats.get("normalised", {})
                if isinstance(stats.get("normalised"), Mapping)
                else {}
            )
            _add(str(spk), "Transcription", "WER raw", raw.get("wer"))
            _add(str(spk), "Transcription", "CER raw", raw.get("cer"))
            _add(str(spk), "Transcription", "MER raw", raw.get("mer"))
            _add(str(spk), "Transcription", "WER norm", norm.get("wer"))
            _add(str(spk), "Transcription", "CER norm", norm.get("cer"))
            _add(str(spk), "Transcription", "MER norm", norm.get("mer"))
            _add(
                str(spk),
                "Transcription",
                "Semantic dist",
                norm.get("semantic_distance"),
            )

    return {k: v for k, v in per_speaker.items() if v}


def _plot_kpis_per_speaker(
    speaker_metrics: Mapping[str, Sequence[Tuple[str, str, float]]],
    aggregate_stage_data: Mapping[str, Sequence[Tuple[str, float]]] | None,
    out_path: str,
    *,
    dpi: int | None = None,
) -> str:
    if not speaker_metrics:
        return ""

    plt, np = _import_plotting()
    Patch = __import__("matplotlib.patches", fromlist=["Patch"]).Patch

    stage_colors = {
        "VAD": "#4C78A8",
        "Diarization": "#F58518",
        "Segmentation": "#54A24B",
        "Transcription": "#B279A2",
        "Label Type": "#E45756",
    }

    speakers = sorted([str(s) for s in speaker_metrics.keys()])
    hatch_cycle = ["///", "\\\\", "xx", "..", "++", "oo", "--", "**"]
    speaker_hatches = {
        spk: hatch_cycle[i % len(hatch_cycle)] for i, spk in enumerate(speakers)
    }

    # No aggregate baseline: per-speaker plot focuses on individual speakers only
    metric_groups: Dict[Tuple[str, str], Dict[str, float]] = {}
    for spk in speakers:
        for stage_name, metric_name, metric_value in speaker_metrics[spk]:
            key = (str(stage_name), str(metric_name))
            metric_groups.setdefault(key, {})[spk] = float(metric_value)

    grouped: Dict[str, List[Tuple[str, str, Dict[str, float]]]] = {
        "high_01": [],
        "low_01": [],
        "low_unbounded": [],
        "high_unbounded": [],
    }

    for (stage_name, metric_name), entity_vals in metric_groups.items():
        if not entity_vals:
            continue
        bucket_value = next(iter(entity_vals.values()))
        bucket = _metric_bucket(stage_name, metric_name, float(bucket_value))
        if bucket not in grouped:
            continue
        grouped[bucket].append((stage_name, metric_name, entity_vals))

    panel_order = [
        ("high_01", "Per-speaker: higher is better (0-1)"),
        ("low_01", "Per-speaker: lower is better (0-1)"),
        ("low_unbounded", "Per-speaker: lower is better (unbounded)"),
        ("high_unbounded", "Per-speaker: higher is better (unbounded)"),
    ]
    non_empty = [(k, t) for k, t in panel_order if grouped[k]]
    if not non_empty:
        return ""

    n_panels = len(non_empty)
    rows_per_panel = [len(grouped[key]) for key, _ in non_empty]
    total_rows = sum(rows_per_panel)
    max_label_len = max(
        (
            len(f"{stage}: {metric}")
            for rows in grouped.values()
            for stage, metric, _ in rows
        ),
        default=28,
    )
    entity_count = max(1, len(speakers))

    # Estimate actual plotted bars (row x speaker presence), not only metric rows.
    total_bars = 0
    for bucket, _ in non_empty:
        for _, _, entity_vals in grouped[bucket]:
            total_bars += sum(1 for spk in speakers if spk in entity_vals)

    # Panel heights track row counts directly.
    height_ratios = [max(1, r) for r in rows_per_panel]
    # Dynamic sizing:
    # - proportional to metric rows and actual bars
    # - slightly boosted for many speakers to keep labels/hatches readable
    fig_h = max(
        9.5, 3.2 + (0.50 * total_rows) + (0.035 * total_bars) + (0.18 * entity_count)
    )
    fig_w = min(24.0, max(16.0, 13.0 + 0.06 * max_label_len + 0.35 * entity_count))
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(fig_w, fig_h),
        squeeze=False,
        gridspec_kw={"height_ratios": height_ratios},
    )
    axes_flat = axes.flatten()

    for ax, (bucket, title) in zip(axes_flat, non_empty):
        rows = sorted(grouped[bucket], key=lambda x: (x[0], x[1]))
        labels = [f"{stage}: {metric}" for stage, metric, _ in rows]
        entity_order = speakers
        active_entities = [
            e
            for e in entity_order
            if any(e in entity_vals for _, _, entity_vals in rows)
        ]
        n_entities = max(1, len(active_entities))
        band = 0.82
        bar_h = max(0.10, min(0.34, band / n_entities))
        y_base = np.arange(len(labels), dtype=float)

        max_val = 1.0
        for entity_idx, entity in enumerate(active_entities):
            offset = (entity_idx - (n_entities - 1) / 2.0) * bar_h
            y = y_base + offset
            values = np.array(
                [entity_vals.get(entity, np.nan) for _, _, entity_vals in rows],
                dtype=float,
            )
            valid_idx = np.where(np.isfinite(values))[0]
            if valid_idx.size == 0:
                continue

            y_valid = y[valid_idx]
            val_valid = values[valid_idx]
            stage_valid = [rows[i][0] for i in valid_idx]

            if entity == "All":
                colors = ["#8E8E8E"] * len(valid_idx)
                alpha = 0.55
                hatch = None
            else:
                colors = [stage_colors.get(stage, "#4C78A8") for stage in stage_valid]
                alpha = 0.90
                hatch = speaker_hatches.get(entity, "")

            bars = ax.barh(
                y_valid,
                val_valid,
                height=bar_h * 0.95,
                color=colors,
                alpha=alpha,
                edgecolor="black" if entity == "All" else None,
                linewidth=0.6 if entity == "All" else 0.0,
            )
            if hatch:
                for rect in bars:
                    rect.set_hatch(hatch)

            max_val = max(max_val, float(np.nanmax(val_valid)))

            for rect, (_, metric_name, _) in zip(bars, [rows[i] for i in valid_idx]):
                x = rect.get_width()
                span_local = max(0.2, max_val * 0.15)
                ax.text(
                    x + span_local * 0.02,
                    rect.get_y() + rect.get_height() / 2,
                    _fmt_value(metric_name, float(x)),
                    va="center",
                    fontsize=10,
                    weight="bold",
                    alpha=0.95 if entity != "All" else 0.75,
                )

        ax.set_yticks(y_base)
        ax.set_yticklabels(labels, fontsize=11)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.25)
        ax.set_title(title, fontsize=14, weight="bold")
        ax.tick_params(axis="x", labelsize=11)
        span = max(0.2, max_val * 0.15)
        span = max(0.1, max_val * 0.08)
        ax.set_xlim(0, max_val + span)

    stage_handles = [
        Patch(facecolor=stage_colors[s], label=s)
        for s in ["VAD", "Diarization", "Transcription", "Segmentation", "Label Type"]
        if any(stage == s for vals in speaker_metrics.values() for stage, _, _ in vals)
    ]
    speaker_handles = [
        Patch(
            facecolor="white", edgecolor="black", hatch=speaker_hatches[spk], label=spk
        )
        for spk in speakers
    ]

    fig.suptitle(
        "Per-Speaker Evaluation Metrics (color=stage, hatch=speaker)",
        fontsize=18,
        weight="bold",
        y=0.992,
    )
    handles = stage_handles + speaker_handles
    if handles:
        fig.legend(
            handles=handles,
            loc="upper center",
            # lowered slightly to sit closer to plots like the combined KPI
            bbox_to_anchor=(0.5, 0.97),
            ncol=max(2, min(6, len(handles))),
            frameon=False,
            fontsize=14,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fmt = os.path.splitext(out_path)[1].lower().lstrip(".")
    save_kwargs: Dict[str, Any] = {"bbox_inches": "tight"}
    if fmt == "png" and dpi is not None:
        save_kwargs["dpi"] = dpi
    fig.savefig(out_path, **save_kwargs)
    plt.close(fig)
    return out_path


def plot_evaluation_results(
    results: Mapping[str, Any],
    output_dir: str,
    *,
    file_prefix: str = "evaluation",
    image_format: str = "pdf",
    dpi: int | None = None,
) -> Dict[str, str]:
    """Create single-run evaluation plots from an evaluation results dictionary.

    Returns a dictionary with generated file paths.
    """
    if image_format not in {"png", "pdf", "svg"}:
        raise ValueError("image_format must be one of: png, pdf, svg")

    stage_data = _stage_metrics(results)

    outputs: Dict[str, str] = {}
    if stage_data:
        kpi_path = os.path.join(output_dir, f"{file_prefix}_kpi.{image_format}")
        outputs["kpi"] = _plot_kpis(stage_data, kpi_path, dpi=dpi)

    per_speaker_data = _extract_per_speaker_stage_metrics(results)
    if per_speaker_data:
        per_speaker_path = os.path.join(
            output_dir,
            f"{file_prefix}_kpi_per_speaker.{image_format}",
        )
        plotted = _plot_kpis_per_speaker(
            per_speaker_data,
            stage_data,
            per_speaker_path,
            dpi=dpi,
        )
        if plotted:
            outputs["kpi_per_speaker"] = plotted

    return outputs


def plot_evaluation_json(
    json_path: str,
    output_dir: str | None = None,
    *,
    file_prefix: str = "evaluation",
    image_format: str = "pdf",
    dpi: int | None = None,
) -> Dict[str, str]:
    """Create single-run evaluation plots from a JSON file path."""
    with open(json_path, "r", encoding="utf-8") as fh:
        results = json.load(fh)

    if not isinstance(results, Mapping):
        raise ValueError("Expected a single-run evaluation JSON object (not a list).")

    out_dir = output_dir or os.path.dirname(json_path) or "."
    return plot_evaluation_results(
        results,
        out_dir,
        file_prefix=file_prefix,
        image_format=image_format,
        dpi=dpi,
    )
