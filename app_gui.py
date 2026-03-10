"""
Local GUI for the Speech VAD / Diarization / Transcription pipeline.

Wraps :func:`src.conversation.process_conversation` in a Streamlit form so
the pipeline can be run without writing any code.

Launch with:
    streamlit run app_gui.py
"""

from __future__ import annotations

import ctypes
import inspect
import os
import re
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional

import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
# Ensure the workspace root (and therefore the `src` package) is importable
# regardless of where `streamlit run` is invoked from.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Load .env so HF_TOKEN and other secrets are available as environment variables.
try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

from src.audio_preprocessing import (  # noqa: E402
    _PROFILES,
    PreprocessConfig,
    analyse_audio,
    auto_profile,
    load_audio,
)
from src.conversation import continue_conversation, process_conversation  # noqa: E402


# ── Docstring-based help text ─────────────────────────────────────────────────
def _parse_param_help(func: Any) -> dict[str, str]:
    """Extract per-parameter descriptions from a Google-style Args: docstring."""
    doc = inspect.getdoc(func) or ""
    m = re.search(r"Args:\s*\n(.*?)(?=\nReturns:|\nRaises:|\Z)", doc, re.DOTALL)
    if not m:
        return {}
    section = m.group(1)
    result: dict[str, str] = {}
    for pm in re.finditer(
        r"^ {4,8}(\w+): (.+?)(?=\n {4,8}\w+:|\Z)", section, re.DOTALL | re.MULTILINE
    ):
        result[pm.group(1)] = " ".join(pm.group(2).split())
    return result


_HELP = _parse_param_help(process_conversation)

# ── Temporary upload directory ────────────────────────────────────────────────
_UPLOAD_DIR = Path(tempfile.gettempdir()) / "speech_pipeline_uploads"
_UPLOAD_DIR.mkdir(exist_ok=True)


# ── Thread-safe stdout capture ────────────────────────────────────────────────
class _LineCapture:
    """Writable object that appends non-empty stripped lines to a list."""

    def __init__(self, target: list[str]) -> None:
        self._target = target
        self._lock = threading.Lock()
        self._buf = ""

    def write(self, text: str) -> int:
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line:
                with self._lock:
                    self._target.append(line)
        return len(text)

    def flush(self) -> None:
        if self._buf:
            with self._lock:
                self._target.append(self._buf)
            self._buf = ""


def _pipeline_worker(kwargs: dict, shared: dict) -> None:
    """Background thread: run process_conversation, capture stdout."""
    capture = _LineCapture(shared["log_lines"])
    old_stdout = sys.stdout
    sys.stdout = capture
    try:
        result = process_conversation(**kwargs)
        shared["result"] = result
        shared["error"] = None
    except Exception as exc:
        shared["result"] = None
        shared["error"] = str(exc)
    finally:
        capture.flush()
        sys.stdout = old_stdout
        shared["done"] = True


def _continue_worker(kwargs: dict, shared: dict) -> None:
    """Background thread: run continue_conversation, capture stdout."""
    capture = _LineCapture(shared["log_lines"])
    old_stdout = sys.stdout
    sys.stdout = capture
    try:
        result = continue_conversation(**kwargs)
        shared["result"] = result
        shared["error"] = None
    except Exception as exc:
        shared["result"] = None
        shared["error"] = str(exc)
    finally:
        capture.flush()
        sys.stdout = old_stdout
        shared["done"] = True


# ── Helpers ───────────────────────────────────────────────────────────────────
def _save_upload(uploaded_file: Any, prefix: str = "") -> str:
    """Save an UploadedFile to a temp path and return the path string."""
    dest = _UPLOAD_DIR / f"{prefix}{uploaded_file.name}"
    dest.write_bytes(uploaded_file.getbuffer())
    return str(dest)


def _cancel_pipeline() -> None:
    """Raise SystemExit in the background pipeline thread to interrupt it."""
    thread: Optional[threading.Thread] = st.session_state.get("_thread")
    if thread and thread.is_alive():
        thread_id = thread.ident or 0
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(thread_id),
            ctypes.py_object(SystemExit),
        )
    st.session_state.running = False
    st.session_state.done = True
    st.session_state.error = "Pipeline cancelled by user."


def _reset_state() -> None:
    for key in (
        "running",
        "done",
        "log_lines",
        "result",
        "error",
        "_thread",
        "_shared",
        "output_dir_input",
        "_output_dir_suggestion",
    ):
        st.session_state.pop(key, None)


def _compute_output_dir_suggestion(
    mode: str, speaker_configs: list, single_file: Any
) -> str:
    """Return a suggested output directory based on uploaded audio file names."""
    subdir = "vad" if mode == "Per-speaker separate files (VAD mode)" else "diarization"
    if mode == "Per-speaker separate files (VAD mode)":
        raw_stems = [Path(f.name).stem for _, f in speaker_configs if f is not None]
        # Strip common speaker prefixes/suffixes: P1_, P2_, _P1, _P2, etc.
        clean_stems = [
            re.sub(r"(^[Pp]\d+[_\-]|[_\-][Pp]\d+$)", "", s) for s in raw_stems
        ]
        if clean_stems:
            common = os.path.commonprefix(clean_stems).rstrip("_-")
            name = common if len(common) >= 3 else clean_stems[0]
        else:
            name = "run"
    else:
        name = Path(single_file.name).stem if single_file is not None else "run"
    return f"outputs/{subdir}/{name}"


def _render_preprocessing_profile(
    key_prefix: str,
    auto_config: Any,
    default_config: Any,
    title: str = "",
) -> tuple[bool, Any]:
    """Render a preprocessing profile picker and return (enabled, PreprocessConfig).

    key_prefix: unique widget-key namespace (e.g. "" for single, "mild_" / "strong_").
    title: optional sub-header shown above the profile radio.
    """
    if title:
        st.markdown(f"**{title}**")

    profile_options = {
        "Auto-detect": "auto",
        "Clean": "clean",
        "Moderate": "moderate",
        "Noisy": "noisy",
        "Manual": "manual",
    }
    selected_profile = st.radio(
        "Select preprocessing profile",
        list(profile_options.keys()),
        horizontal=True,
        key=f"{key_prefix}profile",
        help=(
            "**Auto-detect**: Analyzes LUFS and SNR to choose optimal settings. "
            "**Clean**: Minimal processing (HPF only). Assumed SNR > 25 dB. "
            "**Moderate**: Standard processing for typical recordings. "
            "**Noisy**: Aggressive enhancement for poor recordings with SNR < 18 dB. "
            "**Manual**: Full control over each parameter."
        ),
    )
    profile_key = profile_options[selected_profile]

    if profile_key == "auto":
        cfg = auto_config if auto_config else default_config
        enabled = st.checkbox(
            "Enable preprocessing",
            value=True,
            key=f"{key_prefix}enable",
            help="Apply the auto-detected preprocessing pipeline.",
        )
    elif profile_key == "manual":
        enabled = st.checkbox(
            "Enable preprocessing",
            value=True,
            key=f"{key_prefix}enable",
            help="Apply custom preprocessing pipeline.",
        )
        hp_col, nr_col, ln_col, pl_col = st.columns(4)

        with hp_col:
            with st.expander("High-Pass Filter", expanded=False):
                hp_enabled = st.checkbox(
                    "Enable",
                    value=auto_config.highpass,
                    key=f"{key_prefix}hp_enable",
                    help="Remove DC offset and low-frequency rumble.",
                )
                hp_freq = st.slider(
                    "Cutoff (Hz)",
                    20,
                    200,
                    int(auto_config.highpass_freq),
                    key=f"{key_prefix}hp_freq",
                    help="Frequency below which to remove content.",
                )

        with nr_col:
            with st.expander("Noise Reduction", expanded=False):
                nr_enabled = st.checkbox(
                    "Enable",
                    value=auto_config.noise_reduce,
                    key=f"{key_prefix}nr_enable",
                    help="Reduce stationary background noise via spectral gating.",
                )
                nr_stationary = st.checkbox(
                    "Stationary noise",
                    value=auto_config.noise_reduce_stationary,
                    key=f"{key_prefix}nr_stat",
                    help="Assume noise is stationary (faster; uncheck for adaptive).",
                )
                nr_prop = st.slider(
                    "Reduction strength",
                    0.0,
                    1.0,
                    float(auto_config.noise_reduce_prop_decrease),
                    0.05,
                    key=f"{key_prefix}nr_prop",
                    help="0.0 = none, 1.0 = aggressive.",
                )

        with ln_col:
            with st.expander("Loudness Norm.", expanded=False):
                ln_enabled = st.checkbox(
                    "Enable",
                    value=auto_config.loudness_norm,
                    key=f"{key_prefix}ln_enable",
                    help="Normalise to target LUFS (EBU R 128).",
                )
                ln_target = st.slider(
                    "Target (LUFS)",
                    -40,
                    -10,
                    int(auto_config.target_lufs),
                    key=f"{key_prefix}ln_target",
                )
                ln_auto = st.checkbox(
                    "Auto (skip if close)",
                    value=auto_config.auto_loudness,
                    key=f"{key_prefix}ln_auto",
                    help="Skip normalisation if already within tolerance.",
                )
                ln_tol = st.slider(
                    "Tolerance (dB)",
                    0.5,
                    10.0,
                    float(auto_config.loudness_tolerance_db),
                    0.5,
                    key=f"{key_prefix}ln_tol",
                )

        with pl_col:
            with st.expander("Peak Limiter", expanded=False):
                pl_enabled = st.checkbox(
                    "Enable",
                    value=auto_config.peak_limit,
                    key=f"{key_prefix}pl_enable",
                    help="Hard-clip peaks to prevent digital clipping.",
                )
                pl_ceiling = st.slider(
                    "Ceiling",
                    0.5,
                    0.99,
                    float(auto_config.peak_ceiling),
                    0.01,
                    key=f"{key_prefix}pl_ceiling",
                )

        cfg = PreprocessConfig(
            enabled=enabled,
            highpass=hp_enabled,
            highpass_freq=float(hp_freq),
            noise_reduce=nr_enabled,
            noise_reduce_stationary=nr_stationary,
            noise_reduce_prop_decrease=nr_prop,
            loudness_norm=ln_enabled,
            target_lufs=float(ln_target),
            auto_loudness=ln_auto,
            loudness_tolerance_db=ln_tol,
            peak_limit=pl_enabled,
            peak_ceiling=pl_ceiling,
        )
    else:
        # Preset profile (clean, moderate, noisy)
        enabled = st.checkbox(
            "Enable preprocessing",
            value=True,
            key=f"{key_prefix}enable",
            help=f"Apply the '{selected_profile}' preprocessing preset.",
        )
        preset_dict = _PROFILES[profile_key]
        cfg = PreprocessConfig(enabled=enabled, **preset_dict)

    return enabled, cfg


# ── Main app ──────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="Speech Pipeline",
        page_icon="🗣️",
        layout="wide",
    )
    st.title("🗣️ Speech VAD / Diarization / Transcription")
    st.caption(
        "Configure the pipeline parameters below, upload your audio, then click **Run**."
    )

    # ── Session-state initialisation ──────────────────────────────────────────
    for key, default in {
        "running": False,
        "done": False,
        "log_lines": [],
        "result": None,
        "error": None,
        "_thread": None,
        "_shared": None,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # =========================================================================
    # PIPELINE MODE
    # =========================================================================
    pipeline_mode = st.radio(
        "Pipeline mode",
        ["Full pipeline", "Continue existing file"],
        horizontal=True,
        help=(
            "**Full pipeline** — run VAD → transcription → labeling from scratch. "
            "**Continue existing file** — read an existing labels file (.txt or .ass), "
            "detect what is missing (transcription / turn labels), and add only those "
            "steps while preserving whatever is already present."
        ),
    )

    st.header("Audio Input")

    # ── Continue mode: path to existing file ─────────────────────────────────
    existing_path: str | None = None
    if pipeline_mode == "Continue existing file":
        _label_input_mode = st.radio(
            "Labels file input",
            ["Upload file", "Enter path"],
            horizontal=True,
            help="Upload the existing labels file directly, or type its path on disk.",
        )
        if _label_input_mode == "Upload file":
            _uploaded_labels = st.file_uploader(
                "Existing labels file",
                type=["txt", "ass"],
                help="Upload a tab-separated .txt or Aegisub .ass annotation file.",
            )
            if _uploaded_labels is not None:
                import tempfile

                _labels_tmp_dir = (
                    Path(tempfile.gettempdir()) / "speech_pipeline_uploads"
                )
                _labels_tmp_dir.mkdir(parents=True, exist_ok=True)
                _labels_tmp_path = str(_labels_tmp_dir / _uploaded_labels.name)
                with open(_labels_tmp_path, "wb") as _lf:
                    _lf.write(_uploaded_labels.getbuffer())
                existing_path = _labels_tmp_path
        else:
            existing_path_raw = st.text_input(
                "Existing labels file path",
                placeholder="Path to existing labels file",
                help=(
                    "Path to an existing annotation file. "
                    "Supported: tab-separated .txt (our pipeline output, with or without a header) "
                    "and Aegisub .ass subtitle files."
                ),
            )
            existing_path = existing_path_raw.strip() or None
            if existing_path and not os.path.exists(existing_path):
                st.warning(f"File not found: `{existing_path}`")
                existing_path = None

    # ── Audio input (shared between both modes) ───────────────────────────────
    if pipeline_mode == "Full pipeline":
        mode = st.radio(
            "Input mode",
            [
                "Per-speaker separate files (VAD mode)",
                "Single file – automatic speaker diarization",
            ],
            horizontal=True,
        )
    else:
        mode = "Per-speaker separate files (VAD mode)"
        st.caption(
            "Map each speaker ID (exactly as it appears in the existing file) to its audio "
            "file. For a diarized recording where both speakers share one file, map every "
            "speaker to the same audio path."
        )

    speakers_audio: dict[str, str] | str | None = None

    if mode == "Per-speaker separate files (VAD mode)":
        n_speakers = int(
            st.number_input(
                "Number of speakers", min_value=1, max_value=8, value=2, step=1
            )
        )
        cols = st.columns(n_speakers)
        speaker_configs: list[tuple[str, Any]] = []
        for i, col in enumerate(cols):
            with col:
                name = st.text_input(
                    f"Speaker {i + 1} name", value=f"P{i + 1}", key=f"spk_name_{i}"
                )
                audio_file = st.file_uploader(
                    f"Audio for {name}",
                    type=["wav", "mp3", "flac", "m4a"],
                    key=f"spk_audio_{i}",
                )
                # Show duration and size if file is uploaded
                if audio_file is not None:
                    try:
                        _tmp_path = _save_upload(audio_file, prefix=f"tmp_{i}_")
                        _tmp_audio, _tmp_sr = load_audio(_tmp_path)
                        _tmp_stats = analyse_audio(_tmp_audio, _tmp_sr)
                        _tmp_duration_sec = _tmp_stats.get("duration_sec", 0)
                        _tmp_hours = int(_tmp_duration_sec // 3600)
                        _tmp_minutes = int((_tmp_duration_sec % 3600) // 60)
                        _tmp_seconds = int(_tmp_duration_sec % 60)
                        _tmp_duration_str = (
                            f"{_tmp_hours:02d}:{_tmp_minutes:02d}:{_tmp_seconds:02d}"
                        )
                        _tmp_size_bytes = audio_file.size
                        if _tmp_size_bytes >= 1_000_000:
                            _tmp_size_str = f"{_tmp_size_bytes / 1_000_000:.1f} MB"
                        elif _tmp_size_bytes >= 1_000:
                            _tmp_size_str = f"{_tmp_size_bytes / 1_000:.1f} KB"
                        else:
                            _tmp_size_str = f"{_tmp_size_bytes} B"
                        st.caption(
                            f"Duration: {_tmp_duration_str} | Size: {_tmp_size_str}"
                        )
                    except Exception:
                        st.caption(
                            f"Size: {audio_file.size / 1_000_000:.1f} MB (duration unavailable)"
                        )
                speaker_configs.append((name, audio_file))

        if all(f is not None for _, f in speaker_configs):
            speakers_audio = {
                name: _save_upload(f, prefix=f"{name}_") for name, f in speaker_configs
            }
        else:
            st.info("Upload an audio file for every speaker to enable the Run button.")

    else:
        single_file = st.file_uploader(
            "Audio file",
            type=["wav", "mp3", "flac", "m4a"],
        )
        if single_file is not None:
            # Show duration and size
            try:
                _tmp_path = _save_upload(single_file, prefix="tmp_single_")
                _tmp_audio, _tmp_sr = load_audio(_tmp_path)
                _tmp_stats = analyse_audio(_tmp_audio, _tmp_sr)
                _tmp_duration_sec = _tmp_stats.get("duration_sec", 0)
                _tmp_hours = int(_tmp_duration_sec // 3600)
                _tmp_minutes = int((_tmp_duration_sec % 3600) // 60)
                _tmp_seconds = int(_tmp_duration_sec % 60)
                _tmp_duration_str = (
                    f"{_tmp_hours:02d}:{_tmp_minutes:02d}:{_tmp_seconds:02d}"
                )
                _tmp_size_bytes = single_file.size
                if _tmp_size_bytes >= 1_000_000:
                    _tmp_size_str = f"{_tmp_size_bytes / 1_000_000:.1f} MB"
                elif _tmp_size_bytes >= 1_000:
                    _tmp_size_str = f"{_tmp_size_bytes / 1_000:.1f} KB"
                else:
                    _tmp_size_str = f"{_tmp_size_bytes} B"
                st.caption(f"Duration: {_tmp_duration_str} | Size: {_tmp_size_str}")
            except Exception:
                st.caption(
                    f"Size: {single_file.size / 1_000_000:.1f} MB (duration unavailable)"
                )
            speakers_audio = _save_upload(single_file)
        else:
            st.info("Upload an audio file to enable the Run button.")

    # Dynamic output directory — follows uploaded filenames unless user has customised it
    _spk_cfgs = (
        speaker_configs if mode == "Per-speaker separate files (VAD mode)" else []
    )
    _sngl = (
        single_file if mode == "Single file – automatic speaker diarization" else None
    )
    _suggestion = _compute_output_dir_suggestion(mode, _spk_cfgs, _sngl)
    _prev_suggestion = st.session_state.get("_output_dir_suggestion", "outputs/gui_run")
    _current_out = st.session_state.get("output_dir_input", _prev_suggestion)
    if _current_out == _prev_suggestion:
        # User hasn't customised the field — follow new suggestion
        st.session_state["output_dir_input"] = _suggestion
    st.session_state["_output_dir_suggestion"] = _suggestion
    output_dir = st.text_input("Output directory", key="output_dir_input")

    # =========================================================================
    # VAD SETTINGS (mode-specific)
    # =========================================================================
    _vad_hidden = pipeline_mode == "Continue existing file"
    with st.expander("VAD Settings", expanded=not _vad_hidden):
        if _vad_hidden:
            st.caption(
                "VAD is skipped in *Continue* mode — segments come from the existing file."
            )
            vad_type = "rvad"
            rvad_threshold = 0.4
            vad_min_duration = 0.07
            energy_margin_db = 20.0
            auth_token = None
        else:
            # Different VAD options based on input mode
            if mode == "Per-speaker separate files (VAD mode)":
                st.caption(
                    "Voice Activity Detection (VAD) type options for per-speaker processing:"
                )
                available_vad_types = ["rvad", "silero", "whisper", "pyannote", "nemo"]
                vad_label = "VAD type"
            else:
                st.caption(
                    "Speaker diarization model options for single-file automatic speaker identification:"
                )
                available_vad_types = ["pyannote", "nemo"]
                vad_label = "Diarization model"

            c1, c2, c3, c4 = st.columns(4)
            vad_type = c1.selectbox(
                vad_label,
                available_vad_types,
                help=_HELP.get("vad_type"),
            )
            rvad_threshold = c2.slider(
                "rVAD threshold",
                0.0,
                1.0,
                0.4,
                0.01,
                disabled=(vad_type != "rvad"),
                help="Only applies when VAD type is rvad.",
            )
            vad_min_duration = c3.number_input(
                "Min segment duration (s)",
                0.01,
                5.0,
                0.07,
                format="%.3f",
                help=(
                    "Minimum speech segment duration (applies to both VAD "
                    "and diarization models). Segments shorter than this "
                    "are filtered out. " + _HELP.get("vad_min_duration", "")
                ),
            )
            energy_margin_db = c4.number_input(
                "Energy margin (dB)",
                0.0,
                100.0,
                20.0,
                1.0,
                format="%.1f",
                help=(
                    "Margin in dB below the loudest segment to filter low-energy regions. "
                    "Default 10 dB filters out segments ~68% quieter than the loudest (typical for speech). "
                    "Use 20 dB for aggressive filtering of background noise in noisy recordings; "
                    "use 5 dB for sensitive recording where whispers/backchannels matter."
                ),
            )
            # Only pyannote requires a HuggingFace token; nemo downloads models independently.
            if vad_type == "pyannote":
                st.subheader("HuggingFace Authentication")

                # Check all possible token sources
                _env_token = os.environ.get("HF_TOKEN", "").strip()
                _cached_token_path = Path.home() / ".cache" / "huggingface" / "token"
                _has_cached_token = (
                    _cached_token_path.exists()
                    and _cached_token_path.read_text().strip()
                )

                # Display current authentication status
                auth_col1, auth_col2, auth_col3 = st.columns(3)

                with auth_col1:
                    if _env_token:
                        st.success("✅ HF_TOKEN env var set")
                    else:
                        st.info("⚠ HF_TOKEN env var not set")

                with auth_col2:
                    if _has_cached_token:
                        st.success("✅ Cached login (huggingface-cli)")
                    else:
                        st.info("⚠ No cached login")

                with auth_col3:
                    st.write("")  # spacer

                # Let user select which auth method to use
                auth_method = st.radio(
                    "Authentication method",
                    [
                        "Auto-detect (priority: env → cache → manual)",
                        "Use HF_TOKEN environment variable",
                        "Use cached login (huggingface-cli login)",
                        "Enter token manually",
                    ],
                    horizontal=False,
                    help=(
                        "**Auto-detect**: Uses HF_TOKEN if set, else cached "
                        "login if available, else prompts for manual entry. "
                        "**HF_TOKEN**: Use value from HF_TOKEN environment "
                        "variable. **Cached login**: Use credentials from "
                        "`huggingface-cli login`. **Manual**: Enter token "
                        "directly in text field."
                    ),
                )

                hf_auth_token: Optional[str] = None

                if auth_method == "Auto-detect (priority: env → cache → manual)":
                    if _env_token:
                        hf_auth_token = _env_token
                        st.info("Using HF_TOKEN from environment variable")
                    elif _has_cached_token:
                        hf_auth_token = _cached_token_path.read_text().strip()
                        st.info("Using cached login from huggingface-cli")
                    else:
                        st.warning(
                            "No automatic authentication found. "
                            "Please enter token manually or run "
                            "`huggingface-cli login`"
                        )
                        _manual_token = st.text_input(
                            "Enter HuggingFace token", type="password"
                        )
                        hf_auth_token = _manual_token.strip() or None

                elif auth_method == "Use HF_TOKEN environment variable":
                    if _env_token:
                        hf_auth_token = _env_token
                        st.success("Using HF_TOKEN from environment")
                    else:
                        st.error(
                            "HF_TOKEN environment variable not set. Set it in .env or system environment."
                        )

                elif auth_method == "Use cached login (huggingface-cli login)":
                    if _has_cached_token:
                        hf_auth_token = _cached_token_path.read_text().strip()
                        st.success(
                            "Using cached credentials from huggingface-cli login"
                        )
                    else:
                        st.error(
                            "No cached login found. Run `huggingface-cli login` first."
                        )

                elif auth_method == "Enter token manually":
                    _manual_token = st.text_input(
                        "Enter HuggingFace token", type="password"
                    )
                    hf_auth_token = _manual_token.strip() or None
            else:
                hf_auth_token = None

            auth_token = hf_auth_token

    # =========================================================================
    # TURN MERGING
    # =========================================================================
    with st.expander("Turn Merging", expanded=(pipeline_mode == "Full pipeline")):
        if pipeline_mode == "Continue existing file":
            st.caption(
                "Turn merging is skipped in *Continue* mode — segments come from the existing file."
            )
        _tm_dis = pipeline_mode == "Continue existing file"
        c1, c2, c3 = st.columns(3)
        gap_thresh = c1.slider(
            "Gap threshold (s)",
            0.0,
            5.0,
            0.5,
            0.05,
            disabled=_tm_dis,
            help=_HELP.get("gap_thresh"),
        )
        short_utt_thresh = c2.slider(
            "Short utterance threshold (s)",
            0.1,
            5.0,
            1.0,
            0.1,
            disabled=_tm_dis,
            help=_HELP.get("short_utt_thresh"),
        )
        window_sec = c3.slider(
            "Merge look-ahead window (s)",
            0.5,
            10.0,
            3.0,
            0.5,
            disabled=_tm_dis,
            help=_HELP.get("window_sec"),
        )
        c4, c5, c6, c7 = st.columns(4)
        merge_short_after_long = c4.checkbox(
            "Merge short after long",
            value=True,
            disabled=_tm_dis,
            help=_HELP.get("merge_short_after_long"),
        )
        merge_long_after_short = c5.checkbox(
            "Merge long after short",
            value=True,
            disabled=_tm_dis,
            help=_HELP.get("merge_long_after_short"),
        )
        long_merge_enabled = c6.checkbox(
            "Long + long merge",
            value=True,
            disabled=_tm_dis,
            help=_HELP.get("long_merge_enabled"),
        )
        bridge_short_opponent = c7.checkbox(
            "Bridge short opponent",
            value=True,
            disabled=_tm_dis,
            help=_HELP.get("bridge_short_opponent"),
        )
        merge_max_dur = st.slider(
            "Max merged turn duration (s)",
            10.0,
            300.0,
            60.0,
            5.0,
            disabled=_tm_dis,
            help=_HELP.get("merge_max_dur"),
        )

    # =========================================================================
    # TRANSCRIPTION
    # =========================================================================
    with st.expander("Transcription", expanded=True):
        whisper_models = [
            "large-v3",
            "large-v2",
            "large",
            "medium",
            "small",
            "base",
            "tiny",
        ]
        c1, c2, c3 = st.columns(3)
        transcription_model_name = c1.selectbox("Whisper model", whisper_models)
        whisper_device = c2.selectbox("Device", ["auto", "cpu", "cuda"])
        whisper_language = c3.text_input("Language code", value="da")

        c4, c5, c6 = st.columns(3)
        whisper_model_batch_size = int(
            c4.number_input(
                "Model batch size",
                1,
                512,
                100,
                step=1,
                help=_HELP.get("whisper_model_batch_size"),
            )
        )
        transcription_padding_sec = c5.slider(
            "Segment padding (s)",
            0.0,
            2.0,
            0.2,
            0.05,
            help=_HELP.get("transcription_padding_sec"),
        )
        entropy_threshold = c6.slider(
            "Entropy threshold",
            0.0,
            5.0,
            1.5,
            0.1,
            help=_HELP.get("entropy_threshold"),
        )

        c7, c8 = st.columns(2)
        max_backchannel_dur = c7.slider(
            "Max backchannel duration (s)",
            0.1,
            5.0,
            1.0,
            0.1,
            help=_HELP.get("max_backchannel_dur"),
        )
        max_gap_sec = c8.slider(
            "Max context-merge gap (s)",
            0.0,
            10.0,
            3.0,
            0.5,
            help=_HELP.get("max_gap_sec"),
        )

        batch_size_raw = st.number_input(
            "Batch size (s) — set to 0 to disable batching",
            0.0,
            300.0,
            30.0,
            5.0,
            help=_HELP.get("batch_size"),
        )
        batch_size: float | None = float(batch_size_raw) if batch_size_raw > 0 else None
        min_duration_samples = int(
            st.number_input(
                "Min segment duration (samples)",
                100,
                100_000,
                1600,
                step=100,
                help=_HELP.get("min_duration_samples"),
            )
        )

    # =========================================================================
    # AUDIO PREPROCESSING
    # =========================================================================
    # Defaults — always defined so they're safe to reference in the run-button kwargs.
    preprocess_audio_enabled: bool = False
    preprocess_config: Any = PreprocessConfig()
    preprocess_config_mild: Any = None
    preprocess_config_strong: Any = None

    with st.expander("Audio Preprocessing", expanded=False):
        st.markdown(
            "Adaptive signal conditioning: high-pass filter → noise reduction → "
            "loudness normalisation → peak limiting. All steps are optional and "
            "auto-configured based on input audio quality."
        )

        # Compute per-speaker audio diagnostics
        default_config = PreprocessConfig()
        auto_config = default_config
        all_speaker_stats: dict = (
            {}
        )  # {speaker: {"stats": ..., "auto_profile": ..., "auto_config": ...}}

        if speakers_audio is not None:
            _source = (
                speakers_audio
                if isinstance(speakers_audio, dict)
                else {"Audio": speakers_audio}
            )
            for _spk, _spk_path in _source.items():
                try:
                    _audio, _sr = load_audio(_spk_path)
                    _stats = analyse_audio(_audio, _sr)
                    _pname, _pcfg = auto_profile(_audio, _sr)
                    all_speaker_stats[_spk] = {
                        "stats": _stats,
                        "auto_profile": _pname,
                        "auto_config": _pcfg,
                    }
                except Exception:
                    st.warning(f"Could not analyse [{_spk}].")
            if all_speaker_stats:
                _first = list(all_speaker_stats.keys())[0]
                auto_config = all_speaker_stats[_first]["auto_config"]

        # Display per-speaker diagnostics table
        if all_speaker_stats:
            import pandas as pd

            st.subheader("Audio Quality Assessment (Per Speaker)")
            _diag_rows = []
            for _spk, _spk_data in all_speaker_stats.items():
                _s = _spk_data["stats"]
                _lufs = _s.get("lufs")
                _snr = _s.get("snr_estimate_db")
                _diag_rows.append(
                    {
                        "Speaker": _spk,
                        "Duration (s)": f"{_s.get('duration_sec', 0):.0f}",
                        "LUFS": f"{_lufs:.1f}" if _lufs is not None else "N/A",
                        "SNR (dB)": f"{_snr:.1f}" if _snr is not None else "N/A",
                        "Peak (dBFS)": f"{_s.get('peak_db', 0):.1f}",
                        "Auto Profile": _spk_data["auto_profile"].capitalize(),
                    }
                )
            st.dataframe(pd.DataFrame(_diag_rows), width="stretch", hide_index=True)
            st.caption(
                "LUFS: Clean > −26, Noisy < −35 | SNR: Clean > 25 dB, Noisy < 18 dB. "
                "Auto Profile evaluated independently per speaker."
            )

        # ── Preprocessing mode ────────────────────────────────────────────────
        st.subheader("Processing Profile")
        preprocess_mode = st.radio(
            "Preprocessing mode",
            ["Single", "Dual"],
            horizontal=True,
            help=(
                "**Single**: one preprocessing pass applied uniformly to all audio. "
                "**Dual**: runs preprocessing twice — a *mild* version used for "
                "VAD/diarization (preserving natural speaker characteristics) and a "
                "*strong* version used for ASR transcription (maximising speech clarity). "
                "Recommended for noisy recordings where heavy noise reduction would "
                "otherwise hurt speaker segmentation."
            ),
        )

        if preprocess_mode == "Single":
            preprocess_audio_enabled, preprocess_config = _render_preprocessing_profile(
                "", auto_config, default_config
            )
        else:
            # Dual mode: mild for VAD, strong for ASR
            st.info(
                "**Dual mode** generates two preprocessed versions of your audio:  \n"
                "- **Mild** (left): gentle processing used for VAD/diarization — avoids "
                "artifacts that confuse speaker segmentation.  \n"
                "- **Strong** (right): more aggressive enhancement used for ASR transcription "
                "— maximises speech clarity. Saved with `_mild` / `_strong` suffix."
            )
            mild_col, strong_col = st.columns(2)
            with mild_col:
                _, preprocess_config_mild = _render_preprocessing_profile(
                    "mild_",
                    auto_config,
                    default_config,
                    title="Mild — for VAD / Diarization",
                )
            with strong_col:
                _, preprocess_config_strong = _render_preprocessing_profile(
                    "strong_",
                    auto_config,
                    default_config,
                    title="Strong — for ASR Transcription",
                )

        # ── Per-speaker loudness normalisation impact table ──────────────
        if all_speaker_stats and preprocess_mode == "Single":
            import pandas as pd

            st.markdown("#### Per-Speaker Processing Impact")
            _impact_rows = []
            for _spk, _spk_data in all_speaker_stats.items():
                _s = _spk_data["stats"]
                _lufs = _s.get("lufs")
                _snr = _s.get("snr_estimate_db")

                # Loudness normalisation
                if _lufs is not None and preprocess_config.loudness_norm:
                    _gain_needed = preprocess_config.target_lufs - _lufs
                    _max_gain = preprocess_config.max_loudness_gain_db
                    _tol = preprocess_config.loudness_tolerance_db
                    if abs(_gain_needed) <= _tol:
                        _actual_gain = 0.0
                        _norm_note = f"skip (within ±{_tol:.0f} dB)"
                    elif _gain_needed > _max_gain:
                        _actual_gain = _max_gain
                        _norm_note = f"⚠ capped at +{_max_gain:.0f} dB (would distort)"
                    else:
                        _actual_gain = _gain_needed
                        _norm_note = "applied"
                    _eff_lufs = _lufs + _actual_gain
                else:
                    _actual_gain = 0.0
                    _eff_lufs = _lufs
                    norm_note = (
                        "disabled" if not preprocess_config.loudness_norm else "N/A"
                    )
                    _norm_note = norm_note

                # SNR → profile context
                if _snr is not None:
                    if _snr < 18.0:
                        _snr_note = f"{_snr:.1f} dB → Noisy (< 18)"
                    elif _snr > 25.0:
                        _snr_note = f"{_snr:.1f} dB → Clean (> 25)"
                    else:
                        _snr_note = f"{_snr:.1f} dB → Moderate"
                else:
                    _snr_note = "N/A"

                _impact_rows.append(
                    {
                        "Speaker": _spk,
                        "LUFS (raw)": f"{_lufs:.1f}" if _lufs is not None else "N/A",
                        "Gain Needed": (
                            f"{_gain_needed:+.1f} dB" if _lufs is not None else "N/A"
                        ),
                        "Gain Applied": (
                            f"{_actual_gain:+.1f} dB" if _lufs is not None else "N/A"
                        ),
                        "LUFS After Norm": (
                            f"{_eff_lufs:.1f}" if _eff_lufs is not None else "N/A"
                        ),
                        "Norm Status": _norm_note,
                        "SNR → Profile": _snr_note,
                    }
                )

            st.dataframe(pd.DataFrame(_impact_rows), width="stretch", hide_index=True)

            _notes = []
            if preprocess_config.highpass:
                _notes.append(f"HPF: {preprocess_config.highpass_freq:.0f} Hz cutoff")
            if preprocess_config.noise_reduce:
                _notes.append(
                    f"Noise reduction: strength "
                    f"{preprocess_config.noise_reduce_prop_decrease:.2f}"
                )
            if preprocess_config.peak_limit:
                import math

                _peak_db = 20.0 * math.log10(preprocess_config.peak_ceiling)
                _notes.append(
                    f"Peak limiter: {preprocess_config.peak_ceiling} "
                    f"({_peak_db:.2f} dBFS)"
                )
            if _notes:
                st.caption("Global (applies to all speakers): " + " | ".join(_notes))

    # =========================================================================
    # EVALUATION (OPTIONAL)
    # =========================================================================
    with st.expander("Evaluation (optional)"):
        evaluate_ref_file = st.file_uploader(
            "Reference labels file (leave blank to skip evaluation)",
            type=["txt", "ass", "elan", "rttm"],
            help="Upload your ground-truth annotations (.txt, .ass, .elan, or .rttm). "
            "The pipeline will compare its output against this reference.",
        )
        evaluate_ref_path: str | None = (
            _save_upload(evaluate_ref_file, prefix="ref_")
            if evaluate_ref_file
            else None
        )

        if evaluate_ref_path:
            all_stages = ["vad", "diarization", "transcription", "label_type"]
            evaluate_stages: list[str] | None = st.multiselect(
                "Stages to evaluate", all_stages, default=all_stages
            )
            evaluate_collar = st.slider("Evaluation collar (s)", 0.0, 1.0, 0.25, 0.01)
            evaluate_plot = st.checkbox("Generate evaluation plots", value=True)
            c1, c2 = st.columns(2)
            evaluate_plot_format = c1.selectbox("Plot format", ["pdf", "png", "svg"])
            _dpi_raw = int(
                c2.number_input("Plot DPI (0 = format default)", 0, 600, 0, step=50)
            )
            evaluate_plot_dpi: int | None = _dpi_raw if _dpi_raw > 0 else None
        else:
            evaluate_stages = None
            evaluate_collar = 0.25
            evaluate_plot = False
            evaluate_plot_format = "pdf"
            evaluate_plot_dpi = None

    # =========================================================================
    # ADVANCED / SKIP FLAGS
    # =========================================================================
    with st.expander("Advanced / Skip Flags"):
        c1, c2, c3 = st.columns(3)
        skip_vad_if_exists = c1.checkbox("Skip VAD if output exists", value=False)
        skip_transcription_if_exists = c2.checkbox(
            "Skip transcription if output exists", value=False
        )
        persist_transcription_artifacts = c3.checkbox(
            "Keep per-segment WAV / transcript cache", value=False
        )
        c4, c5 = st.columns(2)
        cleanup_speaker_folders = c4.checkbox(
            "Clean up speaker folders after run", value=True
        )
        export_elan = c5.checkbox("Export ELAN-compatible labels", value=True)

    # =========================================================================
    # RUN BUTTON
    # =========================================================================
    st.divider()

    _continue_mode = pipeline_mode == "Continue existing file"
    _audio_ready = speakers_audio is not None
    _continue_ready = _continue_mode and (existing_path is not None) and _audio_ready
    _full_ready = (not _continue_mode) and _audio_ready
    run_disabled = st.session_state.running or not (_continue_ready or _full_ready)
    if st.button("▶  Run Pipeline", type="primary", disabled=run_disabled):
        if _continue_mode:
            kwargs: dict[str, Any] = dict(
                existing_path=existing_path,
                speakers_audio=speakers_audio,
                output_dir=output_dir,
                transcription_model_name=transcription_model_name,
                whisper_device=whisper_device,
                whisper_language=whisper_language,
                whisper_model_batch_size=whisper_model_batch_size,
                transcription_padding_sec=transcription_padding_sec,
                entropy_threshold=entropy_threshold,
                max_backchannel_dur=max_backchannel_dur,
                max_gap_sec=max_gap_sec,
                batch_size=batch_size,
                persist_transcription_artifacts=persist_transcription_artifacts,
                cleanup_speaker_folders=cleanup_speaker_folders,
                min_duration_samples=float(min_duration_samples),
                export_elan=export_elan,
                evaluate_ref_path=evaluate_ref_path,
                evaluate_stages=evaluate_stages,
                evaluate_collar=evaluate_collar,
                evaluate_plot=evaluate_plot,
                evaluate_plot_format=evaluate_plot_format,
                evaluate_plot_dpi=evaluate_plot_dpi,
            )
            _worker = _continue_worker
        else:
            kwargs = dict(
                speakers_audio=speakers_audio,
                output_dir=output_dir,
                vad_type=vad_type,
                rvad_threshold=rvad_threshold,
                auth_token=auth_token,
                vad_min_duration=vad_min_duration,
                energy_margin_db=energy_margin_db,
                gap_thresh=gap_thresh,
                short_utt_thresh=short_utt_thresh,
                window_sec=window_sec,
                merge_short_after_long=merge_short_after_long,
                merge_long_after_short=merge_long_after_short,
                long_merge_enabled=long_merge_enabled,
                merge_max_dur=merge_max_dur,
                bridge_short_opponent=bridge_short_opponent,
                transcription_model_name=transcription_model_name,
                whisper_device=whisper_device,
                whisper_language=whisper_language,
                whisper_model_batch_size=whisper_model_batch_size,
                transcription_padding_sec=transcription_padding_sec,
                entropy_threshold=entropy_threshold,
                max_backchannel_dur=max_backchannel_dur,
                max_gap_sec=max_gap_sec,
                batch_size=batch_size,
                interactive_energy_filter=False,
                skip_vad_if_exists=skip_vad_if_exists,
                skip_transcription_if_exists=skip_transcription_if_exists,
                persist_transcription_artifacts=persist_transcription_artifacts,
                cleanup_speaker_folders=cleanup_speaker_folders,
                min_duration_samples=float(min_duration_samples),
                export_elan=export_elan,
                preprocess_config=preprocess_config,
                preprocess_config_mild=preprocess_config_mild,
                preprocess_config_strong=preprocess_config_strong,
                evaluate_ref_path=evaluate_ref_path,
                evaluate_stages=evaluate_stages,
                evaluate_collar=evaluate_collar,
                evaluate_plot=evaluate_plot,
                evaluate_plot_format=evaluate_plot_format,
                evaluate_plot_dpi=evaluate_plot_dpi,
            )
            _worker = _pipeline_worker

        shared: dict[str, Any] = {
            "log_lines": [],
            "result": None,
            "error": None,
            "done": False,
        }
        thread = threading.Thread(target=_worker, args=(kwargs, shared), daemon=True)
        st.session_state.log_lines = shared["log_lines"]
        st.session_state._shared = shared
        st.session_state._thread = thread
        st.session_state.running = True
        st.session_state.done = False
        st.session_state.result = None
        st.session_state.error = None
        thread.start()
        st.rerun()

    # =========================================================================
    # LIVE LOG (while running)
    # =========================================================================
    if st.session_state.running:
        st.subheader("Pipeline Log")
        _cancel_col, _ = st.columns([1, 5])
        with _cancel_col:
            if st.button("⏹  Cancel", type="secondary"):
                _cancel_pipeline()
                st.rerun()
        log_box = st.empty()
        shared = st.session_state._shared

        # Poll until the background thread is done, refreshing the log each cycle.
        while not shared["done"]:
            log_box.code("\n".join(st.session_state.log_lines), language="")
            time.sleep(0.4)

        # Thread finished — flush any remaining buffer and promote results.
        log_box.code("\n".join(st.session_state.log_lines), language="")
        st.session_state.running = False
        st.session_state.done = True
        st.session_state.result = shared["result"]
        st.session_state.error = shared["error"]
        st.rerun()

    # =========================================================================
    # RESULTS
    # =========================================================================
    if st.session_state.done:
        if st.session_state.error:
            st.error(f"**Pipeline failed:**\n\n```\n{st.session_state.error}\n```")
        else:
            st.success("✅ Pipeline completed successfully!")
            result: dict = st.session_state.result or {}

            # ── Output file table ────────────────────────────────────────────
            file_rows = [
                {"Output": k, "Path": str(v)}
                for k, v in result.items()
                if isinstance(v, str) and v and k not in ("output_dir",)
            ]
            if file_rows:
                import pandas as pd

                st.subheader("Output Files")
                st.dataframe(pd.DataFrame(file_rows), width="stretch", hide_index=True)

            # ── Final labels preview ────────────────────────────────────────
            final_df = result.get("final_df")
            if final_df is not None:
                with st.expander("Final Labels Preview", expanded=True):
                    st.dataframe(final_df, width="stretch")

            # ── Evaluation plots ────────────────────────────────────────────
            eval_plots = result.get("evaluation_plots")
            if eval_plots:
                with st.expander("Evaluation Plots", expanded=True):
                    for plot_name, plot_path in eval_plots.items():
                        if not plot_path or not os.path.exists(plot_path):
                            continue
                        ext = plot_path.lower().rsplit(".", 1)[-1]
                        st.caption(plot_name)
                        if ext in ["png", "jpg", "jpeg"]:
                            st.image(plot_path, use_container_width=True)
                        elif ext == "svg":
                            with open(plot_path, "r") as _f:
                                svg_data = _f.read()
                            st.markdown(svg_data, unsafe_allow_html=True)
                        elif ext == "pdf":
                            with open(plot_path, "rb") as _f:
                                st.download_button(
                                    label=f"⬇ Download {plot_name} (PDF)",
                                    data=_f.read(),
                                    file_name=os.path.basename(plot_path),
                                    mime="application/pdf",
                                    key=f"dl_pdf_{plot_name}",
                                )

            # ── Log (collapsed) ────────────────────────────────────────────
            if st.session_state.log_lines:
                with st.expander("Full Pipeline Log"):
                    st.code("\n".join(st.session_state.log_lines), language="")

        st.divider()
        if st.button("🔄  Reset / run again"):
            _reset_state()
            st.rerun()


if __name__ == "__main__":
    main()
