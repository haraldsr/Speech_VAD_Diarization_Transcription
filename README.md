# Speech VAD, Diarization & Transcription Pipeline

End-to-end processing of conversation recordings with support for both pre-separated and mixed audio. Produces cleaned, transcribed, and labeled segments; labels can optionally be exported to the user's annotation software of choice (e.g., ELAN) for manual review.

<!-- ![Pipeline Flowchart](docs/figures/Protocol.png)
*Pipeline architecture: splits at Stage 1 (VAD vs Diarization) based on input type, then converges for unified processing.* -->

## Features

- **Multiple VAD methods**: rVAD, Silero (multi-channel)
- **Diarization**: Pyannote.audio and NeMo for single-channel recordings
- **Transcription**: Whisper with GPU acceleration and batching
- **Smart processing**: Turn merging, entropy-based labeling, context-aware annotation
- **Audio preprocessing**: Optional high-pass filtering, noise reduction, and loudness normalisation
- **Stage-wise evaluation**: VAD, diarization, segmentation, transcription, and label-type metrics with plotting
- **Streamlit GUI**: Interactive web interface (`app_gui.py`) for running the pipeline without code
- **Continue mode**: Resume processing from an existing labels file (`.txt` or `.ass`), adding only missing steps

---

## Installation

FFmpeg is required for audio processing. The recommended approach uses Make shortcuts for a one-command install.

### Make Shortcuts (Recommended)

```bash
git clone https://github.com/haraldsr/Speech_VAD_Diarization_Transcription.git
cd Speech_VAD_Diarization_Transcription
make install         # Auto-detects GPU/CPU and installs from lockfile
```

`make install` auto-detects GPU using `nvidia-smi`:
- **GPU detected** → uses `requirements-lock-uv-gpu.txt`
- **No GPU** → uses `requirements-lock-uv-cpu.txt`

Other shortcuts:

```bash
make install-dev     # Install from requirements.txt (for development/testing)
make install-conda   # Full Conda install (slower, no UV)
```

**For maintainers (creating lockfiles):**

```bash
make gen-lock        # Auto-detects GPU and names the lockfile accordingly
```

### Manual Option 1: Pure UV

Use this if you have admin access to install system packages.

```bash
# Install FFmpeg
sudo apt update && sudo apt install -y ffmpeg  # Ubuntu/Debian
# brew install ffmpeg                          # macOS

# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and set up
git clone https://github.com/haraldsr/Speech_VAD_Diarization_Transcription.git
cd Speech_VAD_Diarization_Transcription

uv venv --python 3.10
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -r requirements-lock-uv-cpu.txt  # or requirements-lock-uv-gpu.txt
pip install -e .
```

### Manual Option 2: Hybrid Conda + UV (No Admin Access)

Use this on HPC clusters or systems where you cannot install FFmpeg at the system level. Requires Conda/Mamba to be pre-installed.

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and set up
git clone https://github.com/haraldsr/Speech_VAD_Diarization_Transcription.git
cd Speech_VAD_Diarization_Transcription

# Create Conda environment (provides FFmpeg)
conda env create -f environment-minimal.yml -n vdt  # or use mamba
conda activate vdt

# Install Python packages with UV
uv pip install -r requirements-lock-uv-cpu.txt  # or requirements-lock-uv-gpu.txt
pip install -e .
```

**Note:** Mixing Conda and UV is not ideal, but necessary when system package installation is unavailable.

---

## Quick Start

### Streamlit GUI

The easiest way to run the pipeline is via the interactive web interface:

```bash
make app
```

Alternatively, you can launch the Streamlit app directly:

```bash
streamlit run app_gui.py
```

The GUI supports:
- **Pre-separated audio** (dyad/triad): upload one file per speaker
- **Diarization mode**: upload a single mixed-channel file
- **Continue mode**: upload an existing labels file (`.txt` or `.ass`) to add missing transcription or classification
- **Live log output**, cancel button, and inline evaluation plots

### Python API

See `conversation_pipeline.py` for complete examples with different configurations.

#### Pre-separated Audio (Dyad/Triad)

```python
from speech_vad_diarization_transcription import process_conversation

# Dyad example (two speakers, separate audio files)
results = process_conversation(
    speakers_audio={
        "P1": "path/to/speaker1.wav",
        "P2": "path/to/speaker2.wav",
    },
    output_dir="outputs/dyad",
    vad_type="silero",  # or "rvad"
)

# Triad example (three speakers)
results = process_conversation(
    speakers_audio={
        "P1": "path/to/speaker1.wav",
        "P2": "path/to/speaker2.wav",
        "P3": "path/to/speaker3.wav",
    },
    output_dir="outputs/triad",
    vad_type="rvad",
)
```

### Single Mixed Audio (Diarization)

Requires a HuggingFace token with access to pyannote models. Set via:
- Environment variable: `export HF_TOKEN="your-token"`
- Or login: `huggingface-cli login`

```python
# Single file with multiple speakers - uses pyannote diarization
results = process_conversation(
    speakers_audio="path/to/mixed_audio.wav",
    output_dir="outputs/diarized",
    vad_type="pyannote",  # Required for diarization
    auth_token=os.environ.get("HF_TOKEN"),  # Or None if logged in via CLI
)
```

### Speaker Separation + Pipeline

For mixed audio with overlapping speakers, first separate with SepFormer:

```python
# See run_separation_and_pipeline.py for complete example
from speech_separation_chunked import separator, separate_audio_with_smart_chunking
model = separator.from_hparams(source="speechbrain/sepformer-wsj02mix")
separated = separate_audio_with_smart_chunking(model, "mixed.wav")
# Then use process_conversation() with separated audio paths
```

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vad_type` | `"rvad"` | VAD method: `"rvad"`, `"silero"`, `"pyannote"`, or `"nemo"` |
| `vad_min_duration` | `0.07` | Minimum segment duration (seconds) |
| `energy_margin_db` | `20.0` | Energy threshold for filtering (dB) |
| `gap_thresh` | `0.5` | Max gap for merging segments |
| `transcription_model_name` | `"large-v3"` | Whisper model (or custom like `"CoRal-project/roest-whisper-large-v1"`) |
| `whisper_language` | `"da"` | Target language code |
| `whisper_device` | `"auto"` | `"auto"`, `"cuda"`, or `"cpu"` |
| `batch_size` | `30.0` | Batch size in seconds |
| `skip_vad_if_exists` | `False` | Skip VAD/diarization if `combined_vad.txt` already exists |
| `preprocess_audio_enabled` | `False` | Apply audio enhancement before VAD and transcription |
| `export_elan` | `True` | Export tab-delimited file for annotation software |

---

## Output Files

![Output Structure](docs/figures/Protocol-2.png)
*Each speaker track consists of discrete, timestamped speech intervals (Turns or Backchannels).*

```
outputs/
└── experiment_name/
    ├── P1/                            # Speaker-specific folder
    │   └── speaker1_vad.txt           # VAD timestamps
    ├── P2/
    │   └── speaker2_vad.txt
    ├── merged_turns.txt               # Merged conversation turns
    ├── raw_transcriptions.txt         # Raw Whisper output
    ├── classified_transcriptions.txt  # With entropy labels
    ├── final_labels.txt               # Context-merged annotations (TSV)
    └── final_labels_elan.txt          # ELAN-compatible format
```

**Pipeline output format (`final_labels.txt`):**
```
speaker	start_sec	end_sec	transcription	entropy	type
P1	0.50	2.30	Hello there	2.31	turn
P2	2.45	3.10	Mm-hmm	0.00	backchannel
```

**ELAN import format (`final_labels_elan.txt`):**
```
tier	begin	end	annotation
P1_turn	500	2300	Hello there
P2_backchannel	2450	3100	Mm-hmm
```

To import in ELAN: **File → Import → Tab-delimited Text...** (skip first line: Yes)

---

## Audio Preprocessing / Speech Enhancement

The pipeline includes an optional audio preprocessing step that applies adaptive signal conditioning before VAD and transcription. This can significantly improve downstream quality, especially for recordings with low volume, background noise, or DC offset.

### Enabling Preprocessing

```python
# Simple flag — uses default settings (recommended starting point)
results = process_conversation(
    speakers_audio={"P1": "p1.wav", "P2": "p2.wav"},
    output_dir="outputs/dyad",
    preprocess_audio_enabled=True,
)

# Fine-grained control via PreprocessConfig
from speech_vad_diarization_transcription import PreprocessConfig

config = PreprocessConfig(
    enabled=True,
    highpass=True,          # remove DC offset and rumble (<60 Hz)
    highpass_freq=60.0,
    noise_reduce=True,      # spectral gating noise reduction
    noise_reduce_stationary=True,
    noise_reduce_prop_decrease=0.8,
    loudness_norm=True,     # EBU R 128 loudness normalisation
    target_lufs=-23.0,
    auto_loudness=True,     # only normalise if off by >3 dB
    peak_limit=True,        # prevent clipping after gain
    peak_ceiling=0.95,
)

results = process_conversation(
    speakers_audio={"P1": "p1.wav", "P2": "p2.wav"},
    output_dir="outputs/dyad",
    preprocess_config=config,
)
```

### Processing Steps

| Step | What it does | Adaptive? |
|------|-------------|-----------|
| **High-pass filter** | Butterworth 5th-order HPF removes DC offset and low-frequency rumble | Cut-off frequency configurable |
| **Noise reduction** | Spectral gating reduces stationary background noise | Noise profile estimated from quietest frames |
| **Loudness normalisation** | Targets -23 LUFS (EBU R 128 broadcast standard) | Skipped if already within tolerance |
| **Peak limiter** | Hard-clips to 0.95 to prevent clipping after gain | Always applied after gain |

Preprocessed files are saved to `output_dir/preprocessed/` with an `_enhanced` suffix. All downstream pipeline stages (VAD, energy filtering, transcription) automatically use the enhanced audio.

### Standalone Usage

```python
from speech_vad_diarization_transcription import preprocess_audio, PreprocessConfig, analyse_audio
import soundfile as sf

# Analyse audio properties
audio, sr = sf.read("recording.wav", dtype="float32")
stats = analyse_audio(audio, sr)
print(f"LUFS: {stats['lufs']}, SNR: {stats.get('snr_estimate_db')} dB")

# Preprocess
out_path = preprocess_audio("recording.wav", "output/", config=PreprocessConfig())
```

### Auto Profiles

The `auto_profile()` function analyses the input audio and selects an appropriate preprocessing intensity automatically:

```python
from speech_vad_diarization_transcription import auto_profile
import soundfile as sf

audio, sr = sf.read("recording.wav", dtype="float32")
profile_name, config = auto_profile(audio, sr)
# profile_name is one of: "clean", "moderate", "noisy"
print(f"Selected profile: {profile_name}")
```

| Profile | Condition | Behaviour |
|---------|-----------|-----------|
| **clean** | LUFS > -26 *and* SNR > 25 dB | HPF + gentle loudness norm only |
| **moderate** | Neither clean nor noisy | Standard 4-stage pipeline |
| **noisy** | LUFS < -35 *or* SNR < 18 dB | Aggressive noise reduction (0.95), tighter loudness tolerance |

### Dual-Output Mode

For best results on noisy recordings, use `dual_preprocess()` (or enable **Dual** mode in the GUI) to produce two versions of the audio — a **mild** version for diarization/VAD (sensitive to noise-reduction artefacts) and a **strong** version for ASR transcription:

```python
from speech_vad_diarization_transcription import dual_preprocess

paths = dual_preprocess("recording.wav", "output/")
# paths = {"mild": "output/preprocessed/recording_mild.wav",
#          "strong": "output/preprocessed/recording_strong.wav"}
```

Or pass both configs directly to `process_conversation` to let the pipeline handle it:

```python
from speech_vad_diarization_transcription import PreprocessConfig

results = process_conversation(
    speakers_audio="recording.wav",
    output_dir="outputs/diarized",
    vad_type="pyannote",
    preprocess_config_mild=PreprocessConfig(enabled=True, noise_reduce_prop_decrease=0.5),
    preprocess_config_strong=PreprocessConfig(enabled=True, noise_reduce_prop_decrease=0.9),
)
```

In the GUI, open the **Audio Preprocessing** expander and select **Dual** as the preprocessing mode. A mild and strong profile picker will appear side by side.

---

## Stage-Wise Evaluation

The pipeline includes evaluation metrics for each processing stage, built on top of [`pyannote.metrics`](https://github.com/pyannote/pyannote-audio) and [`jiwer`](https://github.com/jitsi/jiwer). These are independent of the existing turn-taking dynamics evaluator in `compute_turn_errors.py`.

### Evaluated Stages

| Stage | Metrics | Library |
|-------|---------|---------|
| **VAD** | Detection Error Rate, Detection Accuracy, Precision, Recall, F1, Onset/Offset MAE | pyannote.metrics |
| **Diarization** | DER (+ miss/FA/confusion), Greedy DER, JER, Purity, Coverage, Homogeneity, Completeness, IER (+ P/R), Speaker Detection Accuracy, Speaker ID Accuracy (mapped *and* raw) | pyannote.metrics, scipy |
| **Segmentation** | Purity, Coverage, Precision, Recall, F-measure | pyannote.metrics |
| **Transcription** | WER, CER, MER, WIL, WIP, BLEU, Semantic Distance/Similarity (raw + normalised), edit counts (S/D/I/H) — uses many-to-many time alignment | jiwer, sacrebleu, transformers |
| **Label Type** | Per-class P/R/F1, Macro F1, confusion matrix | built-in |

### Inline Evaluation (during pipeline run)

Pass a ground-truth reference file to `process_conversation` to evaluate at the end of the run:

```python
results = process_conversation(
    speakers_audio={"P1": "speaker1.wav", "P2": "speaker2.wav"},
    output_dir="outputs/dyad",
    vad_type="silero",
    # Evaluation parameters
    evaluate_ref_path="examples/Dyad/EXP_12_T2/EXP12_T2_Hanlu.txt",
    evaluate_stages=["vad", "diarization", "segmentation", "label_type"],  # or None for all
    evaluate_collar=0.25,
    evaluate_plot=True,            # optional: generate evaluation plots
    evaluate_plot_format="pdf",   # default: png (png | pdf | svg)
)
# Metrics are printed and saved to outputs/dyad/evaluation_metrics.json
# Plots are saved to outputs/dyad/evaluation_kpi.pdf
# Also available via results["evaluation"]
```

### Standalone CLI Evaluation

Evaluate existing pipeline outputs without re-running the pipeline:

```bash
# Single pair — reference vs hypothesis
python scripts/evaluate.py \
    --ref examples/Dyad/EXP_12_T2/EXP12_T2_Hanlu.txt \
  --hyp outputs/dyad/final_labels.txt \
  --plot

# Point at output directory (auto-finds final_labels.txt)
python scripts/evaluate.py \
    --ref examples/all_files/EXP10_NoiseP1_T1.txt \
  --output-dir outputs/test \
  --plot --plot-format pdf

# Select specific stages and collar
python scripts/evaluate.py \
    --ref ref.txt --hyp hyp.txt \
    --stages vad diarization \
    --collar 0.5

# Batch mode (tab-separated ref<TAB>hyp file)
python scripts/evaluate.py --batch-file eval_pairs.txt --json results.json

# Control plot output location and quality
python scripts/evaluate.py \
  --ref ref.txt --hyp hyp.txt \
  --plot --plot-dir figures --plot-format png --plot-dpi 200

# Default plotting format is PDF (no DPI needed)
python scripts/evaluate.py \
  --ref ref.txt --hyp hyp.txt \
  --plot
```

### Supported Reference Formats

References are auto-detected but can be overridden with `--ref-fmt`:

| Format | Description | Example |
|--------|-------------|---------|
| `exp5` | 5-column TSV: `speaker start end dur type` | `examples/all_files/EXP10_*.txt` |
| `exp6` | 6-column TSV with blank field | `examples/Dyad/*/EXP*_Hanlu.txt` |
| `elan` | ELAN tab-delimited: `tier begin end annotation` | `examples/coral/*_elan.txt` |
| `rttm` | NIST RTTM (standard diarization format) | — |
| `pipeline_output` | Pipeline's own `final_labels.txt` | `outputs/*/final_labels.txt` |

### Output

Evaluation produces `evaluation_metrics.json` in the output directory:

```json
{
  "vad": {
    "pooled": {
      "detection_error_rate": 0.12,
      "detection_accuracy": 0.88,
      "precision": 0.94,
      "recall": 0.91,
      "f1": 0.92,
      "onset_mae": 0.08,
      "offset_mae": 0.11
    },
    "per_speaker": { ... }
  },
  "diarization": {
    "diarization_error_rate": 0.15,
    "greedy_diarization_error_rate": 0.14,
    "jaccard_error_rate": 0.18,
    "der_miss": 0.04, "der_false_alarm": 0.06, "der_confusion": 0.05,
    "purity": 0.92, "coverage": 0.89,
    "homogeneity": 0.91, "completeness": 0.88,
    "identification_error_rate": 0.16,
    "identification_precision": 0.87, "identification_recall": 0.85,
    "speaker_detection_accuracy": 0.90,
    "speaker_id_accuracy": 0.87,
    "speaker_mapping": {"hyp_A": "ref_1", "hyp_B": "ref_2"},
    "per_speaker_id_accuracy": {"ref_1": 0.91, "ref_2": 0.83}
  },
  "segmentation": {
    "purity": 0.93, "coverage": 0.90,
    "precision": 0.88, "recall": 0.85, "f_measure": 0.86
  },
  "transcription": {
    "pooled": {
      "raw": { "wer": 0.32, "cer": 0.15 },
      "normalised": { "wer": 0.25, "cer": 0.11 },
      "coverage": 0.87
    }
  },
  "label_type": {
    "macro_f1": 0.78,
    "per_class": { "turn": { ... }, "backchannel": { ... } }
  }
}
```

### Programmatic API

```python
from speech_vad_diarization_transcription import (
    evaluate_pipeline,
    evaluate_vad,
    evaluate_diarization,
    evaluate_segmentation,
    evaluate_transcription,
    evaluate_label_type,
    load_reference,
    print_evaluation_summary,
    plot_evaluation_results,
    plot_evaluation_json,
)

# Load and evaluate individual stages
ref = load_reference("ground_truth.txt")
hyp = load_reference("outputs/final_labels.txt")

vad_results = evaluate_vad(ref, hyp, collar=0.25)
diar_results = evaluate_diarization(ref, hyp)
seg_results = evaluate_segmentation(ref, hyp)
asr_results = evaluate_transcription(ref, hyp)
type_results = evaluate_label_type(ref, hyp)

# Plot directly from in-memory results
all_results = evaluate_pipeline("ground_truth.txt", "outputs/final_labels.txt")
plot_evaluation_results(all_results, output_dir="outputs")

# Or from an existing JSON file
plot_evaluation_json("outputs/evaluation_metrics.json")
```

---

## Carbon Tracking

The pipeline optionally integrates [CarbonTracker](https://github.com/lfwa/carbontracker) for monitoring energy consumption and CO₂ emissions during processing.

To enable carbon tracking in `conversation_pipeline.py`:

```python
ENABLE_CARBON_TRACKING = True  # Set in conversation_pipeline.py
```

Optionally set an [Electricity Maps](https://www.electricitymaps.com/) API key for accurate CO₂ intensity data:

```bash
export ELECTRICITYMAPS_API_KEY="your-api-key"
```

Logs are saved to `logs/carbon/`. See `conversation_pipeline.py` for configuration options including CPU TDP simulation for systems without direct power measurement.

---

## Troubleshooting

### FFmpeg Not Found / Torchcodec Error
```bash
# The error "Could not load libtorchcodec" means FFmpeg is missing
# Install FFmpeg at system level:

# Ubuntu/Debian:
sudo apt update && sudo apt install -y ffmpeg

# macOS:
brew install ffmpeg

# Verify installation:
ffmpeg -version

# Then reinstall torchcodec:
pip install --force-reinstall torchcodec==0.8.1
```

**Note:** Conda environments include FFmpeg automatically. UV/pip environments require manual FFmpeg installation.

### FFmpeg Version Mismatch

If you see errors about missing symbols or incompatible FFmpeg libraries:

```bash
# Check FFmpeg version (requires 6.x or 7.x for torchcodec)
ffmpeg -version

# Ubuntu 22.04 has FFmpeg 4.x by default - use Conda or upgrade:
conda install -c conda-forge ffmpeg=7.*

# Or reinstall torchcodec to match your FFmpeg version:
pip install --force-reinstall torchcodec
```

### Out of Memory
```python
# Reduce batch sizes
batch_size=15.0, # sec
```

### GPU Not Detected
```python
# Check PyTorch CUDA
import torch
print(torch.cuda.is_available())

# Force CPU if needed
whisper_device="cpu"
```

### Package Version Conflicts
```bash
# Use UV for better dependency resolution
uv pip install -r requirements-lock-uv-gpu.txt --upgrade  # or requirements-lock-uv-cpu.txt
```

---

## Repository Structure

```
.
├── Makefile                      # Build automation (install, lint, clean)
├── README.md
├── LICENSE
├── pyproject.toml                # Package metadata and tool configuration
├── setup.py                      # Package installation
├── environment-minimal.yml       # Minimal Conda env (Python + FFmpeg only)
├── environment.yml               # Full Conda environment
├── requirements.txt              # Flexible dependencies (development)
├── requirements-lock-uv-gpu.txt  # Exact dependencies for GPU (reproducibility)
├── requirements-lock-uv-cpu.txt  # Exact dependencies for CPU (reproducibility)
├── app_gui.py                    # Streamlit GUI for interactive pipeline runs
├── conversation_pipeline.py      # Example usage with dyad/triad/diarization configs
├── docs/
│   └── figures/                  # Pipeline diagrams
├── scripts/
│   ├── evaluate.py               # Standalone CLI evaluation tool
│   └── generate_uv_lock.sh       # Script to regenerate lockfile
└── src/                          # Package source (installed as speech_vad_diarization_transcription)
    ├── __init__.py               # Exports process_conversation, evaluate_*, etc.
    ├── conversation.py           # Main API: process_conversation(), continue_conversation()
    ├── vad.py                    # VAD wrappers (rVAD, Silero, Pyannote, NeMo)
    ├── postprocess_vad.py        # Energy filtering, segment cleaning
    ├── merge_turns.py            # Turn merging logic
    ├── transcription.py          # Whisper transcription
    ├── labeling.py               # Entropy-based labeling
    ├── audio_preprocessing.py    # Audio enhancement (HPF, noise reduction, loudness norm)
    ├── evaluation.py             # Stage-wise evaluation (VAD, diarization, segmentation, ASR, label type)
    ├── evaluation_plots.py       # KPI and per-speaker bar chart plots
    └── compute_turn_errors.py    # Turn-taking dynamics evaluator
```

---

## Development

### Make Commands

```bash
make help          # Show all available commands
make install       # Install with hybrid approach (recommended)
make install-dev   # Install from requirements.txt (development)
make install-conda # Install with Conda/Mamba only
make gen-lock      # Generate lockfile from active environment
make lint          # Run linting (flake8, mypy, isort, black)
make format        # Format code (isort + black)
make clean         # Remove build artifacts
```

### View Installed Packages
```bash
pip list
conda list  # If using conda
```

### Update Environment to Match Working Setup
```bash
# Generate a lockfile from the active environment (use make gen-lock instead)
conda activate vdt
make gen-lock

# Then apply to another environment
conda activate other_env
uv pip install -r requirements-lock-uv-gpu.txt --upgrade  # or requirements-lock-uv-cpu.txt
```

---

## Credits

- **Pyannote.audio**: https://github.com/pyannote/pyannote-audio
- **SpeechBrain**: https://speechbrain.github.io/
- **Whisper**: https://github.com/openai/whisper
- **rVAD**: https://github.com/zhenghuatan/rVADfast
- **UV**: https://github.com/astral-sh/uv
- **CarbonTracker**: https://github.com/lfwa/carbontracker

---

## License

**TODO: Temporary** - All Rights Reserved - Copyright (c) 2025 Harald Skat-Rørdam, Hanlu He

No license is currently granted for use, modification, or distribution of this software. An open-source license will be applied once determined by the copyright holders. See [LICENSE](LICENSE) file for details.
