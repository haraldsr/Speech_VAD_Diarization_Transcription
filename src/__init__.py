"""Speech VAD, Diarization & Transcription Pipeline."""

__all__ = [
    "__version__",
    "process_conversation",
    "continue_conversation",
    "load_whisper_model",
    "transcribe_segments",
    "compute_all_errors",
    "postprocess_turn_df",
    # Stage-wise evaluation
    "evaluate_pipeline",
    "evaluate_pipeline_from_dir",
    "evaluate_vad",
    "evaluate_diarization",
    "evaluate_segmentation",
    "evaluate_transcription",
    "evaluate_label_type",
    "load_reference",
    "print_evaluation_summary",
    "plot_evaluation_results",
    "plot_evaluation_json",
    # Audio preprocessing
    "PreprocessConfig",
    "preprocess_audio",
    "preprocess_speakers_audio",
    "analyse_audio",
    "load_audio",
    "auto_profile",
    "dual_preprocess",
]

__version__ = "1.0.1"

from .audio_preprocessing import (
    PreprocessConfig,
    analyse_audio,
    auto_profile,
    dual_preprocess,
    load_audio,
    preprocess_audio,
    preprocess_speakers_audio,
)
from .compute_turn_errors import compute_all_errors, postprocess_turn_df
from .conversation import continue_conversation, process_conversation
from .evaluation import (
    evaluate_diarization,
    evaluate_label_type,
    evaluate_pipeline,
    evaluate_pipeline_from_dir,
    evaluate_segmentation,
    evaluate_transcription,
    evaluate_vad,
    load_reference,
    print_evaluation_summary,
)
from .evaluation_plots import (
    plot_evaluation_json,
    plot_evaluation_results,
)
from .transcription import load_whisper_model, transcribe_segments
