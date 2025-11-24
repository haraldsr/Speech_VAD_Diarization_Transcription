"""Speech VAD, Diarization & Transcription Pipeline."""

# Apply torchaudio compatibility fix before importing anything else
from . import fix_torchaudio  # noqa: F401

__all__ = [
    "__version__",
    "process_conversation",
    "load_whisper_model",
    "transcribe_segments",
]

__version__ = "0.1.0"

from .conversation import process_conversation  # noqa: E402
from .transcription import load_whisper_model, transcribe_segments  # noqa: E402
