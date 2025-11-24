"""
Compatibility fix for torchaudio 2.8+ with speechbrain 1.0.3

The torchaudio.list_audio_backends() function was deprecated in torchaudio 2.8,
but speechbrain 1.0.3 still tries to use it. This module patches it and suppresses
the deprecation warning.
"""

import warnings

import torchaudio


def _list_audio_backends():
    """Dummy implementation that returns soundfile backend."""
    return ["soundfile"]


# Monkey patch if the function doesn't exist
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = _list_audio_backends

# Suppress the deprecation warning from speechbrain
warnings.filterwarnings(
    "ignore",
    message="torchaudio._backend.list_audio_backends has been deprecated",
    category=UserWarning,
    module="speechbrain.utils.torch_audio_backend",
)
