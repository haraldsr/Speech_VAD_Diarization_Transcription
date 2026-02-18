"""
Based on Conversational_speech_labeling_pipeline by Hanlu He.

https://github.com/hanlululu/Conversational_speech_labeling_pipeline
"""

import gc
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from faster_whisper import BatchedInferencePipeline, WhisperModel
from tqdm.auto import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# from transformers.pipelines.base import Pipeline

# Set PyTorch CUDA memory configuration for better fragmentation handling
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


@dataclass
class TransformersASRModel:
    backend: str
    pipeline: Any
    language: Optional[str]
    transcription_model_name: str
    device: str
    cache_dir: Optional[str]
    model_batch_size: int
    compute_type: str


def load_whisper_model(
    transcription_model_name: str = "openai/whisper-large-v3",
    device: str = "cpu",
    language: Optional[str] = "da",
    cache_dir: Optional[str] = None,
    model_batch_size: int = 100,
    backend: str = "auto",
    compute_type: Optional[str] = None,
) -> TransformersASRModel:
    """Initialise and return a Whisper ASR model via the Transformers pipeline.

    Parameters
    ----------
    transcription_model_name
        Model identifier (e.g., 'openai/whisper-large-v3')
    device
        'cpu' or 'cuda' for GPU inference
    language
        Target language code (e.g., 'da' for Danish)
    cache_dir
        Optional directory for model caching
    model_batch_size
        Maximum number of audio files to process in parallel. Higher values improve
        throughput but require more GPU memory.
        Recommended: 1-8 for large models, 16-32 for smaller models.

    Returns
    -------
    TransformersASRModel
        Wrapper containing the configured transformers pipeline
    """

    if backend == "auto":
        if transcription_model_name.startswith("faster-whisper:"):
            backend = "faster-whisper"
        elif "/" in transcription_model_name:
            backend = "transformers"
        else:
            backend = "faster-whisper"

    if backend == "faster-whisper":
        model_id = transcription_model_name
        if model_id.startswith("faster-whisper:"):
            model_id = model_id.split(":", 1)[1]

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        if compute_type is None:
            compute_type = "float16" if device == "cuda" else "int8"

        fw_model = WhisperModel(
            model_id,
            device=device,
            compute_type=compute_type,
            download_root=cache_dir,
        )
        fw_pipeline = BatchedInferencePipeline(model=fw_model)

        return TransformersASRModel(
            backend="faster-whisper",
            pipeline=fw_pipeline,
            language=language,
            transcription_model_name=transcription_model_name,
            device=device,
            cache_dir=cache_dir,
            model_batch_size=model_batch_size,
            compute_type=compute_type,
        )

    # Convert device string to torch.device
    if device == "auto":
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    elif device == "cuda":
        torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    else:
        torch_device = torch.device("cpu")
        torch_dtype = torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        transcription_model_name,
        torch_dtype=torch_dtype,
        cache_dir=cache_dir,
    )
    model.to(torch_device)

    processor = AutoProcessor.from_pretrained(
        transcription_model_name, cache_dir=cache_dir
    )

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        batch_size=model_batch_size,
        torch_dtype=torch_dtype,
        device=torch_device,
        chunk_length_s=30.0,
    )

    return TransformersASRModel(
        backend="transformers",
        pipeline=pipe,
        language=language,
        transcription_model_name=transcription_model_name,
        device=device,
        cache_dir=cache_dir,
        model_batch_size=model_batch_size,
        compute_type=compute_type or "float16",
    )


def _save_segment_wav(
    out_path: str, audio_array: np.ndarray, sr: int = 16000, compress: bool = True
) -> None:
    """Persist a speech segment to disk as 16-bit PCM WAV."""

    if compress:
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)
        audio_array = np.clip(audio_array, -1.0, 1.0)
        sf.write(out_path, audio_array, samplerate=sr, subtype="PCM_16")
    else:
        sf.write(out_path, audio_array, samplerate=sr)


def _fw_collect_text(segments: Any) -> str:
    """Collect text from faster-whisper segments."""
    return " ".join(segment.text.strip() for segment in segments).strip()


def _fw_transcribe_files(
    files_to_transcribe: List[str],
    model: TransformersASRModel,
) -> List[str]:
    """Transcribe a list of files with faster-whisper, with batching fallback."""
    fw_pipeline: BatchedInferencePipeline = model.pipeline
    language = model.language
    effective_batch_size = max(1, int(model.model_batch_size))

    try:
        segments, _info = fw_pipeline.transcribe(
            files_to_transcribe,
            batch_size=effective_batch_size,
            language=language,
            task="transcribe",
        )

        if isinstance(segments, list):
            if segments and isinstance(segments[0], list):
                return [_fw_collect_text(segs) for segs in segments]
            return [_fw_collect_text(segments)]

        return [_fw_collect_text(list(segments))]
    except Exception:
        texts = []
        for seg_file in files_to_transcribe:
            segments, _info = fw_pipeline.transcribe(
                seg_file,
                batch_size=1,
                language=language,
                task="transcribe",
            )
            texts.append(_fw_collect_text(segments))
        return texts


def _transcribe_batch(
    batch_files: List[str],
    batch_caches: List[str],
    model: TransformersASRModel,
    cache: bool = False,
) -> List[str]:
    """Transcribe a batch of segment files using pipeline batching.

    Returns list of transcribed texts (in same order as input).
    """
    # Check which files need transcription (not cached)
    files_to_transcribe = []
    file_indices = []
    results = [""] * len(batch_files)

    for i, (seg_file, txt_cache) in enumerate(zip(batch_files, batch_caches)):
        if cache and os.path.exists(txt_cache):
            # Load from cache, but skip if it's a failed transcription
            with open(txt_cache, "r", encoding="utf-8") as cache_file:
                cached_text = cache_file.read().strip()
                if not cached_text.startswith("[TRANSCRIPTION_FAILED:"):
                    results[i] = cached_text
                    continue
            # If we get here, cache was invalid - need to re-transcribe
            files_to_transcribe.append(seg_file)
            file_indices.append(i)
        else:
            # Needs transcription
            files_to_transcribe.append(seg_file)
            file_indices.append(i)

    # If no files need transcription, return cached results
    if not files_to_transcribe:
        return results

    if model.backend == "faster-whisper":
        texts = _fw_transcribe_files(files_to_transcribe, model)

        for batch_idx, text in zip(file_indices, texts):
            results[batch_idx] = text
            cache_path = batch_caches[batch_idx]
            with open(cache_path, "w", encoding="utf-8") as cache_file:
                cache_file.write(text)

        return results

    pipe = model.pipeline
    language = model.language

    generate_kwargs: Dict[str, Any] = {
        "task": "transcribe"
    }  # , "return_timestamps": True
    if language:
        generate_kwargs["language"] = language

    # Transcribe batch
    max_pipe_batch = getattr(pipe, "batch_size", None)
    effective_batch_size = (
        min(len(files_to_transcribe), int(max_pipe_batch))
        if max_pipe_batch
        else len(files_to_transcribe)
    )
    batch_results = pipe(
        files_to_transcribe,
        return_timestamps=True,
        generate_kwargs=generate_kwargs,
        batch_size=effective_batch_size,
    )

    if isinstance(batch_results, dict):
        batch_results = [batch_results]

    for batch_idx, result in zip(file_indices, batch_results):
        text = result.get("text", "").strip()
        results[batch_idx] = text

        cache_path = batch_caches[batch_idx]
        with open(cache_path, "w", encoding="utf-8") as cache_file:
            cache_file.write(text)

    return results


def transcribe_segments(
    model: TransformersASRModel,
    segments: pd.DataFrame,
    audio_path: str,
    output_dir: str,
    speaker: str,
    *,
    file_prefix: Optional[str] = None,
    cache: bool = True,
    min_duration_samples: int = 1600,
    batch_size: float | None = 30.0,
    compress: bool = True,
) -> List[Dict[str, Any]]:
    """Run ASR on a set of time-stamped segments extracted from ``audio_path``.

    Parameters
    ----------
    model
        A loaded Whisper model obtained via :func:`load_whisper_model`.
    segments
        DataFrame with ``start_sec`` and ``end_sec`` columns that describe the
        regions to transcribe. A ``speaker`` column is optional and overrides
        the supplied ``speaker`` argument per row when present.
    audio_path
        Source waveform on disk from which to slice the segments.
    output_dir
        Directory where per-segment WAV and cached transcripts are written.
    speaker
        Identifier tagged on each transcription record.
    file_prefix
        Optional custom stem for generated filenames; defaults to ``speaker``.
    cache
        When ``True`` reuses cached transcripts when present.
    min_duration_samples
        Segments shorter than this many samples are skipped to avoid unstable
        recognitions.
    batch_size
        Maximum total audio duration (in seconds) to group into a single batch
        for transcription. Default: 30 seconds. Segments are grouped by duration,
        not count. This is separate from model_batch_size which
        controls how many files Whisper processes in parallel within each batch.
        Use ``None`` or <= 0 to process all segments in one batch.
    compress
        If ``True``, saves segment WAV files as 16-bit PCM to reduce disk usage
    """

    os.makedirs(output_dir, exist_ok=True)
    audio, sr = sf.read(audio_path)
    prefix = file_prefix or speaker

    # Step 1: Extract all segments to WAV files with progress bar
    segment_info = []  # List of segment metadata dicts

    for idx, seg in tqdm(
        segments.iterrows(), total=len(segments), desc="Extracting segments"
    ):
        start = float(seg["start_sec"])
        end = float(seg["end_sec"])
        row_speaker = seg.get("speaker", speaker)
        seg_filename = os.path.join(
            output_dir,
            f"{prefix}_seg_{idx}_{start:.2f}_{end:.2f}.wav",
        )
        txt_cache = seg_filename.replace(".wav", ".txt")

        start_samp = int(start * sr)
        end_samp = int(end * sr)
        segment_audio = audio[start_samp:end_samp]

        # Skip segments that are too short
        if len(segment_audio) < min_duration_samples:
            segment_info.append(
                {
                    "idx": idx,
                    "speaker": row_speaker,
                    "start_sec": start,
                    "end_sec": end,
                    "transcription": "",
                    "skip": True,
                }
            )
            continue

        # Save segment to WAV file (even if cached, for consistency)
        if not os.path.exists(seg_filename):
            _save_segment_wav(seg_filename, segment_audio, sr=sr, compress=compress)

        segment_info.append(
            {
                "idx": idx,
                "speaker": row_speaker,
                "start_sec": start,
                "end_sec": end,
                "seg_filename": seg_filename,
                "txt_cache": txt_cache,
                "skip": False,
            }
        )

    # Step 2: Transcribe segments in batches with progress bar
    valid_segments = [s for s in segment_info if not s["skip"]]

    # Create results list matching segment_info order
    transcriptions = {}  # Maps seg_filename -> transcription text

    # Determine maximum batch duration
    max_batch_duration = (
        float(batch_size) if batch_size and batch_size > 0 else float("inf")
    )

    batches: List[List[Dict[str, Any]]] = []
    current_batch: List[Dict[str, Any]] = []
    current_duration = 0.0

    for seg in valid_segments:
        seg_duration = float(seg["end_sec"] - seg["start_sec"])

        # If this segment alone exceeds the cap, process it alone
        if seg_duration > max_batch_duration:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_duration = 0.0
            batches.append([seg])
            continue

        # Start a new batch if adding would exceed the cap
        if current_batch and current_duration + seg_duration > max_batch_duration:
            batches.append(current_batch)
            current_batch = [seg]
            current_duration = seg_duration
        else:
            current_batch.append(seg)
            current_duration += seg_duration

    if current_batch:
        batches.append(current_batch)

    # Process batches
    for batch in tqdm(batches, desc=f"Transcribing {len(batches)} batches"):
        # Extract batch file paths and caches
        batch_files = [s["seg_filename"] for s in batch]
        batch_caches = [s["txt_cache"] for s in batch]

        # Transcribe batch
        batch_texts = _transcribe_batch(batch_files, batch_caches, model, cache)

        # Store results
        for seg_info, text in zip(batch, batch_texts):
            transcriptions[seg_info["seg_filename"]] = text

        # Clear GPU memory after each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Assemble final results in original order
    results = []
    for seg_info in segment_info:
        if seg_info["skip"]:
            results.append(
                {
                    "speaker": seg_info["speaker"],
                    "start_sec": seg_info["start_sec"],
                    "end_sec": seg_info["end_sec"],
                    "duration_sec": seg_info["end_sec"] - seg_info["start_sec"],
                    "transcription": "",
                }
            )
        else:
            results.append(
                {
                    "speaker": seg_info["speaker"],
                    "start_sec": seg_info["start_sec"],
                    "end_sec": seg_info["end_sec"],
                    "duration_sec": seg_info["end_sec"] - seg_info["start_sec"],
                    "transcription": transcriptions[seg_info["seg_filename"]],
                }
            )

    return results
