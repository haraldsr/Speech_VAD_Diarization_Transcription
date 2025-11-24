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
from tqdm.auto import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.pipelines.base import Pipeline

# Set PyTorch CUDA memory configuration for better fragmentation handling
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


@dataclass
class TransformersASRModel:
    pipeline: Pipeline
    language: Optional[str]


def load_whisper_model(
    transciption_model_name: str = "openai/whisper-large-v3",
    device: str = "cpu",
    language: Optional[str] = "da",
    cache_dir: Optional[str] = None,
    transformers_batch_size: int = 100,
) -> TransformersASRModel:
    """Initialise and return a Whisper ASR model via the Transformers pipeline.

    Parameters
    ----------
    transciption_model_name
        Model identifier (e.g., 'openai/whisper-large-v3')
    device
        'cpu' or 'cuda' for GPU inference
    language
        Target language code (e.g., 'da' for Danish)
    cache_dir
        Optional directory for model caching
    transformers_batch_size
        Maximum number of clips the transformers pipeline should batch internally.

    Returns
    -------
    TransformersASRModel
        Wrapper containing the configured transformers pipeline
    """

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

    # Load model and processor
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        transciption_model_name,
        dtype=torch_dtype,
        cache_dir=cache_dir,
    )
    model.to(torch_device)

    processor = AutoProcessor.from_pretrained(
        transciption_model_name, cache_dir=cache_dir
    )

    # Create pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        batch_size=transformers_batch_size,
        dtype=torch_dtype,
        device=torch_device,
        chunk_length_s=30.0,
    )

    return TransformersASRModel(
        pipeline=pipe,
        language=language,
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
            # Load from cache
            with open(txt_cache, "r", encoding="utf-8") as cache_file:
                results[i] = cache_file.read().strip()
        else:
            # Needs transcription
            files_to_transcribe.append(seg_file)
            file_indices.append(i)

    # If no files need transcription, return cached results
    if not files_to_transcribe:
        return results

    pipe = model.pipeline
    language = model.language

    # Collect metadata for logging without keeping large arrays in memory
    durations = []
    for seg_file in files_to_transcribe:
        try:
            info = sf.info(seg_file)
            durations.append(float(info.duration))
        except RuntimeError:
            durations.append(0.0)

    total_duration = sum(durations)

    generate_kwargs: Dict[str, Any] = {
        "task": "transcribe"
    }  # , "return_timestamps": True
    if language:
        generate_kwargs["language"] = language

    # Try to transcribe batch, if OOM or batching error then split and retry
    try:
        batch_results = pipe(
            files_to_transcribe,
            return_timestamps=True,
            generate_kwargs=generate_kwargs,
            batch_size=len(files_to_transcribe),
        )

        if isinstance(batch_results, dict):
            batch_results = [batch_results]

        for batch_idx, result in zip(file_indices, batch_results):
            text = result.get("text", "").strip()
            results[batch_idx] = text

            cache_path = batch_caches[batch_idx]
            with open(cache_path, "w", encoding="utf-8") as cache_file:
                cache_file.write(text)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except (RuntimeError, ValueError) as e:
        error_msg = str(e).lower()
        if (
            "out of memory" in error_msg
            or "cuda" in error_msg
            or "different keys" in error_msg
        ):
            # OOM or batching error - process individually with memory management
            print(
                f"\n⚠️  Error with batch ({len(files_to_transcribe)} files, \
                {total_duration:.1f}s total): {e}"
            )
            print("    → Splitting into individual transcriptions...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            for batch_idx, seg_file in zip(file_indices, files_to_transcribe):
                batch_result = pipe(
                    seg_file, generate_kwargs=generate_kwargs, return_timestamps=True
                )
                text = batch_result.get("text", "").strip()
                results[batch_idx] = text

                cache_path = batch_caches[batch_idx]
                with open(cache_path, "w", encoding="utf-8") as cache_file:
                    cache_file.write(text)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        else:
            raise  # Re-raise non-OOM/batching errors

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
    batch_size: float | None = 240.0,
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
        Maximum total audio duration (in seconds) to process in parallel.
        Default: 240 seconds. The pipeline will batch audio segments up to
        this total duration. Use ``None`` or <= 0 to process all remaining
        segments in one go.
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
        turn_type = seg.get("Turn_Type", "T")  # Preserve Turn_Type if present
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
                    "turn_type": turn_type,
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
                "turn_type": turn_type,
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
                    "turn_type": seg_info.get("turn_type", "T"),
                    "transcription": "",
                }
            )
        else:
            results.append(
                {
                    "speaker": seg_info["speaker"],
                    "start_sec": seg_info["start_sec"],
                    "end_sec": seg_info["end_sec"],
                    "turn_type": seg_info.get("turn_type", "T"),
                    "transcription": transcriptions[seg_info["seg_filename"]],
                }
            )

    return results
