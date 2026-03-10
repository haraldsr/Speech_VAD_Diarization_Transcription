"""
Based on Conversational_speech_labeling_pipeline by Hanlu He.

https://github.com/hanlululu/Conversational_speech_labeling_pipeline
"""

import json
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchaudio
import wget

# Enable TF32 for better performance
# This provides significant speedup
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Suppress std() degrees of freedom warning in pooling layers
# This occurs with very short audio segments and is harmless
warnings.filterwarnings(
    "ignore",
    message="std\\(\\): degrees of freedom is <= 0",
    category=UserWarning,
)

# Suppress torchaudio deprecation warning about torchcodec
warnings.filterwarnings(
    "ignore",
    message=".*torchaudio.load_with_torchcodec.*",
    category=UserWarning,
)


def convert_to_labels(
    vad_timestamps: list[float], vad_labels: list[int]
) -> list[tuple[float, float]]:
    """
    Convert VAD timestamps and labels into speech intervals.

    Args:
        vad_timestamps: List of timestamps corresponding to VAD labels.
        vad_labels: List of binary labels (0 or 1) indicating speech.

    Returns:
        List of tuples (start_time, end_time) for speech intervals.
    """
    speech_intervals = []
    talking = False
    start_time = 0.0

    for i, label in enumerate(vad_labels):
        if label == 1 and not talking:
            start_time = vad_timestamps[i]
            talking = True
        elif label == 0 and talking:
            end_time = vad_timestamps[i]
            speech_intervals.append((start_time, end_time))
            talking = False

    # Handle case where speech continues to the end
    if talking:
        speech_intervals.append((start_time, vad_timestamps[-1]))

    return speech_intervals


class SpeechActivityDetector:
    """
    Wrapper class for Voice Activity Detection and Diarization.
    Supports rVADfast, Silero, Whisper, Pyannote, and NeMo diarization.
    """

    def __init__(
        self,
        vad_type: str = "rvad",
        auth_token: Optional[str] = None,
        device: Optional[str] = None,
        rvad_threshold: float = 0.4,
    ) -> None:
        """
        Initialize the VAD instance.

        Args:
            vad_type: Type of VAD to use ('rvad', 'silero', 'whisper',
                'pyannote', 'nemo').
            auth_token: HuggingFace auth token (required for pyannote).
            device: Device to run models on ('cpu', 'cuda'). If None, auto-detect.
        """
        self.vad_type = vad_type
        if vad_type == "rvad":
            from rVADfast import rVADfast

            self.vad = rVADfast(vad_threshold=rvad_threshold)
        elif vad_type == "silero":
            # Re-enable TF32 if disabled (already enabled globally at module level)
            self.model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=True
            )
            self.get_speech_timestamps, _, self.read_audio, *_ = utils
        elif vad_type == "whisper":
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

            # TF32 already enabled globally at module level

            torch_device = torch.device(
                device if device else ("cuda" if torch.cuda.is_available() else "cpu")
            )
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            model_id = "openai/whisper-large-v3"
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, dtype=torch_dtype
            )
            model.to(torch_device)

            processor = AutoProcessor.from_pretrained(model_id)

            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                batch_size=1,
                dtype=torch_dtype,
                device=torch_device,
                chunk_length_s=30,
            )
        elif vad_type == "pyannote":
            from pyannote.audio import Pipeline

            # TF32 already enabled globally at module level

            if auth_token is not None:
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-community-1",
                    token=auth_token,
                )
            else:
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-community-1"
                )
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.pipeline is not None:
                self.pipeline.to(torch.device(device))
        elif vad_type == "nemo":
            # NeMo config will be downloaded on-demand in run_diarization()
            pass
        else:
            raise ValueError(f"Unsupported vad_type: {vad_type}")

    @staticmethod
    def _normalize_speaker_label(raw_label: str) -> str:
        """Normalize diarization speaker labels to P1, P2, ... format."""
        # Check if already in P1, P2 format (P followed by digits)
        if raw_label.startswith("P") and raw_label[1:].isdigit():
            return raw_label  # Already normalized, return as-is

        # Handle lowercase format: speaker_0, speaker_1, etc.
        if raw_label.startswith("speaker_"):
            try:
                # Extract number from last part (e.g., "speaker_0" -> "0")
                # Add 1 to convert 0-indexed to 1-indexed (0 -> P1, 1 -> P2)
                return f"P{int(raw_label.split('_')[-1]) + 1}"
            except ValueError:
                # If parsing fails, return original label unchanged
                return raw_label

        # Handle uppercase format: SPEAKER_0, SPEAKER_1, etc. (typical RTTM format)
        if raw_label.startswith("SPEAKER_"):
            try:
                # Extract number from last part (e.g., "SPEAKER_0" -> "0")
                # Add 1 to convert 0-indexed to 1-indexed (0 -> P1, 1 -> P2)
                return f"P{int(raw_label.split('_')[-1]) + 1}"
            except ValueError:
                # If parsing fails, return original label unchanged
                return raw_label

        # If label doesn't match any known format, return unchanged
        return raw_label

    @staticmethod
    def _write_vad_from_rttm(
        rttm_path: str,
        out_dir: str,
        min_duration: float = 0.07,
    ) -> Dict[str, str]:
        """Convert RTTM to per-speaker VAD txt files in expected format."""
        # Dictionary to collect all speech intervals for each speaker
        # Structure: {"P1": [(start1, end1), (start2, end2), ...], "P2": [...], ...}
        speaker_intervals: Dict[str, List[Tuple[float, float]]] = {}

        # Read RTTM file line by line
        with open(rttm_path, "r") as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and non-SPEAKER entries
                if not line or not line.startswith("SPEAKER"):
                    continue

                # Split line by whitespace to parse RTTM format
                # Format: SPEAKER <filename> <channel> <start_time> <duration>
                #         <NA> <NA> <speaker_id> <confidence>
                parts = line.split()

                # Extract timing information from RTTM fields
                start_time = float(parts[3])  # Field 4: start time in seconds
                duration = float(parts[4])  # Field 5: duration in seconds
                end_time = start_time + duration  # Calculate end time

                # Extract and normalize speaker ID from RTTM (field 8)
                # e.g., "SPEAKER_0" -> "P1", "SPEAKER_1" -> "P2"
                speaker_id = SpeechActivityDetector._normalize_speaker_label(parts[7])

                # Add this interval to the speaker's list of intervals
                # setdefault creates an empty list if this speaker doesn't exist yet
                speaker_intervals.setdefault(speaker_id, []).append(
                    (start_time, end_time)
                )

        # Initialize output dictionary and extract base filename
        output_paths: Dict[str, str] = {}
        basename = os.path.splitext(os.path.basename(rttm_path))[0]

        # Process intervals for each speaker
        for speaker, intervals in speaker_intervals.items():
            # Sort intervals by start time (ascending order)
            intervals.sort()

            # Merge overlapping or adjacent intervals to avoid redundant segments
            merged: List[Tuple[float, float]] = []
            if intervals:
                # Start with the first interval
                curr_start, curr_end = intervals[0]

                # Process remaining intervals
                for next_start, next_end in intervals[1:]:
                    # Check if next interval overlaps or is adjacent to current
                    if next_start <= curr_end:
                        # Merge: extend current interval to cover both intervals
                        curr_end = max(curr_end, next_end)
                    else:
                        # No overlap: save current and start a new one
                        merged.append((curr_start, curr_end))
                        curr_start, curr_end = next_start, next_end

                # Don't forget to add the final interval
                merged.append((curr_start, curr_end))

            # Create speaker-specific subdirectory (e.g., "outputs/P1")
            speaker_dir = os.path.join(out_dir, speaker)
            os.makedirs(speaker_dir, exist_ok=True)

            # Build output file path (e.g., "outputs/P1/conv_123_P1_vad.txt")
            out_path = os.path.join(speaker_dir, f"{basename}_{speaker}_vad.txt")

            # Write VAD intervals to text file
            with open(out_path, "w") as f:
                # Write header row with tab-separated column names
                f.write("Start_Time(s)\tEnd_Time(s)\tAnnotation\n")

                # Write each merged interval as a row
                for start, end in merged:
                    # Only include segments that meet minimum duration requirement
                    if (end - start) >= min_duration:
                        # Format: start_time, end_time (2 decimals),
                        # "T" (means "Talking")
                        f.write(f"{start:.2f}\t{end:.2f}\tT\n")

            # Store the output file path in the dictionary for return
            output_paths[speaker] = out_path

        # Return mapping of speaker IDs to their output file paths
        return output_paths

    def run_vad(
        self, wav_path: str, out_txt_path: str, min_duration: float = 0.07
    ) -> str:
        """
        Run Voice Activity Detection on a WAV file and save intervals to text file.

        Args:
            wav_path: Path to the input WAV file.
            out_txt_path: Path to the output text file for VAD intervals.
            min_duration: Minimum duration for speech segments to be included.
            vad_threshold: Threshold for VAD decision.

        Returns:
            Path to the output text file.
        """
        # Load audio
        # Only load with torchaudio if not using pyannote (pyannote loads internally)
        if self.vad_type != "pyannote":
            signal, fs = torchaudio.load(wav_path)

        print(f"Running VAD on {wav_path} using {self.vad_type}...")

        intervals = []

        if self.vad_type == "rvad":
            signal_np = signal.numpy().flatten()
            # Run VAD
            vad_labels, vad_timestamps = self.vad(signal_np, fs)
            # Convert to intervals
            intervals = convert_to_labels(vad_timestamps, vad_labels)
        elif self.vad_type == "silero":
            # Run Silero VAD
            speech_timestamps = self.get_speech_timestamps(
                signal, self.model, sampling_rate=fs, return_seconds=True
            )
            # Convert to intervals
            intervals = [(d["start"], d["end"]) for d in speech_timestamps]
        elif self.vad_type == "whisper":
            # Run Whisper ASR with timestamps
            result: list[dict[str, Any]] = self.pipe(
                wav_path, return_timestamps=True, generate_kwargs={"language": "da"}
            )
            # Extract intervals from chunks
            chunks = result[0]["chunks"]
            intervals = []
            for chunk in chunks:
                timestamp = chunk["timestamp"]
                if isinstance(timestamp, (list, tuple)) and len(timestamp) == 2:
                    intervals.append((float(timestamp[0]), float(timestamp[1])))
        elif self.vad_type == "pyannote":
            assert self.pipeline is not None, "Pipeline not initialized"
            diarization = self.pipeline(wav_path)
            raw_intervals = []
            for turn, _, _ in diarization.itertracks(yield_label=True):
                raw_intervals.append((turn.start, turn.end))

            # Merge overlapping intervals
            raw_intervals.sort()
            if raw_intervals:
                curr_start, curr_end = raw_intervals[0]
                for next_start, next_end in raw_intervals[1:]:
                    if next_start <= curr_end:
                        curr_end = max(curr_end, next_end)
                    else:
                        intervals.append((curr_start, curr_end))
                        curr_start, curr_end = next_start, next_end
                intervals.append((curr_start, curr_end))
        elif self.vad_type == "nemo":
            raise ValueError(
                "NeMo diarization is only supported via run_diarization()."
            )

        # Write to file, filtering by min_duration
        with open(out_txt_path, "w") as f:
            f.write("Start_Time(s)\tEnd_Time(s)\tAnnotation\n")
            for start, end in intervals:
                if (end - start) >= min_duration:
                    f.write(f"{start:.2f}\t{end:.2f}\tT\n")

        return out_txt_path

    def run_diarization(
        self,
        wav_path: str,
        out_dir: str,
        min_duration: float = 0.07,
    ) -> Dict[str, str]:
        """
        Run Diarization on a WAV file and save separate VAD files for each speaker.

        Args:
            wav_path: Path to the input WAV file.
            out_dir: Directory to save the output text files.
            min_duration: Minimum duration for speech segments to be included.

        Returns:
            Dictionary mapping speaker labels (e.g. 'SPEAKER_00') to output file paths.
        """
        basename = os.path.splitext(os.path.basename(wav_path))[0]

        if self.vad_type == "pyannote":
            from pyannote.audio.pipelines.utils.hook import ProgressHook

            print(f"Running Diarization on {wav_path} using {self.vad_type}...")
            assert self.pipeline is not None, "Pipeline not initialized"
            with ProgressHook() as hook:
                diarization = self.pipeline(wav_path, hook=hook)

            os.makedirs(out_dir, exist_ok=True)

            speaker_intervals: Dict[str, List[Tuple[float, float]]] = {}

            # Handle DiarizeOutput object (has speaker_diarization attribute)
            if hasattr(diarization, "speaker_diarization"):
                # DiarizeOutput object (iterable of (turn, speaker))
                for turn, speaker in diarization.speaker_diarization:
                    # Map SPEAKER_00 to P1, SPEAKER_01 to P2, etc.
                    speaker_id = f"P{int(speaker.split('_')[1]) + 1}"
                    if speaker_id not in speaker_intervals:
                        speaker_intervals[speaker_id] = []
                    speaker_intervals[speaker_id].append((turn.start, turn.end))
            else:
                # Standard pyannote Annotation
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    # Map SPEAKER_00 to P1, SPEAKER_01 to P2, etc.
                    speaker_id = f"P{int(speaker.split('_')[1]) + 1}"
                    if speaker_id not in speaker_intervals:
                        speaker_intervals[speaker_id] = []
                    speaker_intervals[speaker_id].append((turn.start, turn.end))

            output_paths: Dict[str, str] = {}

            for speaker, intervals in speaker_intervals.items():
                # Merge intervals for each speaker
                intervals.sort()
                merged = []
                if intervals:
                    curr_start, curr_end = intervals[0]
                    for next_start, next_end in intervals[1:]:
                        if next_start <= curr_end:
                            curr_end = max(curr_end, next_end)
                        else:
                            merged.append((curr_start, curr_end))
                            curr_start, curr_end = next_start, next_end
                    merged.append((curr_start, curr_end))

                out_path = os.path.join(
                    out_dir, speaker, f"{basename}_{speaker}_vad.txt"
                )
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                output_paths[speaker] = out_path

                with open(out_path, "w") as f:
                    f.write("Start_Time(s)\tEnd_Time(s)\tAnnotation\n")
                    for start, end in merged:
                        if (end - start) >= min_duration:
                            f.write(f"{start:.2f}\t{end:.2f}\tT\n")

            return output_paths

        if self.vad_type == "nemo":
            from nemo.collections.asr.models import ClusteringDiarizer
            from omegaconf import OmegaConf

            os.makedirs(out_dir, exist_ok=True)

            # Download NeMo config if it doesn't exist
            config_file_name = "diar_infer_meeting.yaml"
            config_path = os.path.join(out_dir, config_file_name)

            if not os.path.exists(config_path):
                print(f"Downloading NeMo config: {config_file_name}")
                config_url = (
                    "https://raw.githubusercontent.com/NVIDIA/NeMo/main/"
                    "examples/speaker_tasks/diarization/conf/inference/"
                    f"{config_file_name}"
                )
                wget.download(config_url, out_dir)
                print()  # newline after wget progress
            else:
                print(f"Using existing NeMo config: {config_path}")

            # Use subdirectory for NeMo working directory
            # (hardcoded name; deleted below)
            nemo_work_dir = os.path.join(out_dir, "nemo_work")
            os.makedirs(nemo_work_dir, exist_ok=True)

            manifest_path = os.path.join(nemo_work_dir, "input_manifest.json")
            meta = {
                "audio_filepath": wav_path,  # Path to input audio file
                "offset": 0,  # Start time in seconds (0 = beginning of file)
                "duration": None,  # Duration in seconds (None = entire file)
                "label": "infer",  # Mode: 'infer' for diarization inference
                "text": "-",  # Transcript text (not needed for diarization)
                "num_speakers": None,  # Number of speakers (None = auto-detect)
                "rttm_filepath": None,  # Reference RTTM file (None for inference)
                "uem_filepath": None,  # Unpartitioned evaluation map file (optional)
            }
            with open(manifest_path, "w") as fp:
                json.dump(meta, fp)
                fp.write("\n")

            cfg = OmegaConf.load(config_path)
            cfg.diarizer.manifest_filepath = manifest_path
            cfg.diarizer.out_dir = nemo_work_dir

            print(f"Running Diarization on {wav_path} using NeMo...")
            diarizer = ClusteringDiarizer(cfg=cfg)
            diarizer.diarize()

            try:
                # Read RTTM file from NeMo work directory
                rttm_path = os.path.join(
                    nemo_work_dir, "pred_rttms", f"{basename}.rttm"
                )
                if os.path.exists(rttm_path):
                    output_paths = self._write_vad_from_rttm(
                        rttm_path=rttm_path,
                        out_dir=out_dir,
                        min_duration=min_duration,
                    )
                else:
                    raise FileNotFoundError(
                        f"Expected RTTM output not found: {rttm_path}"
                    )
            finally:
                # Clean up NeMo working directory
                def remove_tree(path: str) -> None:
                    """Recursively remove directory tree using only os module."""
                    if os.path.isdir(path):
                        for item in os.listdir(path):
                            item_path = os.path.join(path, item)
                            remove_tree(item_path)
                        os.rmdir(path)
                    elif os.path.exists(path):
                        os.remove(path)

                try:
                    remove_tree(nemo_work_dir)
                    print("✓ Cleaned up NeMo working files")
                except Exception as e:
                    print(f"Warning: Could not clean up NeMo working directory: {e}")

            return output_paths

        raise ValueError(
            "run_diarization is only supported for vad_type='pyannote' or 'nemo'"
        )
