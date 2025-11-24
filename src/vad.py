"""
Based on Conversational_speech_labeling_pipeline by Hanlu He.

https://github.com/hanlululu/Conversational_speech_labeling_pipeline
"""

import os
import warnings
from typing import Dict, List, Tuple

import torch
import torchaudio

# Enable TF32 for better performance on Ampere+ GPUs (RTX 30xx, A100, etc.)
# This provides significant speedup with minimal precision loss for deep learning
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Suppress std() degrees of freedom warning in pooling layers
# This occurs with very short audio segments and is harmless
warnings.filterwarnings(
    "ignore",
    message="std\\(\\): degrees of freedom is <= 0",
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
    Supports rVADfast, Silero, Whisper, and Pyannote.
    """

    def __init__(
        self,
        vad_type: str = "rvad",
        auth_token: str | None = None,
        device: str | None = None,
    ) -> None:
        """
        Initialize the VAD instance.

        Args:
            vad_type: Type of VAD to use ('rvad', 'silero', 'whisper', 'pyannote').
            auth_token: HuggingFace auth token (required for pyannote).
            device: Device to run models on ('cpu', 'cuda'). If None, auto-detect.
        """
        self.vad_type = vad_type
        if vad_type == "rvad":
            from rVADfast import rVADfast

            self.vad = rVADfast()
        elif vad_type == "silero":
            # Re-enable TF32 if disabled (already enabled globally at module level)
            self.model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=True
            )
            (self.get_speech_timestamps, _, self.read_audio, *_) = utils
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
                    use_auth_token=auth_token,
                )
            else:
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-community-1"
                )
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.pipeline.to(torch.device(device))
        else:
            raise ValueError(f"Unsupported vad_type: {vad_type}")

    def run_vad(
        self,
        wav_path: str,
        out_txt_path: str,
        min_duration: float = 0.07,
    ) -> str:
        """
        Run Voice Activity Detection on a WAV file and save intervals to text file.

        Args:
            wav_path: Path to the input WAV file.
            out_txt_path: Path to the output text file for VAD intervals.
            min_duration: Minimum duration for speech segments to be included.

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
            result = self.pipe(
                wav_path, return_timestamps=True, generate_kwargs={"language": "da"}
            )
            # Extract intervals from chunks
            chunks = result["chunks"]  # type: ignore
            intervals = []
            for chunk in chunks:
                timestamp = chunk["timestamp"]
                if isinstance(timestamp, (list, tuple)) and len(timestamp) == 2:
                    intervals.append((float(timestamp[0]), float(timestamp[1])))
        elif self.vad_type == "pyannote":
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
        from pyannote.audio.pipelines.utils.hook import ProgressHook

        if self.vad_type != "pyannote":
            raise ValueError(
                "run_diarization is only supported for vad_type='pyannote'"
            )

        basename = os.path.splitext(os.path.basename(wav_path))[0]

        print(f"Running Diarization on {wav_path} using {self.vad_type}...")
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

        output_paths = {}

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

            out_path = os.path.join(out_dir, speaker, f"{basename}_{speaker}_vad.txt")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            output_paths[speaker] = out_path

            with open(out_path, "w") as f:
                f.write("Start_Time(s)\tEnd_Time(s)\tAnnotation\n")
                for start, end in merged:
                    if (end - start) >= min_duration:
                        f.write(f"{start:.2f}\t{end:.2f}\tT\n")

        return output_paths
