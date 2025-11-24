"""CLI helper for the conversation VAD labeler package."""

from __future__ import annotations

from pathlib import Path

from speech_vad_diarization import process_conversation


def _default_example_inputs() -> dict[str, str]:
    base = Path("examples/recordings")
    vad_type = "rvad"
    return (
        {
            "P1": str(base / "EXP9_None_p1_trial2.wav"),
            "P2": str(base / "EXP9_None_p2_trial2.wav"),
        },
        vad_type,
    )


def _diarize_example_inputs() -> str:
    base = Path("examples/coral")
    vad_type = "pyannote"
    return (str(base / "conv_0cbf895a2078529eb4a9d8b212e710c9.wav"), vad_type)


def main() -> None:
    speakers_audio, vad_type = _diarize_example_inputs()
    output_directory = "outputs/diarize"

    process_conversation(
        speakers_audio=speakers_audio,
        output_dir=output_directory,
        vad_type=vad_type,
        energy_margin_db=20.0,
        whisper_device="auto",
        interactive_energy_filter=False,
        batch_size=30.0,  # sec
        transciption_model_name="CoRal-project/roest-whisper-large-v1",
        skip_transcription_if_exists=False,
    )


if __name__ == "__main__":
    main()
