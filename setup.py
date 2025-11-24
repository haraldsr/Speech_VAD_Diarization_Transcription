"""Setup configuration for the speech_vad_diarization package."""

from pathlib import Path

from setuptools import setup

HERE = Path(__file__).parent
README = (HERE / "README.md").read_text(encoding="utf-8")


setup(
    name="speech_vad_diarization",
    version="0.1.0",
    description="Speech VAD, diarization and transcription pipeline",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Harald Skat-Rørdam, Hanlu He",
    author_email="harsk@dtu.dk",
    packages=["speech_vad_diarization"],
    package_dir={"speech_vad_diarization": "src"},
    python_requires=">=3.10",
    # Dependencies managed via requirements.txt and requirements-lock-uv.txt
)
