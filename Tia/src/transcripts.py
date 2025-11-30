from pathlib import Path
from typing import Optional

import whisper

_model: Optional[whisper.Whisper] = None


def get_model() -> whisper.Whisper:
    """Lazy-load a single Whisper model and reuse it."""
    global _model
    if _model is None:
        _model = whisper.load_model("small")
    return _model


def get_or_make_transcript(video_id: str, video_path: Path, out_dir: Path) -> Path:
    """
    If transcript for this video already exists, return it.
    Otherwise run Whisper, save <video_id>.txt, and return the path.

    If Whisper/ffmpeg fails (e.g., video has no audio track),
    we just save an empty transcript instead of crashing.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_id}.txt"

    if out_path.exists():
        return out_path

    model = get_model()

    try:
        result = model.transcribe(str(video_path))
        text = result.get("text", "").strip()
    except Exception as e:
        print(f"  Whisper failed on {video_path}: {e}")
        text = ""  # no transcript / no audio

    out_path.write_text(text, encoding="utf-8")
    print(f"  saved transcript to {out_path}")
    return out_path
