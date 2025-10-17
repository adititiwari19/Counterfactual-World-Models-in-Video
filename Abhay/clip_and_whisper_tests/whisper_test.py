import whisper
import open_clip
import torch
from pathlib import Path

def main():
    model = whisper.load_model("base")
    result = model.transcribe(str(Path(__file__).parent / "Recording.m4a"), word_timestamps=True)
    for segment in result["segments"]:
        print(segment["start"], segment["end"], segment["text"])


if __name__ == "__main__":
    main()