import whisper
import open_clip
import torch
<<<<<<< HEAD
from pathlib import Path

def main():
    model = whisper.load_model("base")
    result = model.transcribe(str(Path(__file__).parent / "Recording.m4a"), word_timestamps=True)
    for segment in result["segments"]:
        print(segment["start"], segment["end"], segment["text"])
=======
import os

def main():
    model = whisper.load_model("base")
    result = model.transcribe("Recording.m4a", word_timestamps=True)
    for segment in result["segments"]:
        print(segment["start"], segment["end"], segment["text"])
    
    
>>>>>>> b9465e82ac0f292937a7740fbba4629ddf62326f


if __name__ == "__main__":
    main()