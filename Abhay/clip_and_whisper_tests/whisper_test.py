import whisper
import open_clip
import torch
import os

def main():
    model = whisper.load_model("base")
    result = model.transcribe("Recording.m4a", word_timestamps=True)
    for segment in result["segments"]:
        print(segment["start"], segment["end"], segment["text"])
    
    


if __name__ == "__main__":
    main()