import argparse
import os
import math
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torch
import clip


HAS_WHISPER = True
try:
    import whisper
except Exception:
    HAS_WHISPER = False

def extract_fps1_frames(video_file: str, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_file}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / max(fps, 1.0)

    saved = []
    for sec in range(int(math.floor(duration_sec)) + 1):
        frame_idx = int(round(sec * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        out_path = out_dir / f"frame_{sec:03d}.png"
        img.save(out_path)
        saved.append(out_path)

    cap.release()
    return saved

def maybe_transcribe(video_file: str, model_size: str = "small") -> Optional[str]:
    if not HAS_WHISPER:
        print("Whisper not installed or FFmpeg missing. Skipping transcript.")
        return None
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        asr_model = whisper.load_model(model_size, device=device)
        result = asr_model.transcribe(video_file)
        text = (result.get("text") or "").strip()
        if text:
            print("Transcript:", text)
            return text
        print("No speech detected.")
        return None
    except Exception as e:
        print(f"Whisper failed: {e}. Continuing without transcript.")
        return None

def run_clip_similarity(frames: list[Path], keyword: str, out_dir: Path) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # encode images
    images = [preprocess(Image.open(p)).unsqueeze(0) for p in frames]
    image_input = torch.cat(images, dim=0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)

    # encode text
    text_tokens = clip.tokenize([keyword]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)

    # normalize
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    sims = (image_features @ text_features.T).squeeze(1).detach().cpu().numpy()

    # plot
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(sims)), sims)
    plt.xlabel("Frame index (1 fps)")
    plt.ylabel("CLIP similarity")
    plt.title(f'Keyword: "{keyword}"')
    chart_path = out_dir / "similarity_bar_chart.png"
    plt.savefig(chart_path, bbox_inches="tight")
    plt.close()

    top_idx = int(np.argmax(sims))
    top_frame = frames[top_idx]
    top_frame_copy = out_dir / f"top_frame_{top_idx:03d}.png"
    Image.open(top_frame).save(top_frame_copy)

    # print summary
    print(f"Top frame index: {top_idx}")
    print(f"Top frame file: {top_frame_copy.name}")
    print(f"Chart saved to: {chart_path.name}")

    # also save a CSV of scores
    csv_path = out_dir / "similarities.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("frame_index,frame_file,similarity\n")
        for i, (p, s) in enumerate(zip(frames, sims)):
            f.write(f"{i},{p.name},{float(s)}\n")
    print(f"Scores saved to: {csv_path.name}")

    return {
        "top_index": top_idx,
        "top_frame": str(top_frame_copy),
        "chart": str(chart_path),
        "csv": str(csv_path),
    }



def main():
    parser = argparse.ArgumentParser(description="Starter experiment: frames + CLIP similarity")
    parser.add_argument("--video", required=True, help="Path to input video (mp4, mov)")
    parser.add_argument("--keyword", help="Single keyword/phrase (backward compatible)")
    parser.add_argument("--keywords", nargs="+", help="One or more keywords/phrases for comparison")
    parser.add_argument("--whisper", action="store_true", help="Try to transcribe audio to pick a keyword")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    run_root = Path("data/outputs") / video_path.stem
    frames_dir = run_root / "frames"
    results_dir = run_root / "results"

    print("Extracting frames at 1 fps...")
    frames = extract_fps1_frames(str(video_path), frames_dir)
    print(f"Saved {len(frames)} frames to {frames_dir}")

    keyword = args.keyword
    if not keyword and args.whisper:
        text = maybe_transcribe(str(video_path))
        # simple heuristic: pick a noun-like word by hand later if this is empty
        if text:
            print("Pick a keyword that appears in the transcript above, then re-run with --keyword <word>.")
            return

    if not keyword:
        print("No keyword provided. Re-run like this:")
        print(f'python src/run_experiment.py --video "{video_path}" --keyword hammer')
        return

    print(f'Computing CLIP similarity for keyword "{keyword}"...')
    _ = run_clip_similarity(frames, keyword, results_dir)
    print("Done.")

if __name__ == "__main__":
    main()
