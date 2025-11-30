import csv
from pathlib import Path
from typing import Dict, List
import argparse

import clip
import numpy as np
import torch
from PIL import Image

from src.dataloader import VideoCaptionDataset, extract_frames_1fps
from src.captions import build_caption_set, CaptionSet
from src.transcripts import get_or_make_transcript
from typing import Optional

def load_clip(device: Optional[str] = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device


def score_video_with_captions(
    frames: List[np.ndarray],
    captions: CaptionSet,
    model,
    preprocess,
    device: str,
) -> Dict[str, Dict]:
    """
    Returns dict like:
      {
        'factual': {...},
        'counterfactual': {...}
      }
    """
    # frames -> tensor batch
    imgs = [preprocess(Image.fromarray(f)).unsqueeze(0) for f in frames]
    image_input = torch.cat(imgs, dim=0).to(device)

    with torch.no_grad():
        frame_feats = model.encode_image(image_input)

    frame_feats = frame_feats / frame_feats.norm(dim=-1, keepdim=True)  # [N,D]

    texts = [captions.factual, captions.counterfactual]
    text_tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        text_feats = model.encode_text(text_tokens)

    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)  # [2,D]

    keys = ["factual", "counterfactual"]
    results: Dict[str, Dict] = {}

    for i, key in enumerate(keys):
        sims = (frame_feats @ text_feats[i].reshape(-1, 1)).cpu().numpy().ravel()
        top_idx = int(np.argmax(sims))
        max_sim = float(sims[top_idx])

        results[key] = {
            "caption": getattr(captions, key),
            "top_idx": top_idx,
            "max_sim": max_sim,
            "curve": sims.tolist(),
        }

    return results


def run_batch(
    manifest_path: str = "data/manifest.csv",
    video_root: str = "data",
    out_csv: str = "results/batch_clip_results.csv",
):
    dataset = VideoCaptionDataset(manifest_path, video_root=video_root)
    print("num videos in dataset:", len(dataset))

    model, preprocess, device = load_clip()

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "video_id",
        "variant",          # factual or counterfactual
        "caption",
        "top_frame_idx",
        "max_similarity",
        "flip_vs_factual",  # TRUE/FALSE/NA
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ex in dataset.iter_videos():
          print(f"\nProcessing {ex.video_id} ...")

        # 1) Get transcript with Whisper (once per video)
          transcript_path = get_or_make_transcript(
              ex.video_id,
              ex.video_path,
              out_dir=Path("data/transcripts"),
            )
          transcript_text = transcript_path.read_text(encoding="utf-8").strip()
          if transcript_text:
              print("  transcript snippet:", transcript_text[:80])
          else:
              print("  transcript snippet: [EMPTY / no audio detected]")

        # 2) Extract frames for CLIP
          frames = extract_frames_1fps(ex.video_path)
          if not frames:
            print("  WARNING: no frames, skipping")
            continue

          capset = build_caption_set(ex.factual_caption)
          scored = score_video_with_captions(frames, capset, model, preprocess, device)

          factual_top = scored["factual"]["top_idx"]

          for variant in ["factual", "counterfactual"]:
                info = scored[variant]
                flip = "NA"
                if variant != "factual":
                    flip = str(info["top_idx"] != factual_top)

                writer.writerow({
                    "video_id": ex.video_id,
                    "variant": variant,
                    "caption": info["caption"],
                    "top_frame_idx": info["top_idx"],
                    "max_similarity": info["max_sim"],
                    "flip_vs_factual": flip,
                })

    print(f"\nDone. Wrote {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/manifest.csv",
        help="Path to manifest CSV",
    )
    parser.add_argument(
        "--video-root",
        type=str,
        default="data",
        help="Root directory for video_relpath in manifest",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/batch_clip_results.csv",
        help="Output CSV path",
    )

    args = parser.parse_args()

    run_batch(
        manifest_path=args.manifest,
        video_root=args.video_root,
        out_csv=args.out,
    )
