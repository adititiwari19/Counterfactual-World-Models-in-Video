import argparse, math
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch, clip

def extract_frames(video_file: str, fps: float, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_file}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / max(src_fps, 1.0)

    saved: List[Path] = []
    t, step = 0.0, 1.0 / max(fps, 1e-6)
    while t <= duration + 1e-6:
        frame_idx = int(round(t * src_fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        p = out_dir / f"frame_{int(t*1000):06d}.png"
        Image.fromarray(rgb).save(p)
        saved.append(p)
        t += step

    cap.release()
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--factual", required=True, help='Factual phrase (e.g., "picking up bag")')
    ap.add_argument("--counter", required=True, help='Counterfactual phrase (e.g., "putting down bag")')
    ap.add_argument("--fps", type=float, default=1.0, help="Sampling rate (default 1)")
    args = ap.parse_args()

    # Output folders
    video = Path(args.video)
    run_root = Path("data/outputs") / (video.stem + "_cf")
    frames_dir = run_root / "frames"
    results_dir = run_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1) Frames @ 1 FPS
    frames = extract_frames(str(video), args.fps, frames_dir)

    # 2) CLIP features (frames + two texts)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Frame features
    imgs = [preprocess(Image.open(p)).unsqueeze(0) for p in frames]
    image_input = torch.cat(imgs, 0).to(device)
    with torch.no_grad():
        frame_features = model.encode_image(image_input)
    frame_features = frame_features / frame_features.norm(dim=-1, keepdim=True)  # [N,D]

    # Text features (order: factual, counterfactual)
    phrases = [args.factual, args.counter]
    text_tokens = clip.tokenize(phrases).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)     # [2,D]

    # 3) EXACT slide formulas → two arrays [N,1]
    scores_factual = (frame_features @ text_features[0].reshape(-1,1)).cpu().numpy()
    scores_counter = (frame_features @ text_features[1].reshape(-1,1)).cpu().numpy()

    # Save optional CSVs for grading
    (results_dir / "scores_factual.csv").write_text(
        "frame_index,score\n" + "\n".join(f"{i},{float(s)}" for i,s in enumerate(scores_factual[:,0])),
        encoding="utf-8"
    )
    (results_dir / "scores_counter.csv").write_text(
        "frame_index,score\n" + "\n".join(f"{i},{float(s)}" for i,s in enumerate(scores_counter[:,0])),
        encoding="utf-8"
    )

    # Simple overlay plot
    plt.figure(figsize=(10,4))
    plt.plot(range(len(scores_factual)), scores_factual[:,0], label=args.factual)
    plt.plot(range(len(scores_counter)), scores_counter[:,0], label=args.counter)
    plt.xlabel("Frame index (1 fps)")
    plt.ylabel("Cosine similarity")
    plt.title("Factual vs Counterfactual")
    plt.legend()
    plot_path = results_dir / "similarity_counterfactual.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    # Quick console summary
    fi = int(np.argmax(scores_factual)); ci = int(np.argmax(scores_counter))
    print(f"[{args.factual}] peak @ frame {fi} score={float(scores_factual[fi,0]):.4f}")
    print(f"[{args.counter}] peak @ frame {ci} score={float(scores_counter[ci,0]):.4f}")
    print(f"Plot: {plot_path}")

if __name__ == "__main__":
    main()