from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import cv2
import numpy as np
import pandas as pd


@dataclass
class VideoExample:
    video_id: str
    video_path: Path
    factual_caption: str


class VideoCaptionDataset:
    """
    Reads a CSV manifest with columns:
      - video_id
      - video_relpath
      - factual_caption
    and lets you iterate over videos.
    """

    def __init__(self, manifest_path: str, video_root: str = ".") -> None:
        self.manifest_path = Path(manifest_path)
        self.video_root = Path(video_root)

        df = pd.read_csv(self.manifest_path)
        required = {"video_id", "video_relpath", "factual_caption"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Manifest missing columns: {missing}")

        self._df = df

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> VideoExample:
        row = self._df.iloc[idx]
        video_path = self.video_root / row["video_relpath"]
        return VideoExample(
            video_id=str(row["video_id"]),
            video_path=video_path,
            factual_caption=str(row["factual_caption"]),
        )

    def iter_videos(self) -> Iterator[VideoExample]:
        for i in range(len(self)):
            yield self[i]


def extract_frames_1fps(
    video_path: Path,
    save_dir: Optional[Path] = None,
) -> List[np.ndarray]:
    """
    Extract ~1 frame per second from the video.
    If save_dir is given, also saves PNGs there.
    Returns frames as RGB numpy arrays.
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / max(fps, 1.0)

    frames: List[np.ndarray] = []

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    for sec in range(int(duration) + 1):
        idx = int(round(sec * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

        if save_dir is not None:
            out_path = save_dir / f"frame_{sec:04d}.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    cap.release()
    return frames
