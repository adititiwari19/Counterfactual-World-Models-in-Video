import csv
from pathlib import Path

def build_ucf101_manifest(
    root="data/input/UCF101_subset",
    out_csv="data/manifest_ucf101.csv",
    limit=50,
):
    root = Path(root)

    # Accept common video extensions
    exts = {".mp4", ".avi", ".mov", ".mkv"}

    video_paths = []

    # Look inside train, test, val
    for split in ["train", "test", "val"]:
        split_dir = root / split
        if not split_dir.exists():
            continue

        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue

            for vid in class_dir.iterdir():
                if vid.suffix.lower() in exts:
                    video_paths.append(vid)

    print(f"Found {len(video_paths)} total videos.")

    # Limit how many we use
    video_paths = video_paths[:limit]
    print(f"Using first {len(video_paths)} videos.")

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "video_relpath", "factual_caption"])

        for p in video_paths:
            video_id = p.stem
            rel_path = str(p.relative_to("data"))  # path relative to project root
            label = p.parent.name.replace("_", " ").lower()
            caption = f"a person doing {label}"
            writer.writerow([video_id, rel_path, caption])

    print(f"Done. Wrote {out_csv}")


if __name__ == "__main__":
    build_ucf101_manifest()
