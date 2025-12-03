from PIL import Image
import cv2
import datasets
import numpy as np
import open_clip
from pathlib import Path
import requests
import torch
from tqdm import tqdm, trange
from zipfile import ZipFile


def download_videos(video_folder_name: str):
    video_folder = Path(video_folder_name)
    if video_folder.exists() and (video_folder / "marker.txt").exists():
        return
    
    Path("cache/").mkdir(exist_ok=True)
    videos_zip = Path("cache/MSRVTT_videos.zip")

    url = "https://huggingface.co/datasets/friedrichor/MSR-VTT/resolve/main/MSRVTT_Videos.zip"
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open("cache/MSRVTT_videos.zip", 'wb') as f:
            for chunk in tqdm(response.iter_content(chunk_size=8192), 2.04):
                f.write(chunk)
    else:
        print(f"Failed to download videos from link with status code: {response.status_code}")

    video_folder.mkdir(exist_ok=True)
    with ZipFile(videos_zip) as zipFile:
        zipFile.extractall(video_folder)
    
    (video_folder / "marker.txt").touch()

    


def unpack_dataset(video_folder_name: str):
    try:
        ds = datasets.load_dataset("friedrichor/MSR-VTT", "train_7k")
    except PermissionError:
        print("Use \"hf auth login\" on shell to get access.")
        exit()
    
    download_videos(video_folder_name)
    return ds


def save_videos_frames(video_folder: str, output_folder: str, step: int=1):
    if Path(output_folder).exists() and (Path(output_folder) / "marker.txt").exists():
        return
    
    Path(output_folder).mkdir(exist_ok=True)
    videos = Path(video_folder).glob("**/*.mp4")
    for vid in tqdm(list(videos), "Saving video frames"):
        frame_folder = Path(output_folder) / vid.name.split(".")[0]
        Path(frame_folder).mkdir(exist_ok=True)
        
        cap = cv2.VideoCapture()
        cap.open(str(vid))

        i = 0
        ret, frame = cap.read()
        while ret:
            if i % step != 0:
                continue
            
            cv2.imwrite(str(frame_folder / (str(i) + ".png")), frame)
            
            ret, frame = cap.read()
            i += 1

        cap.release()


def generate_raw_embeddings(model, preprocess, images, step, mean_pool):
    with torch.no_grad():
        images = torch.stack([preprocess(img) for img in images])
        image_embs = model.encode_image(images)
        return image_embs


def generate_temporal_embeddings(model, preprocess, frames, step, ema_factor):
    with torch.no_grad():
        embeddings = generate_raw_embeddings(model, preprocess, frames, step, mean_pool=True)
        
        moving_emb = embeddings[0]
        temporal_embs = []
        for emb in embeddings:
            temporal_embs.append(emb * ema_factor + moving_emb * (1 - ema_factor))
        
        return np.array(temporal_embs)
        

def generate_temporal_embeddings_from_folder(folder: Path, ema_factor: float) -> np.ndarray:
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    
    images = []
    i = 0
    for img in tqdm(list(folder.iterdir())):
        if img.name.endswith(".png"):
            images.append(Image.open(img))
            
    emb = generate_temporal_embeddings(model, preprocess, images, 1, ema_factor)
    emb_mmap = np.memmap(folder / "image_embeddings.npy", dtype=emb.dtype, shape=emb.shape, mode="w+")
    emb_mmap[:] = emb[:]

    return emb_mmap


def generate_caption_embeddings(folder: Path, text: str, tokenizer: open_clip.tokenizer):
    emb = tokenizer.tokenize(text)
    emb_mmap = np.memmap("text_embeddings.npy", dtype=emb.dtype, shape=emb.shape, mode="w+")
    emb_mmap[:] = emb[:]

    return emb_mmap


def main() -> None:
    video_folder_name = "MSRVTT_Videos"
    output_folder_name = "output"
    ds = unpack_dataset(video_folder_name)
    save_videos_frames(video_folder_name, output_folder_name)
    generate_temporal_embeddings_from_folder(Path("output/video0/"), 0.95)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    generate_caption_embeddings("video0/", ds[0]["caption"], tokenizer)


if __name__ == "__main__":
    main()