import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from pathlib import Path
import cv2

device = "cuda" if torch.cuda.is_available else "cpu"
print("Using device:", device)

# Choose a model from https://huggingface.co/collections/facebook/dinov3
# Need to create token and agree to terms to use
#IMAGE_MODEL = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
IMAGE_MODEL = "facebook/dinov2-small"

processor = AutoImageProcessor.from_pretrained(IMAGE_MODEL)
model = AutoModel.from_pretrained(IMAGE_MODEL)


def run_video(video_file, wait=False):
    video_path = Path(__file__).parent.parent / "videos" / video_file  # File should be under folder videos/
    cap = cv2.VideoCapture(video_path)

    assert cap.isOpened(), "Video not found."
    delay_ms = int(1000 / cap.get(cv2.CAP_PROP_FPS))

    ret, frame = cap.read()
    while ret:
        if wait:
            cv2.imshow("Video", frame)

        yield frame

        if wait and cv2.waitKey(delay_ms) & 0xFF == ord("q"):
            break

        ret, frame = cap.read()
    
    cv2.destroyAllWindows()


def get_raw_embeddings(frames, step, mean_pool):
    with torch.no_grad():
        frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames][::step]
        inputs = processor(images=frames, return_tensors="pt")

        outputs = model(**inputs)

        if mean_pool:
            return torch.mean(outputs.last_hidden_state, dim=1)
        else:
            outputs.last_hidden_state


def get_temporal_embeddings(frames, step, discount_factor):
    with torch.no_grad():
        embeddings = get_raw_embeddings(frames=frames, step=step, mean_pool=True)

        accum = 0
        emb_accum = torch.zeros_like(embeddings[0])
        temporal_emb_list = []
        for emb in embeddings:
            accum = 1 + discount_factor * accum
            emb_accum = emb + discount_factor * emb_accum
            temporal_emb = emb_accum / accum
            temporal_emb_list.append(temporal_emb.cpu())
        
        return temporal_emb_list
        

def get_temporal_embeddings_from_video(video_file, step, discount_factor):
    frames = run_video(video_file=video_file)
    return get_temporal_embeddings(frames=frames, step=100, discount_factor=discount_factor)


def main():
    embs = get_temporal_embeddings_from_video("how_to_tie_a_tie.mp4", 10, 0.9)
    print(embs)


if __name__ == "__main__":
    main()