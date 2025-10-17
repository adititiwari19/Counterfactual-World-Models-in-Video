import torch
import open_clip
from PIL import Image

def main():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    text = ["sea creature is dying", "water is blue", "the dolphin smiles in the water", "firey ground and mars pluto"]
    text_tokens = tokenizer(text)

    image = preprocess(Image.open("image.png")).unsqueeze(0)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).squeeze(0)

    for t, score in zip(text, similarity):
        print(f"{t}: {score.item():.4f}")

if __name__ == "__main__":
    main()
