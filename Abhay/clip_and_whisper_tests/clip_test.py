import torch
import open_clip
<<<<<<< HEAD
from pathlib import Path
from PIL import Image


=======
from PIL import Image

>>>>>>> b9465e82ac0f292937a7740fbba4629ddf62326f
def main():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    text = ["sea creature is dying", "water is blue", "the dolphin smiles in the water", "firey ground and mars pluto"]
    text_tokens = tokenizer(text)

<<<<<<< HEAD
    image = preprocess(Image.open(Path(__file__).parent / "image.png")).unsqueeze(0)
=======
    image = preprocess(Image.open("image.png")).unsqueeze(0)
>>>>>>> b9465e82ac0f292937a7740fbba4629ddf62326f

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
