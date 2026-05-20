import sys
import torch
from PIL import Image
from torchvision import transforms
from anomalib.models import Patchcore


def inspect_single_image(image_path, checkpoint_path):
    print(f"\nInspecting: {image_path}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    model = Patchcore.load_from_checkpoint(
        checkpoint_path,
        weights_only=False
    )
    model.to(device)
    model.eval()

    ckpt = torch.load(checkpoint_path, weights_only=False)
    sd = ckpt["state_dict"]
    image_min = sd["post_processor.image_min"].item()
    image_max = sd["post_processor.image_max"].item()
    threshold = sd["post_processor._image_threshold"].item()

    inner = model.model
    inner.eval()

    with torch.no_grad():
        # Replicate forward pass exactly
        x = tensor.type(inner.memory_bank.dtype)
        output_size = x.shape[-2:]

        features = inner.feature_extractor(x)
        features = {
            layer: inner.feature_pooler(feat)
            for layer, feat in features.items()
        }
        embedding = inner.generate_embedding(features)

        batch_size, _, width, height = embedding.shape
        embedding = inner.reshape_embedding(embedding)

        patch_scores, locations = inner.nearest_neighbors(
            embedding=embedding,
            n_neighbors=1
        )
        patch_scores = patch_scores.reshape((batch_size, -1))
        locations = locations.reshape((batch_size, -1))

        raw_score = inner.compute_anomaly_score(
            patch_scores, locations, embedding
        ).item()

    norm_score = (raw_score - image_min) / (image_max - image_min)
    norm_score = max(0.0, min(1.0, norm_score))
    norm_threshold = (threshold - image_min) / (image_max - image_min)

    label = "DEFECTIVE" if raw_score > threshold else "GOOD"

    print(f"Result:        {label}")
    print(f"Raw score:     {raw_score:.4f}")
    print(f"Threshold:     {threshold:.4f}")
    print(f"Norm score:    {norm_score:.4f}")
    print(f"Norm threshold:{norm_threshold:.4f}")


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "./datasets/mvtec/metal_nut/test/scratch/000.png"
    checkpoint = "./results/Patchcore/metal_nut/latest/weights/lightning/model.ckpt"
    inspect_single_image(image_path, checkpoint)