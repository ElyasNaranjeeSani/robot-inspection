from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from PIL import Image
from torchvision import transforms
from anomalib.models import Patchcore
import io
import base64
import numpy as np
import cv2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

from huggingface_hub import hf_hub_download
import os

HF_REPO = "elyasn/robot-inspection-model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download model from Hugging Face at startup
print("Downloading model from Hugging Face...")
CHECKPOINT = hf_hub_download(repo_id=HF_REPO, filename="model.ckpt")
print(f"Model downloaded to {CHECKPOINT}")

# Load model once at startup
print("Loading model...")
model = Patchcore.load_from_checkpoint(CHECKPOINT, weights_only=False)
model.to(DEVICE)
model.eval()

ckpt = torch.load(CHECKPOINT, weights_only=False, map_location=DEVICE)
sd = ckpt["state_dict"]
IMAGE_MIN = sd["post_processor.image_min"].item()
IMAGE_MAX = sd["post_processor.image_max"].item()
THRESHOLD = sd["post_processor._image_threshold"].item()
print("Model loaded.")


def run_inference(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    tensor = transform(image).unsqueeze(0).to(DEVICE)
    inner = model.model

    with torch.no_grad():
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
            embedding=embedding, n_neighbors=1
        )
        patch_scores = patch_scores.reshape((batch_size, -1))
        locations = locations.reshape((batch_size, -1))
        raw_score = inner.compute_anomaly_score(
            patch_scores, locations, embedding
        ).item()

        # Generate heatmap
        patch_map = patch_scores.reshape((batch_size, 1, width, height))
        anomaly_map = inner.anomaly_map_generator(patch_map, output_size)
        anomaly_map = anomaly_map.squeeze().cpu().numpy()

    # Normalise score
    norm_score = (raw_score - IMAGE_MIN) / (IMAGE_MAX - IMAGE_MIN)
    norm_score = max(0.0, min(1.0, norm_score))
    is_defective = raw_score > THRESHOLD

    # Create heatmap overlay
    img_array = np.array(image.resize((256, 256)))
    anomaly_norm = ((anomaly_map - anomaly_map.min()) /
                    (anomaly_map.max() - anomaly_map.min() + 1e-8) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(anomaly_norm, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_array, 0.6, heatmap, 0.4, 0)

    # Convert overlay to base64
    overlay_pil = Image.fromarray(overlay)
    buffer = io.BytesIO()
    overlay_pil.save(buffer, format="PNG")
    heatmap_b64 = base64.b64encode(buffer.getvalue()).decode()

    return {
        "label": "DEFECTIVE" if is_defective else "GOOD",
        "score": round(norm_score, 4),
        "raw_score": round(raw_score, 4),
        "threshold": round(THRESHOLD, 4),
        "heatmap": heatmap_b64
    }


@app.get("/", response_class=HTMLResponse)
def root():
    with open("app/static/index.html") as f:
        return f.read()


@app.post("/inspect")
async def inspect(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    result = run_inference(image)
    return result