import torch

checkpoint = "./results/Patchcore/metal_nut/latest/weights/lightning/model.ckpt"
ckpt = torch.load(checkpoint, weights_only=False)

print("Post processor and normalisation keys:")
for key, value in ckpt["state_dict"].items():
    if "memory" not in key and "feature" not in key:
        print(f"  {key}: {value}")