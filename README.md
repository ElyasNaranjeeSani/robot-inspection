---
title: Robot Inspection Demo
emoji: 🔍
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# Robot Inspection Demo

AI-powered manufacturing defect detection using humanoid robot inspection skills.

**Live demo:** https://huggingface.co/spaces/elyasn/robot-inspection

## What this does

Upload a photo of a manufactured part and the AI will classify it as **GOOD** or **DEFECTIVE**, with an anomaly heatmap showing exactly where the defect is located.

## How it works

- Trained a **PatchCore** anomaly detection model on the MVTec industrial dataset
- PatchCore learns what normal parts look like using only good samples — no labeled defects needed
- At inference time it compares new images against its memory of normal parts and flags anything unusual
- Achieves **100% image-level AUROC** on the metal nut category

## Tech stack

- **Model:** PatchCore via Anomalib
- **Backend:** FastAPI + PyTorch
- **Frontend:** HTML/CSS/JavaScript
- **Deployment:** Hugging Face Spaces (Docker)
- **Model storage:** Hugging Face Hub

## Context

Phase 1 proof of concept exploring AI-powered visual inspection for industrial manufacturing environments. Custom models can be trained on customer-specific parts for real-world deployment.