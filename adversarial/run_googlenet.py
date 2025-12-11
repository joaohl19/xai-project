import os
import glob
import cv2
import torch
import numpy as np
from torch import nn
from torchvision.models import googlenet
from pytorch_grad_cam import GradCAM, GuidedBackpropReLUModel
from utils.classifier_output_targets import ClassifierOutputTargets
from pytorch_grad_cam.utils.image import preprocess_image
from utils.counterfactual_gradcam import CounterfactualGradCAM
from utils.explainable_methods import run_gradcam, run_guided_backprop, run_guided_gradcam, run_counterfactual_gradcam
from utils.xai_metrics import (
    compute_drop_in_confidence, insertion_score, compute_confidence,
    blur_image, apply_mask, compute_insertion_score
)
import csv

METRICS_CSV = "xai_metrics.csv"

# Create CSV with header if not exists
if not os.path.exists(METRICS_CSV):
    with open(METRICS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename",
            "predicted_class",
            "confidence_before",
            "confidence_after",
            "drop_in_confidence",
            "insertion_score"
        ])


def search_image_paths(input_folder):
    image_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        ext_images = glob.glob(os.path.join(input_folder, ext))
        image_paths.extend(ext_images)
        ext_images_upper = glob.glob(os.path.join(input_folder, ext.upper()))
        image_paths.extend(ext_images_upper)
    return image_paths


INPUT_FOLDER = "input"
OUTPUT_FOLDER_PREFIX = "outputs_googlenet"

# Setting up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)

# Load model
print("Loading googlenet...")
model = googlenet(weights='DEFAULT')
model.aux1 = nn.Identity()
model.aux2 = nn.Identity()
model.to(device)
model.eval()

# Load images
image_paths = search_image_paths(INPUT_FOLDER)
print(f"{len(image_paths)} images found.")

# GradCAM setup
target_layers = [model.inception5b]
grad_cam = GradCAM(model=model, target_layers=target_layers)
counterfactual_grad_cam = CounterfactualGradCAM(model=model, target_layers=target_layers)
guided_backprop = GuidedBackpropReLUModel(model=model, device=device)

# --------------------------------------------------------------
# COLLECT ALL DROPS HERE
# --------------------------------------------------------------
all_drops = []

# --------------------------------------------------------------
# LOOP
# --------------------------------------------------------------
for img_path in image_paths:
    filename = os.path.basename(img_path)
    try:
        print(f"Processing: {filename}...")

        rgb_img = cv2.imread(img_path, cv2.IMREAD_COLOR)[:, :, ::-1]
        rgb_img = np.float32(rgb_img) / 255

        input_tensor = preprocess_image(
            rgb_img,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ).to(device)

        # 1. Standard prediction
        logits = model(input_tensor)
        prediction = torch.argmax(logits, dim=1).item()

        # Confidence before CAM
        conf_before = compute_confidence(model, input_tensor, prediction)

        # 2. Grad-CAM
        grayscale_cam = grad_cam(input_tensor=input_tensor, targets=None)[0]

        # 3. Confidence AFTER mask
        masked_img = apply_mask(rgb_img, grayscale_cam)
        masked_tensor = preprocess_image(
            masked_img,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ).to(device)

        conf_after = compute_confidence(model, masked_tensor, prediction)

        drop = conf_before - conf_after
        all_drops.append(drop)

        # 4. Insertion score
        insertion_score = compute_insertion_score(model, input_tensor, grayscale_cam)

        # 5. Write CSV
        with open(METRICS_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                filename,
                prediction,
                conf_before,
                conf_after,
                drop,
                insertion_score
            ])

        # 6. Save visualizations
        run_gradcam(filename, None, input_tensor, rgb_img,
                    OUTPUT_FOLDER_PREFIX, grad_cam, prediction)

        run_counterfactual_gradcam(filename, None, input_tensor, rgb_img,
                    OUTPUT_FOLDER_PREFIX, counterfactual_grad_cam, prediction)

        run_guided_backprop(filename, None, input_tensor,
                            OUTPUT_FOLDER_PREFIX, guided_backprop, prediction)

        run_guided_gradcam(filename, None, input_tensor,
                           OUTPUT_FOLDER_PREFIX, guided_backprop, grad_cam, prediction)

    except Exception as e:
        print(f"Error in image {filename}: {e}")

print(f"Results saved in '{OUTPUT_FOLDER_PREFIX}'")

# --------------------------------------------------------------
# FINAL: PRINT STATISTICS
# --------------------------------------------------------------
if len(all_drops) > 0:
    drops_np = np.array(all_drops)

    print("\n" + "=" * 60)
    print("Drop-in-confidence statistics:")
    print("=" * 60)
    print(f"Count:  {len(drops_np)}")
    print(f"Min:    {drops_np.min():.6f}")
    print(f"Max:    {drops_np.max():.6f}")
    print(f"Mean:   {drops_np.mean():.6f}")
    print(f"Median: {np.median(drops_np):.6f}")
    print(f"Std:    {drops_np.std():.6f}")
    print("=" * 60 + "\n")
else:
    print("No valid samples — no statistics computed.")
