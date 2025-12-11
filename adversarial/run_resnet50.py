import os
import glob
import cv2
import csv
import torch
import numpy as np
from torchvision.models import resnet50
from pytorch_grad_cam import GradCAM, GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import preprocess_image

from utils.classifier_output_targets import ClassifierOutputTargets
from utils.counterfactual_gradcam import CounterfactualGradCAM
from utils.explainable_methods import (
    run_gradcam,
    run_guided_backprop,
    run_guided_gradcam,
    run_counterfactual_gradcam
)
from utils.xai_metrics import (
    compute_confidence,
    compute_drop_in_confidence,
    compute_insertion_score,
    apply_mask
)

METRICS_CSV = "xai_metrics_resnet50.csv"

# ------------------------------------
# Create CSV if not exists
# ------------------------------------
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
OUTPUT_FOLDER_PREFIX = "outputs_resnet50"

# ------------------------------------
# Device
# ------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# ------------------------------------
# Load ResNet-50
# ------------------------------------
print("Loading ResNet50...")
model = resnet50(weights='DEFAULT')
model.to(device)
model.eval()

# ------------------------------------
# Load images
# ------------------------------------
image_paths = search_image_paths(INPUT_FOLDER)
print(f"{len(image_paths)} images found.")

# ------------------------------------
# Grad-CAM: last conv layer in ResNet-50
# (layer4[-1] = last bottleneck block)
# ------------------------------------
target_layers = [model.layer4[-1]]
grad_cam = GradCAM(model=model, target_layers=target_layers)
counter_grad_cam = CounterfactualGradCAM(model=model, target_layers=target_layers)
guided_backprop = GuidedBackpropReLUModel(model=model, device=device)

# ------------------------------------
# Store drops for statistics
# ------------------------------------
drop_values = []

# ============================================
# PROCESS ALL IMAGES
# ============================================
for img_path in image_paths:
    filename = os.path.basename(img_path)

    try:
        print(f"\nProcessing: {filename}...")

        # Load image (BGR → RGB)
        rgb_img = cv2.imread(img_path, cv2.IMREAD_COLOR)[:, :, ::-1]
        rgb_img = np.float32(rgb_img) / 255.0

        # Preprocess
        input_tensor = preprocess_image(
            rgb_img,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ).to(device)

        # ------------------------------------
        # 1. Standard prediction
        # ------------------------------------
        logits = model(input_tensor)
        prediction = torch.argmax(logits, dim=1).item()

        conf_before = compute_confidence(model, input_tensor, prediction)

        # ------------------------------------
        # 2. Grad-CAM heatmap
        # ------------------------------------
        grayscale_cam = grad_cam(input_tensor=input_tensor, targets=None)[0]

        # ------------------------------------
        # 3. Masked confidence
        # ------------------------------------
        masked_img = apply_mask(rgb_img, grayscale_cam)
        masked_tensor = preprocess_image(
            masked_img,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ).to(device)

        conf_after = compute_confidence(model, masked_tensor, prediction)

        drop = conf_before - conf_after
        drop_values.append(drop)

        # ------------------------------------
        # 4. Insertion score
        # ------------------------------------
        insertion_score = compute_insertion_score(model, input_tensor, grayscale_cam)

        # ------------------------------------
        # 5. Save to CSV
        # ------------------------------------
        with open(METRICS_CSV, "a", newline="") as f:
            csv.writer(f).writerow([
                filename,
                prediction,
                conf_before,
                conf_after,
                drop,
                insertion_score
            ])

        # ------------------------------------
        # 6. Generate visualizations
        # ------------------------------------
        run_gradcam(filename, None, input_tensor, rgb_img,
                    OUTPUT_FOLDER_PREFIX, grad_cam, prediction)

        run_counterfactual_gradcam(filename, None, input_tensor, rgb_img,
                                   OUTPUT_FOLDER_PREFIX, counter_grad_cam, prediction)

        run_guided_backprop(filename, None, input_tensor,
                            OUTPUT_FOLDER_PREFIX, guided_backprop, prediction)

        run_guided_gradcam(filename, None, input_tensor,
                           OUTPUT_FOLDER_PREFIX, guided_backprop, grad_cam, prediction)

    except Exception as e:
        print(f"Error in image {filename}: {e}")

print(f"\nResults saved in '{OUTPUT_FOLDER_PREFIX}' and {METRICS_CSV}")

# ============================================
# PRINT GLOBAL STATS
# ============================================
if len(drop_values) > 0:
    print("\n===== DROP-IN-CONFIDENCE STATISTICS =====")
    print(f"Samples: {len(drop_values)}")
    print(f"Min drop:    {np.min(drop_values):.4f}")
    print(f"Max drop:    {np.max(drop_values):.4f}")
    print(f"Mean drop:   {np.mean(drop_values):.4f}")
    print(f"Median drop: {np.median(drop_values):.4f}")
    print(f"Std dev:     {np.std(drop_values):.4f}")
else:
    print("No drop-in-confidence values collected!")
