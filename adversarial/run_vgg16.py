import os
import glob
import cv2
import torch
import numpy as np
import csv
from torchvision.models import vgg16
from pytorch_grad_cam import GradCAM, GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import preprocess_image

from utils.classifier_output_targets import ClassifierOutputTargets
from utils.counterfactual_gradcam import CounterfactualGradCAM
from utils.explainable_methods import (
    run_gradcam, run_guided_backprop, run_guided_gradcam, run_counterfactual_gradcam
)

from utils.xai_metrics import (
    compute_confidence,
    compute_drop_in_confidence,
    apply_mask,
    compute_insertion_score
)

# -------------------------------------------------------------------
# Create CSV
# -------------------------------------------------------------------
METRICS_CSV = "xai_metrics_vgg16.csv"
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

# -------------------------------------------------------------------
def search_image_paths(input_folder):
    image_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        ext_images = glob.glob(os.path.join(input_folder, ext))
        image_paths.extend(ext_images)
        ext_images_upper = glob.glob(os.path.join(input_folder, ext.upper()))
        image_paths.extend(ext_images_upper)
    return image_paths

# -------------------------------------------------------------------
INPUT_FOLDER = "input"
OUTPUT_FOLDER_PREFIX = "outputs_vgg16"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

print("Loading VGG16...")
model = vgg16(weights='DEFAULT')
model.to(device)
model.eval()

# Image list
image_paths = search_image_paths(INPUT_FOLDER)
print(f"{len(image_paths)} images found.")

# Last conv layer
target_layers = [model.features[-1]]
grad_cam = GradCAM(model=model, target_layers=target_layers)
counterfactual_grad_cam = CounterfactualGradCAM(model=model, target_layers=target_layers)

guided_backprop = GuidedBackpropReLUModel(model=model, device=device)

# Metrics for statistics
drops = []
insertions = []

# -------------------------------------------------------------------
# MAIN LOOP
# -------------------------------------------------------------------
for img_path in image_paths:
    filename = os.path.basename(img_path)
    try:
        print(f"\nProcessing: {filename}...")

        rgb_img = cv2.imread(img_path, cv2.IMREAD_COLOR)[:, :, ::-1]
        rgb_img = np.float32(rgb_img) / 255

        input_tensor = preprocess_image(
            rgb_img,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ).to(device)

        # -----------------------------
        # 1. Standard prediction
        # -----------------------------
        logits = model(input_tensor)
        prediction = torch.argmax(logits, dim=1).item()

        conf_before = compute_confidence(model, input_tensor, prediction)

        # -----------------------------
        # 2. Grad-CAM
        # -----------------------------
        grayscale_cam = grad_cam(input_tensor=input_tensor, targets=None)[0]

        # -----------------------------
        # 3. Confidence after masking
        # -----------------------------
        masked_img = apply_mask(rgb_img, grayscale_cam)

        masked_tensor = preprocess_image(
            masked_img,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ).to(device)

        conf_after = compute_confidence(model, masked_tensor, prediction)
        drop = conf_before - conf_after

        # -----------------------------
        # 4. Insertion score
        # -----------------------------
        insertion_score = compute_insertion_score(model, input_tensor, grayscale_cam)

        # Collect for global stats
        drops.append(drop)
        insertions.append(insertion_score)

        # -----------------------------
        # 5. Save CSV row
        # -----------------------------
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

        # -----------------------------
        # 6. Save visualizations
        # -----------------------------
        run_gradcam(filename, None, input_tensor, rgb_img,
                    OUTPUT_FOLDER_PREFIX, grad_cam, prediction)

        run_counterfactual_gradcam(filename, None,
                                   input_tensor, rgb_img,
                                   OUTPUT_FOLDER_PREFIX, counterfactual_grad_cam, prediction)

        run_guided_backprop(filename, None,
                            input_tensor, OUTPUT_FOLDER_PREFIX,
                            guided_backprop, prediction)

        run_guided_gradcam(filename, None, input_tensor,
                           OUTPUT_FOLDER_PREFIX, guided_backprop, grad_cam, prediction)

    except Exception as e:
        print(f"Error in image {filename}: {e}")

# -------------------------------------------------------------------
# GLOBAL STATISTICS
# -------------------------------------------------------------------
import numpy as np

print("\n================ GLOBAL STATS (VGG16) ================")

if len(drops) > 0:
    drops = np.array(drops)
    print("Drop in Confidence:")
    print("  Min:", float(drops.min()))
    print("  Max:", float(drops.max()))
    print("  Mean:", float(drops.mean()))
    print("  Median:", float(np.median(drops)))
    print("  Std:", float(drops.std()))

if len(insertions) > 0:
    insertions = np.array(insertions)
    print("\nInsertion Score:")
    print("  Min:", float(insertions.min()))
    print("  Max:", float(insertions.max()))
    print("  Mean:", float(insertions.mean()))
    print("  Median:", float(np.median(insertions)))
    print("  Std:", float(insertions.std()))

print("\nResults saved in", OUTPUT_FOLDER_PREFIX)
print("CSV saved in:", METRICS_CSV)
