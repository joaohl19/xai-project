import os
import glob
import sys
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
from torchvision.models import resnet50, ResNet50_Weights
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image


from imagenet_classes import classes

# Paths relative to the adversarial/ directory
BASE_DIR = "adversarial"
INPUT_CLEAN = os.path.join(BASE_DIR, "input_clean")
INPUT_ADVERSARIAL = os.path.join(BASE_DIR, "input_adversarial")
RESULTS_CLEAN = os.path.join(BASE_DIR, "results_clean")
RESULTS_ADVERSARIAL = os.path.join(BASE_DIR, "results_adversarial")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def load_resnet50(device: torch.device) -> torch.nn.Module:
    print("Loading ResNet50...")
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.to(device)
    model.eval()
    return model


def load_image_for_art(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = img.resize((224, 224), Image.BILINEAR)
    arr = np.array(img).astype(np.float32)   # HWC in [0,255]
    arr = np.transpose(arr, (2, 0, 1))       # CHW
    arr = np.expand_dims(arr, axis=0)        # NCHW
    return arr


def save_adversarial_image(x_adv: np.ndarray, save_path: str) -> np.ndarray:
    """Save adversarial image (NCHW, [0,255]) and return HWC array."""
    arr = x_adv[0].transpose(1, 2, 0)       # CHW → HWC
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(save_path)
    return arr


def compare_images(image_a: np.ndarray, image_b: np.ndarray) -> float:
    """Return 1 - SSIM between two RGB images in [0,255]."""
    h, w = min(image_a.shape[0], image_b.shape[0]), min(image_a.shape[1], image_b.shape[1])
    win_size = min(7, h, w)
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        return 0.0
    return 1.0 - ssim(image_a, image_b, channel_axis=-1, win_size=win_size)


class NormalizedModel(nn.Module):
    """Wraps a model so ART can work on unnormalized [0,255] inputs."""

    def __init__(self, model: nn.Module, mean, std):
        super().__init__()
        self.model = model
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ART gives x in [0,255], convert to [0,1]
        x = x / 255.0

        if x.device != self.mean.device:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)

        x = (x - self.mean) / self.std
        return self.model(x)


def build_pgd_attack(
    base_model: nn.Module,
    device: torch.device,
    eps: float,
    eps_step: float,
    max_iter: int,
) -> ProjectedGradientDescent:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    wrapped_model = NormalizedModel(base_model, mean, std).to(device).eval()
    criterion = nn.CrossEntropyLoss()

    classifier = PyTorchClassifier(
        model=wrapped_model,
        loss=criterion,
        optimizer=None,
        input_shape=(3, 224, 224),
        nb_classes=1000,
        preprocessing=None,  # IMPORTANT: no preprocessing here
    )

    attack = ProjectedGradientDescent(
        estimator=classifier,
        eps=eps,
        eps_step=eps_step,
        max_iter=max_iter,
        targeted=False,
    )
    return attack


def generate_adversarial_folder(
    base_model: nn.Module,
    device: torch.device,
    input_folder: str = INPUT_CLEAN,
    output_folder: str = INPUT_ADVERSARIAL,
    eps: float = 15.0,
    eps_step: float = 2.0,
    max_iter: int = 50,
) -> None:
    os.makedirs(output_folder, exist_ok=True)

    attack = build_pgd_attack(
        base_model=base_model,
        device=device,
        eps=eps,
        eps_step=eps_step,
        max_iter=max_iter,
    )

    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(glob.glob(os.path.join(input_folder, ext)))
        image_paths.extend(glob.glob(os.path.join(input_folder, ext.upper())))

    print(f"{len(image_paths)} images found for adversarial generation.")

    for path in image_paths:
        fname =  os.path.basename(path)
        out_path = os.path.join(output_folder, "adversarial_" + fname)

        try:
            print(f"Attacking: {fname}")

            x = load_image_for_art(path)  # dtype float32, [0,255]
            x_adv = attack.generate(x)

            orig = x[0].transpose(1, 2, 0).astype(np.uint8)
            adv = save_adversarial_image(x_adv, out_path)

            distance = compare_images(orig, adv)
            print(f"SSIM distance (1 - SSIM) between clean and adversarial: {distance:.4f}")

        except Exception as e:
            print(f"ERROR processing {fname}: {e}")

    print(f"\nAdversarial samples saved in: {output_folder}")


def create_gradcam(model: nn.Module) -> GradCAM:
    target_layers = [model.layer4[-1]]
    return GradCAM(model=model, target_layers=target_layers)


def predict_top1(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
) -> Tuple[int, float]:
    with torch.no_grad():
        logits = model(input_tensor.to(device))
        probs = torch.softmax(logits, dim=1)
        idx = torch.argmax(probs, dim=1).item()
        conf = probs[0, idx].item()
    return idx, conf


def run_gradcam_for_folder(
    model: nn.Module,
    grad_cam: GradCAM,
    device: torch.device,
    input_folder: str,
    output_folder: str,
) -> None:
    os.makedirs(output_folder, exist_ok=True)

    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(glob.glob(os.path.join(input_folder, ext)))
        image_paths.extend(glob.glob(os.path.join(input_folder, ext.upper())))

    print(f"{len(image_paths)} images found for Grad-CAM in '{input_folder}'.")

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        try:
            print(f"\nProcessing: {filename}...")

            # Read image (BGR → RGB)
            rgb_img = cv2.imread(img_path, cv2.IMREAD_COLOR)[:, :, ::-1]
            rgb_img = np.float32(rgb_img) / 255.0

            input_tensor = preprocess_image(
                rgb_img,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ).to(device)

            pred_idx, conf = predict_top1(model, input_tensor, device)
            class_name = classes[pred_idx]
            print(f"Predicted class: {class_name}")
            print(f"Confidence: {conf:.4f}")

            grayscale_cam = grad_cam(input_tensor=input_tensor, targets=None)[0]
            gradcam_img = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            save_path = os.path.join(output_folder, f"grad-cam_{filename}")
            cv2.imwrite(save_path, gradcam_img[:, :, ::-1])

        except Exception as e:
            print(f"Error in image {filename}: {e}")

    print(f"\nGrad-CAM results saved in '{output_folder}'")


def main() -> None:
    os.makedirs(INPUT_CLEAN, exist_ok=True)
    os.makedirs(INPUT_ADVERSARIAL, exist_ok=True)

    device = get_device()
    model = load_resnet50(device)

    # 1) Generate adversarial images from clean images → input_adversarial/
    generate_adversarial_folder(
        base_model=model,
        device=device,
        input_folder=INPUT_CLEAN,
        output_folder=INPUT_ADVERSARIAL,
        eps=15.0,
        eps_step=2.0,
        max_iter=50,
    )

    # 2) Set up Grad-CAM
    grad_cam = create_gradcam(model)

    # 3) Grad-CAM on clean images → results_clean/
    run_gradcam_for_folder(
        model=model,
        grad_cam=grad_cam,
        device=device,
        input_folder=INPUT_CLEAN,
        output_folder=RESULTS_CLEAN,
    )

    # 4) Grad-CAM on adversarial images → results_adversarial/
    run_gradcam_for_folder(
        model=model,
        grad_cam=grad_cam,
        device=device,
        input_folder=INPUT_ADVERSARIAL,
        output_folder=RESULTS_ADVERSARIAL,
    )


if __name__ == "__main__":
    main()

