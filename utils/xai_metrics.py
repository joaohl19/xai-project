import numpy as np
import torch
import cv2
from pytorch_grad_cam.utils.image import preprocess_image
import torch.nn.functional as F


def compute_drop_in_confidence(model, input_tensor, mask, device):
    with torch.no_grad():
        orig_logits = model(input_tensor.to(device))
        orig_prob = torch.softmax(orig_logits, dim=1).max().item()

    rgb = input_tensor[0].cpu().numpy().transpose(1,2,0)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

    mask_3ch = np.stack([mask, mask, mask], axis=2)
    masked_rgb = rgb * mask_3ch

    masked_tensor = preprocess_image(
        masked_rgb.astype(np.float32),
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ).to(device)

    with torch.no_grad():
        new_logits = model(masked_tensor)
        new_prob = torch.softmax(new_logits, dim=1).max().item()

    drop = orig_prob - new_prob
    return drop, orig_prob, new_prob


def blur_image(img, ksize=51):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def insertion_score(model, rgb_img, cam_mask, steps=50, device='cuda'):
    H, W, _ = rgb_img.shape
    baseline = blur_image(rgb_img)
    insertion_img = baseline.copy()

    cam_flat = cam_mask.flatten()
    sorted_idx = np.argsort(cam_flat)[::-1]

    pixels_per_step = len(sorted_idx) // steps
    probs = []

    for s in range(steps):
        idx = sorted_idx[s * pixels_per_step : (s+1) * pixels_per_step]
        ys = idx // W
        xs = idx % W

        insertion_img[ys, xs] = rgb_img[ys, xs]

        tensor = preprocess_image(
            insertion_img.astype(np.float32),
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ).to(device)

        with torch.no_grad():
            prob = torch.softmax(model(tensor), dim=1).max().item()

        probs.append(prob)

    auc = np.trapz(probs) / steps
    return auc, probs

def compute_confidence(model, input_tensor, target_class):
    """Return softmax confidence for a given class."""
    with torch.no_grad():
        out = model(input_tensor)
        probs = F.softmax(out, dim=1)
        return float(probs[0, target_class].cpu().item())

def apply_mask(image, cam, strength=1.0):
    """Mask the input using CAM (for after-confidence)."""
    cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
    mask = 1 - strength * cam_resized
    masked = image * mask[..., None]
    return masked

def compute_insertion_score(model, image_tensor, cam):
    """
    Very simple insertion metric:
    Add the most important pixels gradually and measure confidence increase.
    """
    cam_norm = cam / cam.max()
    cam_norm = cv2.resize(cam_norm, (image_tensor.shape[3], image_tensor.shape[2]))

    steps = 10
    scores = []

    for alpha in np.linspace(0, 1, steps):
        mask = cam_norm >= alpha
        masked_image = image_tensor.clone()
        masked_image[:, :, ~mask] = 0
        
        with torch.no_grad():
            conf = F.softmax(model(masked_image), dim=1).max().item()

        scores.append(conf)

    return float(np.mean(scores))
