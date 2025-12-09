from sklearn.metrics import auc
import torch
from pytorch_grad_cam import GradCAM
import numpy as np
from typing import List
from torch.nn import functional as F
from torch.utils.data import DataLoader
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
def run_insertion_score(dataloader: DataLoader, device: torch.device, grad_cam:GradCAM, rates: List[int], model):
    insertion_values_per_rate = []
    for rate in rates:
        values = []
        for (imgs, labels) in dataloader:
            for i in range(len(imgs)):
                # Preprocess image
                img = imgs[i].unsqueeze(0).to(device)
                
                # Run model for image and get score and predicted class
                with torch.no_grad():
                    output = model(img).squeeze(0)
                    prediction = torch.argmax(output).item()

                # Erase the top N most important pixels after applying Grad-CAM
                target = [ClassifierOutputTarget(prediction)]
                grad_cam_vis = grad_cam(input_tensor=img, targets=target)[0, :]
                threshold = np.percentile(grad_cam_vis, 100*(1-rate))
                mask = (grad_cam_vis >= threshold).astype(np.float32) # (224, 224)

                mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device) # (1, 1, 224, 224)

                new_img = mask_tensor * img 

                # Run the model again for the new image and obtaining its score for the predicted class
                with torch.no_grad():
                    new_output = model(new_img).squeeze(0)
                    new_probs = F.softmax(new_output, dim=0)
                    # Get score for modified image
                    new_score = new_probs[prediction].item()

                values.append(new_score)
                
            insertion_values_per_rate.append(float(np.mean(np.array(values))))

    insertion_auc_score = auc(rates, insertion_values_per_rate)
    return insertion_values_per_rate, insertion_auc_score

def run_drop_in_confidence(dataloader: DataLoader, device: torch.device, grad_cam:GradCAM, rates: List[int], model):
    drop_values_per_rate = []
    for rate in rates:
        values = []
        for (imgs, labels) in dataloader:
            for i in range(len(imgs)):
                # Preprocess image
                img = imgs[i].unsqueeze(0).to(device)
                # Run model for image and get score and predicted class
                with torch.no_grad():
                    output = model(img).squeeze(0)
                    prediction = torch.argmax(output).item()

                    probs = F.softmax(output, dim=0)

                    score = torch.max(probs, dim=0).values.item()

                # Erase the top N most important pixels after applying Grad-CAM
                target = [ClassifierOutputTarget(prediction)]
                grad_cam_vis = grad_cam(input_tensor=img, targets=target)[0, :]
                threshold = np.percentile(grad_cam_vis, 100*(1 - rate))
                mask = (grad_cam_vis < threshold).astype(np.float32) # (224, 224)

                mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device) # (1, 1, 224, 224)
             
                new_img = mask_tensor * img

                # Run the model again for the new image and obtaining its score for the predicted class
                with torch.no_grad():
                    new_output = model(new_img).squeeze(0)
                    new_probs = F.softmax(new_output, dim=0)
                    # Get score for modified image
                    new_score = new_probs[prediction].item()

                # Compute [Score[original] - Score[new] / Score[original]]
                value = max(0, score - new_score) / score
                values.append(value)

            drop_values_per_rate.append(float(np.mean(np.array(values))))

    deletion_auc_score = auc(rates, drop_values_per_rate)
    return drop_values_per_rate, deletion_auc_score