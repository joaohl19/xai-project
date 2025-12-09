import os
import glob
import cv2
import torch
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights
from pytorch_grad_cam import GradCAM, GuidedBackpropReLUModel
from utils.counterfactual_gradcam import CounterfactualGradCAM
from pytorch_grad_cam.utils.image import preprocess_image
from utils.explainable_methods import run_gradcam, run_guided_backprop, run_guided_gradcam, run_counterfactual_gradcam

def search_image_paths(input_folder):
    image_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        # Searching for images in the input directory and adding them to image_paths
        ext_images = glob.glob(os.path.join(input_folder, ext))
        image_paths.extend(ext_images)
        ext_images_upper = glob.glob(os.path.join(input_folder, ext.upper()))
        image_paths.extend(ext_images_upper)
    return image_paths


INPUT_FOLDER = "input"
OUTPUT_FOLDER_PREFIX = "outputs_resnet50"

# Setting up the device where the model will run
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Loading the model into the device
print("Loading ResNet50...")
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.to(device)
model.eval()

# Searching for images in the input folder
image_paths = search_image_paths(INPUT_FOLDER)
print(f"{len(image_paths)} images found.")

# Setting up Grad-CAM  and counterfactual Grad-CAM for the last convolutional layer
target_layers = [model.layer4[-1]]
grad_cam = GradCAM(model=model, target_layers=target_layers) 
counterfactual_grad_cam = CounterfactualGradCAM(model=model, target_layers=target_layers)

guided_backprop = GuidedBackpropReLUModel(model=model, device=device)
for img_path in image_paths:
    filename = os.path.basename(img_path)
    try:
        print(f"Processing: {filename}...")
        
        # Reading the image reversing color channels (BGR --> RGB)
        rgb_img = cv2.imread(img_path, cv2.IMREAD_COLOR)[:, :, ::-1] 

        # Normalization
        rgb_img = np.float32(rgb_img) / 255

        # Pre-processing the image
        input_tensor = preprocess_image(rgb_img,
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        prediction = torch.argmax(model(input_tensor).squeeze(0)).item()

        targets_gradcam = None
        print("Runnig Grad-CAM...")
        run_gradcam(filename, targets_gradcam, 
                    input_tensor,rgb_img, 
                    OUTPUT_FOLDER_PREFIX,grad_cam, prediction)
        
        print("Runnig counterfactual Grad-CAM...")
        run_counterfactual_gradcam(filename, targets_gradcam, 
                    input_tensor,rgb_img, 
                    OUTPUT_FOLDER_PREFIX,counterfactual_grad_cam, prediction)
        
        target_guided_backprop = None
        print("Runnig Guided Backpropagation...")
        run_guided_backprop(filename, target_guided_backprop, input_tensor, 
                            OUTPUT_FOLDER_PREFIX, guided_backprop, prediction)
        
        target_guided_gradcam = None
        print("Runnig Guided Grad-CAM...")
        run_guided_gradcam(filename, target_guided_gradcam,input_tensor, 
                           OUTPUT_FOLDER_PREFIX, guided_backprop, grad_cam, prediction)
    except Exception as e:
        print(f"Error in image {filename}: {e}")

print(f"Results saved in '{OUTPUT_FOLDER_PREFIX}'")