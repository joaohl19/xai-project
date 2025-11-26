import os
import glob
import cv2
import numpy as np
import torch
from imagenet_classes import classes
from torchvision.models import resnet50
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

# Method that generalizes the generation of heatmaps to output targets
class ClassifierOutputTargets:
    def __init__(self, categories):
        self.categories = categories
    
    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return torch.sum(model_output[self.categories])
        return torch.sum(torch.cat([model_output[:, c] for c in self.categories], dim=0), dim=1)

INPUT_FOLDER = "input_images"
OUTPUT_FOLDER = "result_images"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Setting up the device where the model will run
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Loading the model into the device
print("Loading ResNet50...")
model = resnet50(weights='DEFAULT')
model.to(device)
model.eval()

# Setting up Grad-CAM for the last convolutional layer
target_layers = [model.layer4[-1]]
gradcam = GradCAM(model=model, target_layers=target_layers) 

image_paths = []
for ext in ('*.jpg', '*.jpeg', '*.png'):
    # Searching for images in the input directory and adding them to image_paths
    ext_images = glob.glob(os.path.join(INPUT_FOLDER, ext))
    image_paths.extend(ext_images)
    ext_images_upper = glob.glob(os.path.join(INPUT_FOLDER, ext.upper()))
    image_paths.extend(ext_images_upper)

print(f"{len(image_paths)} images found.")

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

        # Generate Grad-CAM (targets=None gets the predicted class)
        targets = [ClassifierOutputTargets([245, 281])] # French Bulldog and Tabby Cat
        classes_labels = [classes[245], classes[281]]
        grayscale_cam = gradcam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        # Overlay heatmap in image
        gradcam_img = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        # Save cam_img
        save_path = os.path.join(OUTPUT_FOLDER, f"grad-cam_{filename}")
        cv2.imwrite(save_path, gradcam_img[:, :, ::-1])
        
    except Exception as e:
        print(f"Error in image {filename}: {e}")

print(f"Results saved in '{OUTPUT_FOLDER}'")