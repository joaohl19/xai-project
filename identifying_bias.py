import os
import cv2
import re
import torch
from PIL import Image
from numpy import asarray
from model import SimpleCNN
from torchvision import transforms
from torch.nn import functional as F
from pytorch_grad_cam import GradCAM, GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import preprocess_image
from utils.explainable_methods import run_gradcam, run_guided_backprop, run_guided_gradcam, run_counterfactual_gradcam

# Image preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Setting up the device where the model will run
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Dictionary equal to 'train_dataset.class_to_idx' from training.py
class_to_idx = {
    'cat' : 0,
    'dog' : 1
}

idx_to_class = ['cat', 'dog']

# Instantiating the model and loading pretrained weights
model = SimpleCNN(dropout_rate=0.3)
model.load_state_dict(torch.load("pretrained_model.pth", weights_only=True, map_location=device))
model.eval()
model.to(device)

# Setting up Grad-CAM for the last convolutional layer
target_layers = [model.conv4]
grad_cam = GradCAM(model=model, target_layers=target_layers) 

IMG_REGEX = re.compile(r".*\.(jpg|jpeg|png)$", re.IGNORECASE)


# Testing the biased model for cat images
path_cat = "../test/dog"
directory = os.fsencode(path_cat)

# If directory is bytes, decode it once
if isinstance(directory, bytes):
    directory = directory.decode()   # or .decode("utf-8")

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    fullpath = os.path.join(directory, filename)

    # 1. Skip if not a file
    if not os.path.isfile(fullpath):
        continue

    # 2. Skip if not an image (via regex on extension)
    if not IMG_REGEX.match(filename):
        continue

    # Reading the image and resizing it to fit the model's architecture
    rgb_img = Image.open(os.path.join(path_cat, filename)).convert('RGB')
    rgb_img = rgb_img.resize((224,224), Image.LANCZOS)

    # Pre-processing the image
    input_tensor = transform(rgb_img).unsqueeze(0).to(device)

    # Generate heatmp from Grad-CAM
    target= [ClassifierOutputTarget(class_to_idx['cat'])]
    grad_cam_vis = grad_cam(input_tensor=input_tensor, targets=None)[0, :]
    
    numpy_img = asarray(rgb_img) / 255
    
    # Overlay heatmap in image
    gradcam_img = show_cam_on_image(numpy_img, grad_cam_vis, use_rgb=True)
    
    # Get probabilities of the output
    output = model(input_tensor)
    probs = F.softmax(output, dim=1).squeeze(0)

    # Save the grad_cam image
    new_filename = "test"+filename
    cv2.imwrite(os.path.join(path_cat, new_filename), gradcam_img[:, :, ::-1])
    
    for idx in range(len(probs)):
        print(f"{filename} prediction: {idx_to_class[idx]} with {probs[idx].item() * 100 :.2f}% of confidence")