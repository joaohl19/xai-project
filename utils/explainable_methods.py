import os
import cv2
import torch
import numpy as np
from typing import List, Optional
from imagenet_classes import classes
from pytorch_grad_cam.utils.image import deprocess_image, show_cam_on_image
from pytorch_grad_cam import GradCAM, GuidedBackpropReLUModel
from utils.classifier_output_targets import ClassifierOutputTargets

def run_gradcam(filename:str, targets:List[ClassifierOutputTargets],
                input_tensor:torch.Tensor, rgb_img:np.ndarray, 
                output_folder_prefix:str, grad_cam:GradCAM):
    # Generate heatmp from Grad-CAM
    grad_cam_vis = grad_cam(input_tensor=input_tensor, targets=targets)[0, :]

    # Overlay heatmap in image
    gradcam_img = show_cam_on_image(rgb_img, grad_cam_vis, use_rgb=True)

    # Save grad-cam img
    if(targets is None) :
        classes_labels = "predictedclass"
    elif isinstance(targets[0].categories, List):
        delimeter = "_"
        categories_id = targets[0].categories
        classes_labels = delimeter.join([classes[i] for i in categories_id])
    else:
        classes_labels = classes[targets[0].categories]
    save_folder = os.path.join(output_folder_prefix, "grad-cam")
    save_path =  os.path.join(save_folder, f"{classes_labels}_{filename}")
    cv2.imwrite(save_path, gradcam_img[:, :, ::-1])
    

def run_guided_backprop(filename:str, target:int, 
                        input_tensor:torch.Tensor, output_folder_prefix:str, 
                        guided_backprop:GuidedBackpropReLUModel):
    
    guided_backprop_vis = guided_backprop(input_tensor, target_category=target)
    guided_backprop_img = deprocess_image(guided_backprop_vis)

    # Save guided_backprop img
    if(target is None):
        class_label = "predictedclass"
    else:
        class_label = classes[target]
    save_folder = os.path.join(output_folder_prefix, f"guided_backprop")
    save_path = os.path.join(save_folder, f"{class_label}_{filename}")
    cv2.imwrite(save_path, guided_backprop_img[:, :, ::-1])

def run_guided_gradcam(filename:str, target:Optional[int], 
                    input_tensor:torch.Tensor, output_folder_prefix:str, 
                    guided_backprop:GuidedBackpropReLUModel, grad_cam:GradCAM):
            
    # Get guided backprop visualization
    guided_backprop_vis = guided_backprop(input_tensor, target_category=target)

    # Get grad-cam visualization
    grad_cam_target = [ClassifierOutputTargets(target)]
    grad_cam_vis = grad_cam(input_tensor=input_tensor, targets=grad_cam_target)[0, :]
    grad_cam_vis = np.stack((grad_cam_vis, grad_cam_vis, grad_cam_vis), axis=2)
    
    # Merge both visualizations and convert them to a visible image
    guided_gradcam_img = deprocess_image(guided_backprop_vis * grad_cam_vis)

    # Save guided_backprop img
    if(target is None):
        class_label = "predictedclass"
    else:
        class_label = classes[target]
    save_folder = os.path.join(output_folder_prefix, f"guided_grad-cam")
    save_path = os.path.join(save_folder, f"{class_label}_{filename}")
    cv2.imwrite(save_path, guided_gradcam_img[:, :, ::-1])

