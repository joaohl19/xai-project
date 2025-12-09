import torch
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from torch.utils.data import DataLoader
from torchvision.models import vgg16
from torchvision.models import VGG16_Weights
from imagenetv2_pytorch import ImageNetV2Dataset
from utils.plot_metrics import plot_drop_in_confidence, plot_insertion_score
from utils.calculate_metrics import run_drop_in_confidence, run_insertion_score

OUTPUT_DIR="results/vgg16"
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = ImageNetV2Dataset(variant="matched-frequency", transform=preprocess)

# Setting up the device where the model will run
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")


model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
model.to(device)
model.eval()

# Setting up Grad-CAM for the last convolutional layer
target_layers = [model.features[-1]]
grad_cam = GradCAM(model=model, target_layers=target_layers)

dataloader = DataLoader(dataset, batch_size=32)

rates = [0.0, 0.25, 0.5, 0.75, 1.0]

# Run drop in confidence
drop_values_per_rate, deletion_auc_score = run_drop_in_confidence(dataloader, device, grad_cam, rates, model)

# Plot drop in confidence graphic
plot_drop_in_confidence(rates=rates, drop_values_per_rate=drop_values_per_rate, deletion_auc_score=deletion_auc_score, output_dir=OUTPUT_DIR)

# Run insertion score
insertion_values_per_rate, insertion_auc_score = run_insertion_score(dataloader, device, grad_cam, rates, model)

# Plot insertion score graphic
plot_insertion_score(rates=rates, insertion_values_per_rate=insertion_values_per_rate, insertion_auc_score=insertion_auc_score, output_dir=OUTPUT_DIR)