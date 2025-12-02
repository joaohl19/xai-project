from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Images preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Loading data based on directory structure and names
dataset = datasets.ImageFolder(root='dataset/train', transform=transform)

print(dataset.class_to_idx) 
# {'dog': 0, 'cat': 1}

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)