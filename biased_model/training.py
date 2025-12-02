import torch
import numpy as np
from torch import nn
from model import SimpleCNN
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Loading data based on directory structure and names
train_dataset = datasets.ImageFolder(root='dataset/train', transform=transform)

print(train_dataset.class_to_idx) 
# {'cat': 0, 'dog': 1}

# Instantiating the model and configuring the parameters
model = SimpleCNN(dropout_rate=0.3)
num_epochs = 10
learning_rate = 0.001
batch_size = 32
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training the model
model.to(device)
model.train()
for epoch in range(num_epochs):
    train_losses = []
    train_avg_losses = []
    for X_batch, y_batch in train_dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()

        outputs = model.forward(X_batch)
      
        loss = loss_function(outputs, y_batch.view(-1))

        loss.backward()

        optimizer.step()

        train_losses.append(loss.item())

    train_avg_loss = np.mean(train_losses)
    train_avg_losses.append(train_avg_loss)
    print(f'Epoch#{epoch+1}: Train Average Loss = {train_avg_loss:.5f}')

torch.save(model.state_dict(), "pretrained_model.pth")