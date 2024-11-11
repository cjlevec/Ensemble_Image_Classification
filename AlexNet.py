import torch
from torchvision import models, transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import torch.nn as nn


num_epochs = 30

# load pretrained AlexNet model from pytorch
alexnet = models.alexnet(pretrained=True)

# we have to modify the classifier to have 100 output classes for CIFAR-100
alexnet.classifier[6] = nn.Linear(4096, 100)

# define the transformation
# AlexNet needs imgs of 224x224
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# load CIFAR-100 dataset
train_dataset = CIFAR100(root='./data', train=True, transform=transform, download=True)
test_dataset = CIFAR100(root='./data', train=False, transform=transform, download=True)


# initialize dataloaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

alexnet = alexnet.to(device)

# use cross entropy loss and Adam optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(alexnet.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs): # trying 30 epochs first
    alexnet.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = alexnet(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Testing loop
alexnet.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = alexnet(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy on test set: {100 * correct / total:.2f}%')
