import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torchvision.models import alexnet
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader
import argparse


def main():

    # Read arguments from command line
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-e', metavar='epochs', type=int, help='# of epochs [30]')
    argParser.add_argument('-b', metavar='batch size', type=int, help='batch size [32]')
    argParser.add_argument('-p', metavar='use pre-trained weights? (BOOL)', type=bool, help='Was trained on pre-trained weights')

    args = argParser.parse_args()

    n_epochs = args.e if args.e else 30
    batch_size = args.b if args.b else 32
    pre_train = args.p if args.p else True

    print('\t\tn epochs = ', n_epochs)
    print('\t\tbatch size = ', batch_size)
    print('\t\tuse pre-trained weights? (BOOL) = ', pre_train)


    # Check if GPU is available and set device
    device = "cpu"
    if torch.cuda.is_available():
      device = "cuda"

    print(f"Using device: {device}")

    # Load CIFAR-100 dataset with transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match AlexNet input size
        transforms.ToTensor(),
    ])

    # Load CIFAR dataset
    dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

    # Split dataset into training and validation sets (80% train 20% validation)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    trainset, valset = random_split(dataset, [train_size, val_size])


    # Initialize dataloaders to batch data and prepare it for training
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    valloader = DataLoader(valset, batch_size=32, shuffle=False)

    # AlexNet was previously trained on ImageNet, this uses those weights as a starting point
    model = alexnet(pretrained=True)

    # Adjust final layer (#6) for 100 classes
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 100)

    # Send model to device to prepare for training
    model = model.to(device)

    # Cross entropy loss is standard for classification applications where class probability outputs are a focus
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # Small learning rate makes training more stable

    # 30 epochs is sufficient for our application
    epochs = 30
    # Arrays to hold loss values over each epoch
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')

        # Training phase
        model.train()
        # Reset running loss value at start of each epoch
        running_loss = 0.0
        for inputs, labels in trainloader:
            # Move data to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()         # Zero the parameter gradients
            outputs = model(inputs)       # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            loss.backward()               # Backward pass
            optimizer.step()              # Update weights
            running_loss += loss.item()   # Accumulate loss

        # Save the average training loss for this epoch
        train_epoch_loss = running_loss / len(trainloader)
        train_losses.append(train_epoch_loss)

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad(): # Don't update gradients during validation
            for inputs, labels in valloader:
                # Move data to device
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)     # Forward pass
                loss = criterion(outputs, labels)   # Calculate loss
                val_running_loss += loss.item()     # Accumulate validation loss

        # Save the average validation loss for this epoch
        val_epoch_loss = val_running_loss / len(valloader)
        val_losses.append(val_epoch_loss)

        print(f"Train Loss: {train_epoch_loss:.4f}, Validation Loss: {val_epoch_loss:.4f}")

        # Save model parameters after 5 epochs
        if epoch == 4: # (account for fact that the epoch loop started at 0)
            torch.save(model.state_dict(), "alexnet_5_epochs.pth")
            print("Checkpoint saved at 5 epochs as alexnet_5_epochs.pth")

    # Save the final model parameters after full training
    torch.save(model.state_dict(), "alexnet_final.pth")
    print("Model training complete and saved as alexnet_final.pth")

    # Plot the loss curves for training and validation
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Training and Validation Loss for AlexNet over 30 Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("alexnet_train_val_loss.png")
    plt.show()

if __name__ == '__main__':
    main()