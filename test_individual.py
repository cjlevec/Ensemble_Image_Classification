import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18, alexnet, vgg16
from torchvision.datasets import CIFAR100
import argparse
import os


def main():
    n_epochs = 30
    batch_size = 32
    pre_train = True

    # Read arguments from command line
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-p', metavar='path', type=str, required=True,
                           help="Provide path to folder with weights (e.g., '/path/to/weights/')")
    argParser.add_argument('-e', metavar='epochs', type=str, choices=['full', 'five'], default='full',
                           help="Number of epochs - 'full' for full convergence and 'five' for 5-epoch weights")
    argParser.add_argument('-m', metavar='model', type=str, choices=['vgg16', 'resnet18', 'alexnet'], required=True,
                           help="The model being tested: 'vgg16', 'resnet18' or 'alexnet'")

    args = argParser.parse_args()

    path = args.p
    epochs = args.e
    modelType = args.m

    # Check for GPU availability and set device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print(f"Using device: {device}")

    # Load CIFAR-100 dataset with transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match model input size
        transforms.ToTensor(),
    ])
    # Load CIFAR dataset
    dataset = CIFAR100(root='./data', train=False, download=True, transform=transform)

    # Split dataset and only use validation set
    val_size = int(0.2 * len(dataset))
    # Use _, to ignore the training set
    _, valset = random_split(dataset, [len(dataset) - val_size, val_size])
    valloader = DataLoader(valset, batch_size=32, shuffle=False)

    # Initialize model and adjust final layer for 100 classes
    if modelType == "resnet18":  # Resnet18 testing
        model = resnet18(pretrained=False)  # Set pretrained to False as we are now loading our trained weights
        model.fc = torch.nn.Linear(model.fc.in_features, 100)  # Adjust final layer for CIFAR-100
        filename = "resnet18_5_epochs.pth" if epochs == "five" else "resnet18_final.pth"
        print(f"Testing ResNet18 with {'5 epochs' if epochs == 'five' else 'full convergence'}")

    elif modelType == "alexnet":  # AlexNet testing
        model = alexnet(pretrained=False)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 100)
        filename = "alexnet_5_epochs.pth" if epochs == "five" else "alexnet_final.pth"
        print(f"Testing AlexNet with {'5 epochs' if epochs == 'five' else 'full convergence'}")

    elif modelType == "vgg16":  # VGG16 testing
        model = vgg16(pretrained=False)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 100)
        filename = "vgg16_5_epochs.pth" if epochs == "five" else "vgg16_final.pth"
        print(f"Testing VGG16 with {'5 epochs' if epochs == 'five' else 'full convergence'}")

    # Load previous model weights
    weight_path = os.path.join(path, filename)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    # Send model to device
    model = model.to(device)
    model.eval()

    # Variables to store error counts
    total_images = 0
    top1_incorrect = 0
    top5_incorrect = 0

    # Validation phase to calculate Top-1 and Top-5 error rates
    with torch.no_grad(): # Don't update gradients during validation
        for inputs, labels in valloader:
            # Move data to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Get the top 5 predictions for each image in the batch
            _, top5_preds = outputs.topk(5, dim=1)

            # Calculate Top-1 error (this is the % of time the correct answer is not in the highest likely prediction)
            top1_preds = torch.argmax(outputs, dim=1)  # Get top-1 predictions
            incorrect_preds = (top1_preds != labels).sum()  # Count incorrect predictions
            top1_incorrect += incorrect_preds.item()  # Add to the total incorrect count

            # Calculate Top-5 error (this is the % of time the correct answer is not in the top 5 predictions)
            expanded_labels = labels.view(-1, 1).expand_as(top5_preds)  # Expand labels to match top-5 predictions shape
            correct_in_top5 = top5_preds.eq(expanded_labels)  # Check if any top-5 predictions match the labels

            # Count where correct is not in top 5
            incorrect_top5 = 0
            for i in range(correct_in_top5.size(0)):  # Iterate over the batch
                if correct_in_top5[i].sum() == 0:  # If no correct predictions in top 5
                    incorrect_top5 += 1

            top5_incorrect += incorrect_top5

            # Update total images count
            total_images += labels.size(0)

    # Calculate Top-1 and Top-5 error rates
    top1_error_rate = 100 * top1_incorrect / total_images
    top5_error_rate = 100 * top5_incorrect / total_images

    print(f"Top-1 Error Rate: {top1_error_rate:.2f}%")
    print(f"Top-5 Error Rate: {top5_error_rate:.2f}%")

if __name__ == '__main__':
    main()
