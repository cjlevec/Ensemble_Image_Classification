import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18, alexnet, vgg16
from torchvision.datasets import CIFAR100
import argparse
import os


def main():
    # Read arguments from command line
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-p', metavar='path', type=str,
                           help="provide path to folder with weights")
    argParser.add_argument('-e', metavar='epochs', type=str,
                           help="number of epochs - 'full' for full convergence and 'five' for 5-epoch weights")
    argParser.add_argument('-m', metavar='method', type=str,
                           help="ensemble method - 'max' for maximum probability, 'maj' for majority voting, or 'avg' for average probability")

    args = argParser.parse_args()

    path = args.p if args.p else "path"
    epochs = args.e if args.e else "full"
    method = args.m if args.m else "max"

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

    # Display the selected training parameters
    print(f"Using parameters trained after {'5 epochs' if epochs == 'five' else 'full convergence'}")
    print(f"Using ensemble method: {method}")

    # Load each model with respective weights
    def load_model(model_class, filename, final_layer_size=100):
        model = model_class(pretrained=False)
        if model_class == resnet18:
            model.fc = torch.nn.Linear(model.fc.in_features, final_layer_size)
        elif model_class in [alexnet, vgg16]:
            model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, final_layer_size)
        model.load_state_dict(torch.load(os.path.join(path, filename), map_location=device))
        return model.to(device).eval()

    # Define model file names based on epochs
    model_files = {
        "ResNet": "resnet18_5_epochs.pth" if epochs == "five" else "resnet18_final.pth",
        "AlexNet": "alexnet_5_epochs.pth" if epochs == "five" else "alexnet_final.pth",
        "VGG16": "vgg16_5_epochs.pth" if epochs == "five" else "vgg16_final.pth",
    }

    # Load models
    models = [
        load_model(resnet18, model_files["ResNet"]),
        load_model(alexnet, model_files["AlexNet"]),
        load_model(vgg16, model_files["VGG16"]),
    ]

    # Ensemble Methods
    def ensemble_max_prob(outputs):
        return torch.max(torch.stack(outputs), dim=0).values

    def ensemble_avg_prob(outputs):
        return torch.mean(torch.stack(outputs), dim=0)

    def ensemble_majority_voting(outputs):
        predictions = [torch.argmax(output, dim=1) for output in outputs]  # Get top-1 predictions for each model
        stacked_preds = torch.stack(predictions, dim=1)

        # Calculate majority vote, but handle ties with ResNet preference
        majority_preds = []
        resnet_preds = stacked_preds[:, 0]  # Assuming ResNet is the first model in the list

        for i, pred_row in enumerate(stacked_preds):
            mode, count = torch.mode(pred_row, dim=0)
            if (pred_row == mode).sum() > 1:  # Check if there's a majority
                majority_preds.append(mode)
            else:
                # If there's a tie, default to the ResNet prediction
                majority_preds.append(resnet_preds[i])

        return torch.tensor(majority_preds, device=outputs[0].device)

    # Function to calculate Top-1 error rate
    def calculate_top1_error(predictions, labels):
        _, top1_preds = torch.max(predictions, dim=1)  # Top-1 predictions
        top1_incorrect = (top1_preds != labels).sum().item()
        return top1_incorrect

    # Initialize counters
    total_images = 0
    top1_incorrect = 0

    # Validation phase
    with torch.no_grad():
        for inputs, labels in valloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Gather predictions from each model
            outputs = [model(inputs) for model in models]

            # Apply ensemble method
            if method == "max":
                ensemble_preds = ensemble_max_prob(outputs)
            elif method == "avg":
                ensemble_preds = ensemble_avg_prob(outputs)
            elif method == "maj":
                ensemble_preds = ensemble_majority_voting(outputs)
            else:
                raise ValueError("Invalid ensemble method. Choose 'max', 'avg', or 'maj'.")

            # Calculate Top-1 error
            if method == "maj":
                top1_incorrect += (ensemble_preds != labels).sum().item()
            else:
                top1_incorrect += calculate_top1_error(ensemble_preds, labels)

            total_images += labels.size(0)

    # Calculate Top-1 error rate
    top1_error_rate = 100 * top1_incorrect / total_images

    print(f"Ensemble ({method.capitalize()} Probability) - Top-1 Error Rate: {top1_error_rate:.2f}%")

if __name__ == '__main__':
    main()
