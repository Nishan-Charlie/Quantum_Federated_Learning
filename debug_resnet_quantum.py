import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import cv2
import numpy as np
from resnet50_Quantum import HybridResNet50, tiff_loader
import os

def debug_model():
    print("Starting Debugging Session...")
    
    # 1. Data Loading & Distribution Check
    data_dir = 'D:\\Quantum Federated Learning\\CT_Scan'
    
    # Use simple transforms for debugging
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print("\n--- Checking Data Distribution ---")
    for split in ['train', 'valid']:
        dataset = torchvision.datasets.ImageFolder(root=f"{data_dir}/{split}", transform=data_transforms, loader=tiff_loader)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Count classes
        targets = dataset.targets
        class_counts = np.bincount(targets)
        print(f"{split.capitalize()} Set: Total={len(dataset)}")
        for i, count in enumerate(class_counts):
            print(f"  Class {i} ({dataset.classes[i]}): {count}")
            
    # 2. Model Initialization
    print("\n--- Initializing Model ---")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = HybridResNet50().to(device)
    
    # 3. Forward Pass Check
    print("\n--- Running Forward Pass on Validation Batch ---")
    # Get one batch
    inputs, labels = next(iter(loader))
    inputs, labels = inputs.to(device), labels.to(device)
    
    model.train() # Set to train to check gradients later
    outputs = model(inputs)
    
    print(f"Input Batch Shape: {inputs.shape}")
    print(f"Output Logits Shape: {outputs.shape}")
    print(f"First 5 Logits:\n{outputs[:5].detach().cpu().numpy()}")
    
    # Check predictions
    _, preds = torch.max(outputs, 1)
    print(f"Predicted Classes: {preds.cpu().numpy()}")
    print(f"Actual Labels:     {labels.cpu().numpy()}")
    
    # Check if all predictions are the same
    unique_preds = torch.unique(preds)
    if len(unique_preds) == 1:
        print("WARNING: Model is predicting only one class!")
    else:
        print("Model is predicting multiple classes.")
        
    # 4. Gradient Check
    print("\n--- Checking Gradients ---")
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)
    loss.backward()
    
    print(f"Loss: {loss.item():.4f}")
    
    # Check gradients in quantum layer weights
    if model.qlayer.q_weights.grad is not None:
        grad_norm = model.qlayer.q_weights.grad.norm().item()
        print(f"Quantum Layer Weights Gradient Norm: {grad_norm:.6f}")
        if grad_norm == 0:
             print("WARNING: Quantum layer gradients are ZERO!")
    else:
        print("WARNING: Quantum layer weights have NO gradient!")
        
    # Check gradients in final fc layer
    if model.fc.weight.grad is not None:
         print(f"Final FC Layer Gradient Norm: {model.fc.weight.grad.norm().item():.6f}")

if __name__ == "__main__":
    try:
        debug_model()
    except Exception as e:
        print(f"An error occurred: {e}")
