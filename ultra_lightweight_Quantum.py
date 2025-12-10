import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import numpy as np
from quantum_layer import KnittedVQC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import csv
import os
from tqdm import tqdm
import argparse

# Import the Ultra Lightweight model
from Ultra_Lightweight import UltraLightEfficentNet_L1

# Hybrid Ultra Lightweight Model
class HybridUltraLight(nn.Module):
    def __init__(self, num_classes=2, n_quantum_layers=7):
        super(HybridUltraLight, self).__init__()
        
        # Initialize the Ultra Lightweight model
        # using parameters from the example in Ultra_Lightweight.py
        self.backbone = UltraLightEfficentNet_L1(
            num_classes=1000, # Dummy class count, we will replace the classifier
            image_size=224,   # Matching our transforms
            dims=[48, 64], 
            channels=[8, 16, 32, 48, 288]
        )
        
        # The classifier in UltraLightEfficentNet_L1 is:
        # nn.Sequential(Flatten, Dropout, Linear(channels[4], num_classes))
        # We want to intercept features before the final Linear layer or replace the whole classifier.
        # The backbone.classifier[2] is the Linear layer.
        # The input to the classifier is (B, channels[4]) -> (B, 288) based on config.
        
        # Replace the classifier with our Quantum head
        # We keep Flatten and Dropout if desired, or build our own structure.
        # Let's rebuild the classifier structure.
        
        # Input features to the classifier
        in_features = 288 # channels[4]
        
        self.backbone.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            KnittedVQC(n_sub_circuits=8, n_qubits=8, n_layers=n_quantum_layers),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_epoch(loader, model, device, criterion, optimizer):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for inputs, labels in tqdm(loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
            
    epoch_loss = running_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return epoch_loss, accuracy, precision, recall, f1

def evaluate(loader, model, device, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    epoch_loss = running_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return epoch_loss, accuracy, precision, recall, f1

def plot_metrics(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)
    
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1']
    for metric in metrics:
        plt.figure()
        plt.plot(epochs, history[f'train_{metric}'], label=f'Train {metric.capitalize()}')
        plt.plot(epochs, history[f'valid_{metric}'], label=f'Valid {metric.capitalize()}')
        plt.title(f'Training and Validation {metric.capitalize()}')
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.savefig(os.path.join(save_dir, f'{metric}.png'))
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Hybrid Ultra Lightweight Quantum Model')
    parser.add_argument('--classes', type=int, default=7, help='Number of output classes (default: 2)')
    parser.add_argument('--quantum_layers', type=int, default=7, help='Number of quantum layers (default: 7)')
    parser.add_argument('--dataset_path', type=str, default='ISIC 2018', help='Path to dataset (default: chest_xray)')
    parser.add_argument('--output_dir', type=str, default='outputs_ultra', help='Directory to save outputs (default: outputs_ultra)')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate (default: 0.0001)')
    
    args = parser.parse_args()
    
    data_dir = args.dataset_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, 'plots_ultra')

    data_transforms = {
        
        'train': transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, shear=10, scale=(0.9, 1.1)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    image_datasets = {
        x: torchvision.datasets.ImageFolder(root=f"{data_dir}/{x}", transform=data_transforms[x])
        for x in ['train', 'valid', 'test']
    }

    # Set num_workers=0 to avoid multiprocessing issues with quantum simulator
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=32, shuffle=True if x == 'train' else False, num_workers=0)
        for x in ['train', 'valid', 'test']
    }

    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset: {data_dir}")
    print(f"Output Config: {output_dir}")
    print(f"Model Config: {args.classes} classes, {args.quantum_layers} quantum layers")
    print(f"Training Config: LR={args.lr}")

    # Initialize model, loss, and optimizer
    model = HybridUltraLight(num_classes=args.classes, n_quantum_layers=args.quantum_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.005)
    
    # Print trainable parameters to verify
    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}")
    
    # Early Stopping
    early_stopping = EarlyStopping(patience=20, min_delta=0.0001)

    # Metrics history
    history = {
        'train_loss': [], 'train_accuracy': [], 'train_precision': [], 'train_recall': [], 'train_f1': [],
        'valid_loss': [], 'valid_accuracy': [], 'valid_precision': [], 'valid_recall': [], 'valid_f1': []
    }
    
    # CSV file setup
    csv_file = os.path.join(output_dir, 'training_metrics_ultra.csv')
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Train Prec', 'Train Recall', 'Train F1',
                         'Valid Loss', 'Valid Acc', 'Valid Prec', 'Valid Recall', 'Valid F1'])

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch(dataloaders['train'], model, device, criterion, optimizer)
        
        # Validation
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(dataloaders['valid'], model, device, criterion)
        
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Valid - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_acc)
        history['train_precision'].append(train_prec)
        history['train_recall'].append(train_rec)
        history['train_f1'].append(train_f1)
        
        history['valid_loss'].append(val_loss)
        history['valid_accuracy'].append(val_acc)
        history['valid_precision'].append(val_prec)
        history['valid_recall'].append(val_rec)
        history['valid_f1'].append(val_f1)
        
        # Write to CSV
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, train_loss, train_acc, train_prec, train_rec, train_f1,
                             val_loss, val_acc, val_prec, val_rec, val_f1])
            
        # Early Stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Plotting
    print("Plotting metrics...")
    plot_metrics(history, save_dir=plots_dir)

    # Testing after training
    print("Testing model...")
    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(dataloaders['test'], model, device, criterion)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")

    # Save the model
    torch.save(model.state_dict(), os.path.join(output_dir, 'hybrid_ultralight_quantum.pth'))
