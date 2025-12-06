import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, models
import cv2
import numpy as np
from quantum_layer import KnittedVQC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import csv
import os
from tqdm import tqdm

# Hybrid model
class HybridResNet50(nn.Module):
    def __init__(self):
        
        super(HybridResNet50, self).__init__()

        # Pre-trained ResNet50
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Freeze ResNet layers to use as feature extractor
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        # Unfreeze layer4 for fine-tuning
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        
        # Replace final fully connected layer to output 64 features (KnittedVQC: 8 sub-circuits * 8 qubits = 64)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 64)
        
        # Batch Normalization to stabilize inputs to quantum layer
        self.bn = nn.BatchNorm1d(64)
        
        # Quantum layer integrated via KnittedVQC
        self.qlayer = KnittedVQC(n_sub_circuits=8, n_qubits=8, n_layers=3)
        
        # Final classical layer for binary classification (2 classes) from KnittedVQC outputs 64 features
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        # Extract features with ResNet50
        x = self.resnet(x)
        # Apply Batch Norm
        x = self.bn(x)
        # Pass through quantum layer
        x = self.qlayer(x)
        # Final linear layer
        x = self.fc(x)
        return x

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

def plot_metrics(history, save_dir='plots'):
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
    
    data_dir = 'chest_xray'

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

    # Initialize model, loss, and optimizer
    model = HybridResNet50().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001,weight_decay=0.004)
    
    # Print trainable parameters to verify
    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}")
    
    # Early Stopping
    early_stopping = EarlyStopping(patience=10, min_delta=0.0001)

    # Metrics history
    history = {
        'train_loss': [], 'train_accuracy': [], 'train_precision': [], 'train_recall': [], 'train_f1': [],
        'valid_loss': [], 'valid_accuracy': [], 'valid_precision': [], 'valid_recall': [], 'valid_f1': []
    }
    
    # CSV file setup
    csv_file = 'training_metrics.csv'
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Train Prec', 'Train Recall', 'Train F1',
                         'Valid Loss', 'Valid Acc', 'Valid Prec', 'Valid Recall', 'Valid F1'])

    # Training loop
    num_epochs = 50
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
    plot_metrics(history)

    # Testing after training
    print("Testing model...")
    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(dataloaders['test'], model, device, criterion)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")

    # Save the model
    torch.save(model.state_dict(), 'hybrid_resnet50_quantum.pth')