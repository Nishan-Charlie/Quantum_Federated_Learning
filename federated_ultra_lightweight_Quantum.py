import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import torchvision
from torchvision import transforms
import numpy as np
from quantum_layer import KnittedVQC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import csv
import os
import copy
from tqdm import tqdm

# Import the Ultra Lightweight model structure
from Ultra_Lightweight import UltraLightEfficentNet_L1

# Hybrid Ultra Lightweight Model (Same as original)
class HybridUltraLight(nn.Module):
    def __init__(self):
        super(HybridUltraLight, self).__init__()
        
        # Initialize the Ultra Lightweight model
        self.backbone = UltraLightEfficentNet_L1(
            num_classes=1000, 
            image_size=224,   
            dims=[48, 64], 
            channels=[8, 16, 32, 48, 288]
        )
        
        # Input features to the classifier
        in_features = 288 # channels[4]
        
        self.backbone.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            KnittedVQC(n_sub_circuits=8, n_qubits=8, n_layers=3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.backbone(x)

def train_client(model, loader, device, criterion, optimizer, epochs=1):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for epoch in range(epochs):
        for inputs, labels in loader:
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
            
    # Calculate metrics for the last epoch roughly
    epoch_loss = running_loss / (len(loader.dataset) * epochs)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return model.state_dict(), epoch_loss, accuracy

def evaluate(loader, model, device, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
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

def fed_avg(global_model_state, client_weights):
    """
    Performs Federated Averaging on a list of client state dicts.
    Assumes equal weighting for simplicity (can be modified for weighted avg based on dataset size).
    """
    avg_state = copy.deepcopy(global_model_state)
    
    for key in avg_state.keys():
        # Initialize with float tensors to avoid integer overflow/rounding during summation
        if 'num_batches_tracked' in key:
            # Handle num_batches_tracked separately, usually by taking the max or sum, 
            # but usually it's better to just keep the global one or max.
            # For simplicity, we'll take the first client's value or keep global.
            # PyTorch BN tracking is tricky in FL. Often ignored or averaged.
            # Let's simple average for now to be safe or just cast to Float.
             avg_state[key] = torch.stack([c[key].float() for c in client_weights]).mean().long()
        else:
             avg_state[key] = torch.stack([c[key].float() for c in client_weights]).mean()
             
        # Convert back to original type if needed (though mean() usually returns float)
        # Weights should be float usually.
        
    return avg_state

def plot_metrics(history, save_dir='plots_fed_ultra'):
    os.makedirs(save_dir, exist_ok=True)
    rounds = range(1, len(history['train_loss']) + 1)
    
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1']
    for metric in metrics:
        plt.figure()
        plt.plot(rounds, history[f'train_{metric}'], label=f'Avg Client {metric.capitalize()}')
        plt.plot(rounds, history[f'valid_{metric}'], label=f'Global Valid {metric.capitalize()}')
        plt.title(f'Federated Training and Validation {metric.capitalize()}')
        plt.xlabel('Rounds')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.savefig(os.path.join(save_dir, f'{metric}.png'))
        plt.close()

if __name__ == '__main__':
    
    # Configuration
    NUM_CLIENTS = 5
    NUM_ROUNDS = 20
    LOCAL_EPOCHS = 5
    LR = 0.0001
    DATA_DIR = 'chest_xray'
    SAVE_DIR = 'plots_fed_ultra'
    
    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Transforms (Same as original)
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
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # Load Datasets
    full_train_dataset = torchvision.datasets.ImageFolder(root=f"{DATA_DIR}/train", transform=data_transforms['train'])
    valid_dataset = torchvision.datasets.ImageFolder(root=f"{DATA_DIR}/valid", transform=data_transforms['valid'])
    
    # Split training data for clients
    total_train_size = len(full_train_dataset)
    subset_size = total_train_size // NUM_CLIENTS
    # Handle remainder
    lengths = [subset_size] * NUM_CLIENTS
    lengths[0] += total_train_size % NUM_CLIENTS
    
    client_datasets = random_split(full_train_dataset, lengths)
    
    client_loaders = [
        DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)
        for ds in client_datasets
    ]
    
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Initialize Global Model
    global_model = HybridUltraLight().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Metrics history
    history = {
        'train_loss': [], 'train_accuracy': [], 'train_precision': [], 'train_recall': [], 'train_f1': [],
        'valid_loss': [], 'valid_accuracy': [], 'valid_precision': [], 'valid_recall': [], 'valid_f1': []
    }
    
    # CSV file setup
    csv_file = 'training_metrics_fed_ultra.csv'
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Round', 'Avg Client Loss', 'Avg Client Acc', 'Valid Loss', 'Valid Acc', 'Valid Prec', 'Valid Recall', 'Valid F1'])

    print(f"Starting Federated Learning with {NUM_CLIENTS} clients for {NUM_ROUNDS} rounds.")

    for round_num in range(NUM_ROUNDS):
        print(f"\n--- Round {round_num + 1}/{NUM_ROUNDS} ---")
        
        global_state = global_model.state_dict()
        client_weights = []
        client_losses = []
        client_accs = []
        
        # Train Clients
        for i in range(NUM_CLIENTS):
            # Create local model copy
            local_model = HybridUltraLight().to(device)
            local_model.load_state_dict(copy.deepcopy(global_state))
            
            # Optimizer for local model
            local_optimizer = optim.AdamW(local_model.parameters(), lr=LR, weight_decay=0.005)
            
            print(f"Training Client {i+1}...")
            w, loss, acc = train_client(local_model, client_loaders[i], device, criterion, local_optimizer, epochs=LOCAL_EPOCHS)
            
            client_weights.append(w)
            client_losses.append(loss)
            client_accs.append(acc)
        
        # Aggregate
        avg_client_loss = np.mean(client_losses)
        avg_client_acc = np.mean(client_accs)
        
        new_global_weights = fed_avg(global_state, client_weights)
        global_model.load_state_dict(new_global_weights)
        
        # Evaluate Global Model
        print("Evaluating Global Model...")
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(valid_loader, global_model, device, criterion)
        
        print(f"Round {round_num+1} Result:")
        print(f"  Avg Client Loss: {avg_client_loss:.4f}, Acc: {avg_client_acc:.4f}")
        print(f"  Global Valid Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        # Save history (Approximating training metrics with client averages)
        history['train_loss'].append(avg_client_loss)
        history['train_accuracy'].append(avg_client_acc)
        # Precision/Recall/F1 for training not easily aggregated without more data, skipping or using simplified
        history['train_precision'].append(0) 
        history['train_recall'].append(0)
        history['train_f1'].append(0)
        
        history['valid_loss'].append(val_loss)
        history['valid_accuracy'].append(val_acc)
        history['valid_precision'].append(val_prec)
        history['valid_recall'].append(val_rec)
        history['valid_f1'].append(val_f1)
        
        # Write to CSV
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([round_num + 1, avg_client_loss, avg_client_acc, val_loss, val_acc, val_prec, val_rec, val_f1])
            
    # Save Model
    torch.save(global_model.state_dict(), 'hybrid_ultralight_quantum_fed.pth')
    print("Federated Learning Completed. Model Saved.")
    
    # Plotting
    plot_metrics(history, save_dir=SAVE_DIR)
