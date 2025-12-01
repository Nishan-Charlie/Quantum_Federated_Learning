import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np

# =========================================================================
# CUSTOM DATASET WRAPPER (The Patching Engine)
# =========================================================================

class PatchedDataset(Dataset):
    """
    Wraps a standard dataset and performs the 32x32 patching on the fly.
    """
    def __init__(self, underlying_dataset):
        self.dataset = underlying_dataset

    def __getitem__(self, index):
        img, label = self.dataset[index]
        # img shape is (1, 224, 224) - assuming Grayscale
        
        # PATCHING LOGIC:
        # We want (49, 1, 32, 32)
        # 1. Unfold Height: (1, 7, 32, 224)
        # 2. Unfold Width:  (1, 7, 32, 7, 32)
        # 3. Permute/Reshape to merge the 7x7 grid into a sequence of 49
        
        # Using torch.unfold is efficient
        # Input: (C, H, W)
        kc, kh, kw = 1, 32, 32  # Kernel size
        dc, dh, dw = 1, 32, 32  # Stride
        
        # Unfold creates patches
        patches = img.unfold(1, kh, dh).unfold(2, kw, dw)
        # Shape becomes: (1, 7, 7, 32, 32)
        
        # Reshape to (49, 1, 32, 32)
        # We permute to ensure the order is row-major (top-left to bottom-right)
        patches = patches.permute(1, 2, 0, 3, 4).contiguous()
        patches = patches.view(49, 1, 32, 32)
        
        return patches, label

    def __len__(self):
        return len(self.dataset)

# =========================================================================
# FEDERATED DATA LOADER
# =========================================================================

def load_partitioned_data(num_clients: int, batch_size: int):
    """
    1. Loads MNIST (as a proxy for X-Ray for easy testing) or Real X-Ray data.
    2. Splits it into 'num_clients' partitions.
    3. Returns a list of DataLoaders (one for each client).
    """
    
    # --- 1. PREPROCESSING ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Resize to standard
        transforms.Grayscale(num_output_channels=1), # Ensure 1 channel
        transforms.ToTensor(),
        # Normalize to [0,1] (approx mean/std for medical images)
        transforms.Normalize((0.5,), (0.5,)) 
    ])

    # --- 2. LOAD DATASET ---
    # Load CT Scan Data from the 'train' folder
    # We assume the structure: CT_Scan/train/Pos and CT_Scan/train/Neg
    data_dir = r"D:\\Quantum Federated Learning\\CT_Scan\\train"
    print(f"Loading Dataset from: {data_dir}")
    
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Wrap in Patcher
    patched_dataset = PatchedDataset(full_dataset)

    # --- 3. PARTITIONING (Splitting for FL) ---
    # Calculate size per client
    partition_size = len(patched_dataset) // num_clients
    lengths = [partition_size] * num_clients
    
    # Handle remainder if dataset size isn't perfectly divisible
    remainder = len(patched_dataset) - sum(lengths)
    if remainder > 0:
        lengths[0] += remainder

    # Randomly split the data to simulate different hospitals having different patients
    datasets_split = random_split(patched_dataset, lengths)

    # --- 4. CREATE LOADERS ---
    trainloaders = []
    valloaders = []
    
    for ds in datasets_split:
        # Further split local data into Train/Val (80/20)
        len_train = int(len(ds) * 0.8)
        len_val = len(ds) - len_train
        ds_train, ds_val = random_split(ds, [len_train, len_val])
        
        trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=batch_size))
        
    return trainloaders, valloaders

if __name__ == "__main__":
    # Test Patching
    loader, _ = load_partitioned_data(num_clients=2, batch_size=1)
    images, labels = next(iter(loader[0]))
    print(f"Data Shape Check: {images.shape}") 
    # Expected: (1, 49, 1, 32, 32)