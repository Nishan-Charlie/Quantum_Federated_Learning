import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import flwr as fl
from collections import OrderedDict

# Import your custom modules
from hybrid_encoder import HybridQuantumBackbone
from quantum_layer import KnittedVQC

# =========================================================================
# SECTION 1: THE FULL HYBRID MODEL
# =========================================================================

class MedicalQuantumClassifier(nn.Module):
    """
    The complete End-to-End Model:
    Input (Images) -> Hybrid Encoder -> Quantum Knitting -> Class Prediction
    """
    def __init__(self):
        super().__init__()
        
        # 1. The Classical Bridge (CNN + Transformer)
        # Input: (B, 49, 1, 32, 32) -> Output: (B, 32)
        self.encoder = HybridQuantumBackbone(
            input_channels=1,
            cnn_feature_dim=64,
            transformer_dim=96,
            quantum_feature_dim=32 
        )
        
        # 2. The Quantum Knitting Layer
        # Input: (B, 32) -> Output: (B, 32) (Knitted Quantum Features)
        self.quantum_layer = KnittedVQC(n_sub_circuits=4, n_qubits=8)
        
        # 3. The Final Classification Head
        # Input: (B, 32) -> Output: (B, 1) (Probability of Disease)
        self.head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Flow: x -> Encoder -> Quantum -> Head
        features = self.encoder(x)
        q_features = self.quantum_layer(features)
        return self.head(q_features)

# =========================================================================
# SECTION 2: LOCAL TRAINING HELPERS
# =========================================================================

def train(net, trainloader, epochs, device):
    """Local training loop for the Virtual Client."""
    criterion = nn.BCELoss()
    # We use Adam. Note: Learning rate for Quantum usually needs tuning (0.01 - 0.001)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def test(net, testloader, device):
    """Evaluation loop for the Virtual Client."""
    criterion = nn.BCELoss()
    correct, total, loss = 0, 0, 0.0
    
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            
            # Binary accuracy threshold at 0.5
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = correct / total
    return loss / len(testloader), accuracy

# =========================================================================
# SECTION 3: THE FLOWER CLIENT
# =========================================================================

class Client(fl.client.NumPyClient):
    """
    The Flower Client class.
    In simulation, 'main.py' spawns instances of this class on the Server.
    """
    def __init__(self, trainloader, testloader, device):
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.model = MedicalQuantumClassifier().to(device)

    def get_parameters(self, config):
        """Return local model parameters to the server."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Update local model with global parameters from the server."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """
        Train the model on local data.
        Returns: Updated parameters, Number of samples, Metrics
        """
        self.set_parameters(parameters)
        
        # Read config from server (e.g., "local_epochs": 1)
        epochs = config.get("local_epochs", 1)
        
        train(self.model, self.trainloader, epochs=epochs, device=self.device)
        
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        """
        Evaluate the model on local holdout data.
        """
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.testloader, device=self.device)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}