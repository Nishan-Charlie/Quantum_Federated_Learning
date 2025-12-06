import flwr as fl
import torch
from collections import OrderedDict
import numpy as np

# Import our components
from client import Client, MedicalQuantumClassifier
from data_setup import load_partitioned_data

# =========================================================================
# CONFIGURATION
# =========================================================================
NUM_CLIENTS = 5        # Number of hospitals
BATCH_SIZE = 32         # Keep small for Quantum Simulation speed
NUM_ROUNDS = 50         # Number of Global FL Rounds
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Starting Federated Quantum Simulation on: {DEVICE}")

# =========================================================================
# 1. PREPARE DATA PARTITIONS
# =========================================================================
# This generates a list of dataloaders, one pair (train/val) for each client ID
trainloaders, valloaders = load_partitioned_data(NUM_CLIENTS, BATCH_SIZE)


# =========================================================================
# 2. DEFINE CLIENT GENERATOR
# =========================================================================
def client_fn(cid: str) -> fl.client.Client:
    """
    This function creates a new Virtual Client when the simulation asks for one.
    'cid' is the Client ID (string "0", "1", "2", etc.)
    """
    # Map string CID to integer index
    idx = int(cid)
    
    # Get the specific data partition for this client
    trainloader = trainloaders[idx]
    testloader = valloaders[idx]

    # Create the Flower Client (containing the Quantum Model)
    return Client(trainloader, testloader, DEVICE)


# =========================================================================
# 3. DEFINE SERVER STRATEGY (AGGREGATION)
# =========================================================================
def weighted_average(metrics):
    """Aggregates accuracy metrics from multiple clients."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

# We use FedAvg (Federated Averaging)
# In a real setup, you might use FedAdagrad to handle Quantum gradient noise better.
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,              # Sample 100% of clients per round
    fraction_evaluate=1.0,         # Evaluate on 100% of clients
    min_fit_clients=2,             # Minimum clients to train
    min_available_clients=NUM_CLIENTS,
    evaluate_metrics_aggregation_fn=weighted_average, # Aggregate accuracy
)

# =========================================================================
# 4. START SIMULATION
# =========================================================================

if __name__ == "__main__":
    # Initialize Global Model (Optional: to pass initial parameters)
    # net = MedicalQuantumClassifier().to(DEVICE)
    # params = [val.cpu().numpy() for _, val in net.state_dict().items()]
    
    print("🚀 Launching Simulation...")
    
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        # Resources per virtual client:
        # If you have 1 GPU, set num_gpus to e.g. 0.25 to run 4 clients in parallel (conceptually)
        # If using CPU, set num_cpus=1
        client_resources={"num_cpus": 1, "num_gpus": 0.0 if DEVICE.type == 'cpu' else 0.25},
    )