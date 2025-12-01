import torch
import torch.nn as nn
import pennylane as qml
import numpy as np

# =========================================================================
# CONFIGURATION
# =========================================================================
N_QUBITS = 8             # Processing chunk size
N_SUB_CIRCUITS = 4       # 4 * 8 = 32 features total
N_LAYERS = 3             # Increased depth for Data Re-uploading (SOTA technique)

# Simulator Device
# We use "default.qubit" for exact state-vector simulation.
# For faster approximations on huge circuits, "lightning.qubit" requires C++ compilation.
dev = qml.device("default.qubit", wires=N_QUBITS)

# =========================================================================
# SECTION 1: SOTA QUANTUM CIRCUIT (Data Re-uploading)
# =========================================================================

@qml.qnode(dev, interface="torch")
def reuploading_circuit(inputs, weights):
    """
    Implements a Data Re-uploading Classifier.
    
    Theory: By encoding the input x into the circuit multiple times (interleaved 
    with variational weights), the circuit acts as a universal function approximator
    akin to a Fourier Series.
    
    Args:
        inputs (Tensor): Shape (8,) - The feature vector chunk.
        weights (Tensor): Shape (N_LAYERS, N_QUBITS, 3) - Trainable parameters.
    """
    
    # Loop over layers (Re-uploading)
    for l in range(N_LAYERS):
        # 1. Data Encoding (Re-uploading Step)
        # We map data to Y-rotations. 
        # Ideally, we might normalize inputs to [-pi, pi] before this call in the classical layer.
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation='Y')
        
        # 2. Trainable Variational Block
        # General Rotation (Rot) covers generic SU(2) operations (Rx, Ry, Rz combination)
        # weights[l] has shape (N_QUBITS, 3)
        for q in range(N_QUBITS):
            qml.Rot(*weights[l, q], wires=q)
        
        # 3. Entanglement (The Mixing Layer)
        # We use a Ring Topology (Periodic Boundary Conditions) for max connectivity
        # CNOT q0->q1, q1->q2, ..., q7->q0
        if N_QUBITS > 1:
            # Ring topology
            for i in range(N_QUBITS):
                qml.CNOT(wires=[i, (i + 1) % N_QUBITS])

    # 4. Measurement
    # We measure Pauli Z expectation to get a continuous signal [-1, 1]
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(N_QUBITS)]

# =========================================================================
# SECTION 2: THE KNITTING LAYER
# =========================================================================

class KnittedVQC(nn.Module):
    """
    PyTorch Module that manages the 'Circuit Knitting'.
    It splits the input vector, runs the simulation, and stitches results.
    """
    def __init__(self, n_sub_circuits=N_SUB_CIRCUITS, n_qubits=N_QUBITS, n_layers=N_LAYERS):
        super().__init__()
        self.n_sub_circuits = n_sub_circuits
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Weight Initialization
        # We need independent weights for each of the 4 sub-circuits.
        # Shape per circuit: (Layers, Qubits, 3 params per Rot gate)
        weight_shapes = (n_sub_circuits, n_layers, n_qubits, 3)
        
        # Initialize with uniform distribution for better convergence start
        # (Standard practice in QML to avoid barren plateaus logic, though random is often used)
        self.q_weights = nn.Parameter(torch.empty(weight_shapes).uniform_(0, 2 * np.pi))
        
        # Learnable scaling factor (Simulates 'classical post-processing' of quantum measurement)
        self.output_scale = nn.Parameter(torch.ones(n_sub_circuits * n_qubits))

    def forward(self, x):
        """
        Args:
            x (Tensor): Shape (Batch, 32)
        Returns:
            Tensor: Shape (Batch, 32)
        """
        batch_size = x.shape[0]
        
        # 1. Split Data (Knitting Cut)
        # Chunks is a tuple of 4 tensors, each (Batch, 8)
        chunks = torch.split(x, self.n_qubits, dim=1)
        
        results = []
        
        # 2. Run Parallel Simulations
        for i in range(self.n_sub_circuits):
            input_chunk = chunks[i]       # (Batch, 8)
            circuit_weights = self.q_weights[i] # (Layers, 8, 3)
            
            # Process the batch
            # We use a list comprehension here. While `torch.vmap` is experimental in 
            # some PennyLane versions, this approach is the most stable for standard installs.
            
            # Optimization Note: For large batches in simulation, 
            # PennyLane's 'broadcast' feature or 'lightning.qubit' is preferred.
            # Here we keep it explicit for clarity and Torch gradient safety.
            circuit_out = torch.stack([
                reuploading_circuit(x_sample, circuit_weights) 
                for x_sample in input_chunk
            ])
            
            results.append(circuit_out)
            
        # 3. Stitch (Knitting Recombination)
        # (Batch, 32)
        knitted_raw = torch.cat(results, dim=1)
        
        # 4. Final Scaling (Hybrid Trick)
        # Raw quantum output is [-1, 1]. 
        # We multiply by a learnable scalar to match the distribution expected by the next Dense layer.
        return knitted_raw * self.output_scale

if __name__ == "__main__":
    # === Simulation Test ===
    print("Initializing SOTA Knitted Quantum Layer...")
    model = KnittedVQC()
    
    # Create dummy input: Batch of 2, 32 Features
    # Normalized roughly to [-pi, pi] range for optimal encoding
    dummy_input = torch.randn(2, 32) 
    
    print(f"Running forward pass on device: {dev.name}...")
    output = model(dummy_input)
    
    print("\n--- Dimensions ---")
    print(f"Input:  {dummy_input.shape}")
    print(f"Output: {output.shape}")
    
    print("\n--- Gradient Check ---")
    # Calculate a dummy loss and backprop to ensure the quantum circuit is differentiable
    loss = output.mean()
    loss.backward()
    print(f"Gradients on weights: {model.q_weights.grad is not None}")
    print(f"Grad Norm: {model.q_weights.grad.norm().item():.4f}")
    print("✅ Quantum Simulation & Gradient Flow Successful")