import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# =========================================================================
# SECTION 1: CORE BLOCKS (Kept from your provided code)
# =========================================================================

class ConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, stride: int = 1, padding: int = 0,
                 use_norm: bool = True, use_act: bool = True, act_layer: nn.Module = nn.SiLU, bias: bool = False,
                 groups: int = 1, dilation: int = 1):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=bias,
                      groups=groups, dilation=dilation)
        ]
        if use_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if use_act:
            layers.append(act_layer())
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class DilatedBottleneck(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        if stride == 1:
            padding = dilation
        elif stride == 2:
            if dilation == 1: padding = 1
            elif dilation == 2: padding = 2
            else: padding = (2 * dilation - 1) // 2
        else:
            raise ValueError(f"Unsupported stride: {stride}")

        self.use_res = (in_ch == out_ch and stride == 1)
        self.block = nn.Sequential(
            ConvLayer(in_ch, mid_ch, kernel_size=1),
            ConvLayer(mid_ch, mid_ch, kernel_size=3, stride=stride, padding=padding,
                      groups=mid_ch, dilation=dilation),
            ConvLayer(mid_ch, out_ch, kernel_size=1, use_act=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_res and x.shape == out.shape:
            out = out + x
        return out

class LocalRepresentationBlock(nn.Module):
    def __init__(self, Cin: int, TransformerDim: int):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(Cin, Cin, kernel_size=3, padding=1, groups=Cin)
        self.pointwise_conv = nn.Conv2d(Cin, TransformerDim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_conv(x)
        return self.pointwise_conv(x)

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class LinearSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, attn_dropout: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        self.qkv_proj = ConvLayer(
            in_channels=embed_dim,
            out_channels=1 + (2 * embed_dim),
            bias=bias, kernel_size=1, use_norm=False, use_act=False,
        )
        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.out_proj = ConvLayer(
            in_channels=embed_dim,
            out_channels=embed_dim,
            bias=bias, kernel_size=1, use_norm=False, use_act=False,
        )
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, p_dim, n, d = x.shape
        x_reshaped = rearrange(x, 'b p_dim n d -> b d p_dim n')
        qkv = self.qkv_proj(x_reshaped)
        query, key, value = torch.split(qkv, [1, self.embed_dim, self.embed_dim], dim=1)
        
        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_dropout(context_scores)
        context_vector = key * context_scores
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)
        
        intermediate = F.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(intermediate)
        out = rearrange(out, 'b d p_dim n -> b p_dim n d')
        return out

class Transformer(nn.Module):
    def __init__(self, dim: int, depth: int, mlp_dim: int, dropout: float = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LinearSelfAttention(embed_dim=dim, attn_dropout=dropout),
                FeedForward(dim, mlp_dim, dropout)
            ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class EfficentAttentionBlock(nn.Module):
    def __init__(self, Cin: int, TransformerDim: int, Cout: int, depth: int = 2, patch_size: int = 1):
        super().__init__()
        self.patch_size = patch_size
        self.TransformerDim = TransformerDim
        self.Cin = Cin 
        
        self.local = LocalRepresentationBlock(Cin, TransformerDim)
        self.transformer = Transformer(
            dim=TransformerDim, depth=depth, mlp_dim=TransformerDim * 2, dropout=0.1
        )
        self.conv_proj = nn.Conv2d(TransformerDim, Cin, kernel_size=1)
        self.fusion = nn.Conv2d(Cin + Cin, Cout, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        res = x 
        
        # 1. Local Representations
        local_out = self.local(x) # (B, TransDim, H, W)
        
        # 2. Prepare Residual (Align channels)
        if C < self.TransformerDim:
            padding = torch.zeros(B, self.TransformerDim - C, H, W, device=x.device)
            res_padded = torch.cat([x, padding], dim=1)
        else:
            res_padded = x[:, :self.TransformerDim, :, :]

        # 3. Unfold (Patchify)
        # Note: For 7x7 grid and patch_size=1, this effectively treats every pixel as a token
        ph = pw = self.patch_size
        h_patches, w_patches = H // ph, W // pw
        
        local_patches = rearrange(local_out, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=ph, pw=pw)
        res_patches = rearrange(res_padded, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=ph, pw=pw)
        
        # 4. Transformer
        seq = local_patches + res_patches
        seq_t = self.transformer(seq)
        
        # 5. Fold back
        fm = rearrange(seq_t, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', ph=ph, pw=pw, h=h_patches, w=w_patches)
        
        # 6. Project & Fuse
        projected = self.conv_proj(fm)
        fused = torch.cat([res, projected], dim=1)
        out = self.fusion(fused)
        
        return out + res # Residual connection (assuming Cin == Cout)


# =========================================================================
# SECTION 2: THE PIPELINE-SPECIFIC ARCHITECTURE
# =========================================================================

class HybridQuantumBackbone(nn.Module):
    """
    1. Takes 49 patches (32x32).
    2. Processes each patch via a shared CNN Stem (PatchCNN).
    3. Arranges them into a 7x7 grid.
    4. Applies Hybrid Attention (Global Context).
    5. Reduces to 32 features for the Quantum Circuit.
    """
    def __init__(self, 
                 input_channels: int = 1, 
                 cnn_feature_dim: int = 64, 
                 transformer_dim: int = 96,
                 quantum_feature_dim: int = 32):
        super().__init__()
        
        # --- A. The "Patch CNN" (Step 2) ---
        # Processes a single 32x32 patch into a feature vector
        self.patch_cnn = nn.Sequential(
            # 32x32 -> 16x16
            ConvLayer(input_channels, 16, kernel_size=3, stride=2, padding=1),
            # 16x16 -> 8x8
            DilatedBottleneck(16, 32, 32, stride=2),
            # 8x8 -> 4x4
            DilatedBottleneck(32, 64, cnn_feature_dim, stride=2),
            # Global Pool: (B, C, 4, 4) -> (B, C, 1, 1)
            nn.AdaptiveAvgPool2d(1) 
        )
        
        self.cnn_feature_dim = cnn_feature_dim

        # --- B. The "Hybrid Bridge" (Step 3) ---
        # Processes the 7x7 Grid of features
        self.hybrid_mixer = EfficentAttentionBlock(
            Cin=cnn_feature_dim, 
            TransformerDim=transformer_dim, 
            Cout=cnn_feature_dim, 
            depth=2, 
            patch_size=1 # Treat every patch in the 7x7 grid as a token
        )
        
        # --- C. Quantum Prep (Step 4 Prep) ---
        # Reduce 7x7x64 -> 32 scalar features for the 8-qubit circuits
        self.quantum_adapter = nn.Sequential(
            ConvLayer(cnn_feature_dim, 64, kernel_size=3, stride=2, padding=1), # 7x7 -> 4x4
            nn.Flatten(), # 64 * 4 * 4 = 1024
            nn.Linear(1024, 128),
            nn.SiLU(),
            nn.Linear(128, quantum_feature_dim) # Final output: 32 floats
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input Shape: (Batch, 49, 1, 32, 32)
        B, NumPatches, C, H, W = x.shape
        
        # 1. Flatten Batch and Patches to process in parallel
        # (B * 49, 1, 32, 32)
        x_flat = x.view(B * NumPatches, C, H, W)
        
        # 2. Apply CNN Stem
        # Output: (B*49, 64, 1, 1)
        cnn_feats = self.patch_cnn(x_flat)
        
        # 3. Reshape back to Grid: (B, 64, 7, 7)
        # We know NumPatches is 49, which forms a 7x7 grid
        cnn_feats = cnn_feats.view(B, self.cnn_feature_dim, 7, 7)
        
        # 4. Apply Hybrid Transformer (Global correlations between lung patches)
        # Output: (B, 64, 7, 7)
        global_feats = self.hybrid_mixer(cnn_feats)
        
        # 5. Prepare for Quantum Circuit
        # Output: (B, 32)
        quantum_input = self.quantum_adapter(global_feats)
        
        return quantum_input


if __name__ == '__main__':
    from torchinfo import summary
    from calflops import calculate_flops
    
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate Model
    model = HybridQuantumBackbone(
        input_channels=1,          # Grayscale X-ray
        cnn_feature_dim=64,        # Features per patch
        transformer_dim=96,        # Internal Transformer width
        quantum_feature_dim=32     # 4 Groups x 8 Qubits = 32 params
    ).to(device)
    
    # Create dummy input: 1 Image split into 49 patches of 32x32
    # Shape: (Batch=1, Patches=49, Channels=1, H=32, W=32)
    dummy_input = torch.randn(1, 49, 1, 32, 32).to(device)
    
    # Forward Pass
    output = model(dummy_input)
    
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape (to Quantum Layer): {output.shape}") # Should be [1, 32]
    
    # Summary
    # Note: torchinfo might struggle with the view() operation in summary logic 
    # so we inspect the components or use calflops on the forward pass.
    print("\n--- Complexity Analysis ---")
    flops, macs, params = calculate_flops(
        model=model,
        kwargs={'x': dummy_input}, # Pass input as kwarg for custom forward args
        print_results=False
    )
    print(f"FLOPs: {flops}")
    print(f"MACs:  {macs}")
    print(f"Params: {params}")