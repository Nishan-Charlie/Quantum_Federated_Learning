import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from einops import rearrange
from calflops import calculate_flops

# === Flexible Conv Layer ===
class ConvLayer(nn.Module):
    """
    A flexible 2D Convolutional Layer with optional Batch Normalization and Activation.
    """
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
        # print("ConvLayer-S") # Debug prints, consider removing in production
        # print("X:",x.shape) # Debug prints, consider removing in production

        block = self.block(x)
        # print("block",block.shape) # Debug prints, consider removing in production
        # print("ConvLayer-E") # Debug prints, consider removing in production
        return block


# === Residual Depthwise Bottleneck with Dilation ===
class DilatedBottleneck(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, stride: int = 1, dilation: int = 1):
        super().__init__()

        # Padding calculation for stride and dilation
        # For kernel_size=3, to maintain spatial dims with stride=1, padding=dilation.
        # To halve spatial dims with stride=2, padding=dilation (assuming kernel_size=3).
        if stride == 1:
            padding = dilation
        elif stride == 2:
            # This logic works for dilation 1 and 2 with kernel_size 3 to halve the spatial dimension.
            # A more general approach for kernel_size=3 to halve the spatial dim
            # would simply be padding = dilation.
            # The current 'else' branch for stride=2 and dilation > 2 might not always
            # yield (I-1)/2, which is generally desired for halving.
            if dilation == 1:
                padding = 1
            elif dilation == 2:
                padding = 2
            else:
                # For kernel_size=3, K_eff = 2*dilation + 1.
                # To get (I-1)/2 output with stride 2, we need P = (K_eff - 1) / 2 = (2*dilation)/2 = dilation.
                # The current (2 * dilation - 1) // 2 is only equal to dilation when dilation is odd.
                # E.g., if dilation=4, (8-1)//2 = 3, but we want 4.
                # If this module is only used with dilation=1 or 2 when stride=2, it's fine.
                # If other dilations are used with stride=2, this should be revisited.
                padding = (2 * dilation - 1) // 2
        else:
            raise ValueError(f"Unsupported stride: {stride}")

        # Enable residual only when in_ch == out_ch and no spatial change (stride == 1)
        self.use_res = (in_ch == out_ch and stride == 1)

        self.block = nn.Sequential(
            ConvLayer(in_ch, mid_ch, kernel_size=1),
            ConvLayer(mid_ch, mid_ch, kernel_size=3, stride=stride, padding=padding,
                      groups=mid_ch, dilation=dilation),
            ConvLayer(mid_ch, out_ch, kernel_size=1, use_act=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print("[DilatedBottleneck] Start") # Debug prints, consider removing in production
        # print("Input shape :", x.shape) # Debug prints, consider removing in production

        out = self.block(x)
        # print("Output shape:", out.shape) # Debug prints, consider removing in production

        if self.use_res and x.shape == out.shape:
            # print("✅ Residual connection applied") # Debug prints, consider removing in production
            out = out + x
        else:
            # print("⛔ Residual skipped due to shape mismatch") # Debug prints, consider removing in production
            pass # No need to print if it's expected behavior

        # print("[DilatedBottleneck] End") # Debug prints, consider removing in production
        return out


# === Local Representation Block ===
class LocalRepresentationBlock(nn.Module):
    """
    Transforms local CNN features into a format suitable for the Transformer block
    using depthwise and pointwise convolutions.
    """
    def __init__(self, Cin: int, TransformerDim: int):
        super().__init__()
        # Kernel size 3, padding 1 for depthwise maintains spatial dimensions.
        self.depthwise_conv = nn.Conv2d(Cin, Cin, kernel_size=3, padding=1, groups=Cin)
        self.pointwise_conv = nn.Conv2d(Cin, TransformerDim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print("\nLocalRepresentationBlock-S") # Debug prints, consider removing in production
        x = self.depthwise_conv(x)
        # print("x-depthwise",x.shape) # Debug prints, consider removing in production
        x = self.pointwise_conv(x)
        # print("x-pointwise",x.shape) # Debug prints, consider removing in production
        # print("LocalRepresentationBlock-E") # Debug prints, consider removing in production
        return x

# === FeedForward Module ===
class FeedForward(nn.Module):
    """
    A standard Feed-Forward Network (FFN) typically used in Transformer blocks.
    Includes Layer Normalization, SiLU activation, and Dropout.
    """
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

# --- LinearSelfAttention based on MobileViTv2 reference ---
class LinearSelfAttention(nn.Module):
    """
    Implements a linear complexity self-attention mechanism inspired by MobileViTv2.
    It processes unfolded patches as (Batch, Channels, Pixels_in_patch, Num_patches).
    """
    def __init__(self, embed_dim: int, attn_dropout: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        self.qkv_proj = ConvLayer(
            in_channels=embed_dim,
            out_channels=1 + (2 * embed_dim), # 1 for query, embed_dim for key, embed_dim for value
            bias=bias,
            kernel_size=1,
            use_norm=False,
            use_act=False,
        )

        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.out_proj = ConvLayer(
            in_channels=embed_dim,
            out_channels=embed_dim,
            bias=bias,
            kernel_size=1,
            use_norm=False,
            use_act=False,
        )
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print("\nLinearSelfAttention-S") # Debug prints, consider removing in production
        b, p_dim, n, d = x.shape
        # print("x",x.shape) # Debug prints, consider removing in production
        # Reshape for Conv2d: (b, channels, height, width) -> (b, d, p_dim, n)
        # Here, 'd' (embed_dim) becomes channels, 'p_dim' becomes height, 'n' becomes width.
        x_reshaped_for_conv = rearrange(x, 'b p_dim n d -> b d p_dim n')
        # print("x_reshaped_for_conv",x_reshaped_for_conv.shape) # Debug prints, consider removing in production

        qkv = self.qkv_proj(x_reshaped_for_conv)
        # print("qkv",qkv.shape) # Debug prints, consider removing in production

        query, key, value = torch.split(
            qkv, split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=1
        )
        # print("query",query.shape) # Debug prints, consider removing in production
        # print("key",key.shape) # Debug prints, consider removing in production
        # print("value",value.shape) # Debug prints, consider removing in production

        context_scores = F.softmax(query, dim=-1) # Softmax over 'n' (number of patches)
        # print("context_scores",context_scores.shape) # Debug prints, consider removing in production
        context_scores = self.attn_dropout(context_scores)
        # print("context_scores",context_scores.shape) # Debug prints, consider removing in production

        context_vector = key * context_scores # (B, D, P_dim, N) * (B, 1, P_dim, N) -> (B, D, P_dim, N)
        # print("context_vector",context_vector.shape) # Debug prints, consider removing in production
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True) # Sum over 'N' (number of patches)
        # print("context_vector",context_vector.shape) # Debug prints, consider removing in production
        # Result: (B, D, P_dim, 1)

        intermediate_output = F.relu(value) * context_vector.expand_as(value)
        # print("intermediate_output",intermediate_output.shape) # Debug prints, consider removing in production
        out = self.out_proj(intermediate_output)
        # print("out",out.shape) # Debug prints, consider removing in production

        # Reshape back to original patch format: (b, p_dim, n, d)
        out = rearrange(out, 'b d p_dim n -> b p_dim n d')
        # print("out-rearrrang",out.shape) # Debug prints, consider removing in production
        # print("LinearSelfAttention-E") # Debug prints, consider removing in production
        return out

# === Transformer Module ===
class Transformer(nn.Module):
    """
    A sequence of Transformer blocks, each containing a LinearSelfAttention layer
    and a FeedForward Network.
    """
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
            x = attn(x) + x # Residual connection for attention
            x = ff(x) + x   # Residual connection for FFN
        return x


# === Efficent Attention Block ===
class EfficentAttentionBlock(nn.Module):
    """
    Combines local CNN processing with global linear attention and coordinate attention
    for efficient feature learning. This is the core hybrid block.
    """
    def __init__(self, Cin: int, TransformerDim: int, Cout: int, depth: int = 2, patch_size: int = 2):
        super().__init__()
        self.patch_size = patch_size
        self.TransformerDim = TransformerDim
        self.Cin = Cin # Store Cin for channel matching
        self.Cout = Cout # Store Cout for the output of the fusion layer

        self.local = LocalRepresentationBlock(Cin, TransformerDim)

        mlp_dim = TransformerDim * 2
        dropout = 0.2
        self.transformer = Transformer(
            dim=TransformerDim,
            depth=depth,
            mlp_dim=mlp_dim,
            dropout=dropout
        )
        self.conv_proj = nn.Conv2d(TransformerDim, Cin, kernel_size=1) # Project Transformer output back to Cin
        # The fusion layer now takes concatenated features (Cin + Cin) and outputs Cout
        self.fusion = nn.Conv2d(Cin + Cin, Cout, kernel_size=3, padding=1) # Fuse original and projected features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print("\nAffixAttentionBlock-S") # Debug prints, consider removing in production
        # Store original dimensions for later cropping and residual connection
        B, C, H, W = x.shape
        res = x # This is the original input for the final residual connection and coord_att
        #print("res",res.shape) # Debug prints, consider removing in production

        ph = pw = self.patch_size

        # Calculate number of patches for the padded tensor
        # Assuming H, W are multiples of ph, pw due to previous downsampling
        h_patches = H // ph
        w_patches = W // pw

        # 1. Local feature processing on input x
        local_out = self.local(x) # Output: (B, TransformerDim, H, W)
        # print("local_out",local_out.shape) # Debug prints, consider removing in production

        # 2. Prepare residual connection for Transformer (from original x)
        # This tensor should also have TransformerDim channels and spatial dims matching local_out.
        # Current logic: pad with zeros if Cin < TransformerDim, slice if Cin > TransformerDim.
        # For a more robust residual, consider a ConvLayer(Cin, TransformerDim, 1) here.
        if C < self.TransformerDim:
            padding_channels = torch.zeros(B, self.TransformerDim - C, H, W, device=x.device, dtype=x.dtype)
            # print("padding_channels",padding_channels.shape) # Debug prints, consider removing in production
            res_n_padded = torch.cat([x, padding_channels], dim=1)
            # print("res_n_padded",res_n_padded.shape) # Debug prints, consider removing in production
        else: # C >= self.TransformerDim
            print("Need to follow these conditions: C2 < D1 and C3 < D2")
            raise ValueError("Invalid channel configuration: C must be smaller than TransformerDim")
        # 3. Unfold features into patches for Transformer
        # local_out and res_n_padded are (B, TransformerDim, H, W)
        local_patches = rearrange(local_out, 'b d (h ph) (w pw) -> b (ph pw) (h w) d',
                                 ph=ph, pw=pw, h=h_patches, w=w_patches)
        # print("local_patches",local_patches.shape) # Debug prints, consider removing in production

        res_patches = rearrange(res_n_padded, 'b d (h ph) (w pw) -> b (ph pw) (h w) d',
                               ph=ph, pw=pw, h=h_patches, w=w_patches)
        # print("res_patches",res_patches.shape) # Debug prints, consider removing in production

        # 4. Apply Transformer
        seq = local_patches + res_patches # Element-wise sum of local features and residual patches
        # print("seq",seq.shape) # Debug prints, consider removing in production

        seq_t = self.transformer(seq)
        # print("seq_t",seq_t.shape) # Debug prints, consider removing in production

        # 5. Fold patches back into feature map
        fm = rearrange(seq_t, 'b (ph pw) (h w) d -> b d (h ph) (w pw)',
                      ph=ph, pw=pw, h=h_patches, w=w_patches)
        # print("fm",fm.shape) # Debug prints, consider removing in production

        # 6. Project Transformer output back to original channel dimension (Cin)
        projected = self.conv_proj(fm) # Output: (B, Cin, H, W)
        #print("projected",projected.shape) # Debug prints, consider removing in production

        # 7. Concatenate original features (res) and projected features (projected)
        # and apply fusion convolution.
        fused_features = torch.cat([res, projected], dim=1) # Concatenate along channel dimension
        # print("fused_features", fused_features.shape) # Debug prints, consider removing in production

        out = self.fusion(fused_features) # Pass concatenated tensor to fusion layer
        #print("out",out.shape)

        # Final residual connection with the original 'res'
        # Requires Cout == Cin and spatial dimensions to match, which they do here.
        # The fusion output has Cout channels, while res has Cin channels.
        # A residual connection here would require Cin == Cout.
        # If Cin != Cout, a residual connection is not directly possible.
        # Assuming Cin == Cout for residual, otherwise remove the residual.
        # Based on the model definition, Cin and Cout are often the same in these blocks,
        # but it's good to be mindful of this.
        # return out + res # Residual connection if Cin == Cout
        return out # Returning the fused output

        # print("AffixAttentionBlock-E") # Debug prints, consider removing in production

# === Ultra-Lightweight Network ===
class UltraLightEfficentNet_L1(nn.Module):
    """
    An ultra-lightweight neural network architecture for image classification,
    combining convolutional blocks with a linear attention mechanism.
    """
    def __init__(self, num_classes: int, image_size: int, dims: list[int], channels: list[int]):
        super().__init__()
        # Input image size is 256x256

        # Stem: Initial layers for downsampling and feature extraction
        # 256x256 -> (ConvLayer stride 2) 128x128 -> (MaxPool stride 2) 64x64
        self.stem = nn.Sequential(
            ConvLayer(3, channels[0], kernel_size=3, stride=2, padding=1)
        )

        # Stage 1: Mostly convolutional layers
        # Input: 64x64
        expansion = 2
        self.stage1 = nn.Sequential(
            DilatedBottleneck(channels[0], expansion * channels[0], channels[0], stride=2), # 64x64 -> 64x64
            DilatedBottleneck(channels[0], expansion * channels[0], channels[1], stride=2), # 64x64 -> 32x32
            DilatedBottleneck(channels[1], expansion * channels[1], channels[1], stride=1)  # 32x32 -> 32x32
        )

                # Stage 2: Added an extra DilatedBottleneck block
        self.stage2 = nn.Sequential(
            DilatedBottleneck(channels[1], expansion * channels[1], channels[2], stride=2, dilation=2),
            DilatedBottleneck(channels[2], expansion * channels[2], channels[2], stride=1),  # Added
            # EfficentAttentionBlock(Cin=channels[2], TransformerDim=dims[0], Cout=channels[2], depth=2)
        )

        # Stage 3: Added an extra DilatedBottleneck block
        self.stage3 = nn.Sequential(
            DilatedBottleneck(channels[2], expansion * channels[2], channels[3], stride=1, dilation=4),
            DilatedBottleneck(channels[3], expansion * channels[3], channels[3], stride=1),
            DilatedBottleneck(channels[3], expansion * channels[3], channels[3], stride=1),  # Added
            # EfficentAttentionBlock(Cin=channels[3], TransformerDim=dims[1], Cout=channels[3], depth=3)
        )
        # Head: Global pooling and final feature projection before classification
        # Input: (B, channels[3], 16, 16)
        self.head = nn.Sequential(
            ConvLayer(channels[3], channels[4], kernel_size=1), # (B, channels[4], 16, 16)
            nn.AdaptiveAvgPool2d(1) # (B, channels[4], 1, 1)
        )

        # Classifier: Flatten and linear layer for final classification
        # Input: (B, channels[4]) after flatten
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(channels[4], num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #print("\nUltraLightBlockNet_L1-S") # Debug prints, consider removing in production
        #print("\nStem-S") # Debug prints, consider removing in production
        x = self.stem(x)
        #print("stem:",x.shape)
        # print("\nStem-E") # Debug prints, consider removing in production
        # print("\nStage1-S") # Debug prints, consider removing in production
        x = self.stage1(x)
        # print("\nStage1-E") # Debug prints, consider removing in production
        # print("\nStage2-S") # Debug prints, consider removing in production
        x = self.stage2(x)
        # print("\nStage2-E") # Debug prints, consider removing in production
        # print("\nStage3-S") # Debug prints, consider removing in production
        x = self.stage3(x)
        #print("stage3:",x.shape)
        # print("\nStage3-E") # Debug prints, consider removing in production
        # print("\nUltraLightBlockNet_L1-E") # Debug prints, consider removing in production
        # print("\nHead-S") # Debug prints, consider removing in production
        x = self.head(x)
        # print("\nHead-E") # Debug prints, consider removing in production
        return self.classifier(x)

if __name__ == '__main__':
    # Example usage and testing

    # Instantiate the model with specified parameters
    model = UltraLightEfficentNet_L1(
            num_classes=10,
            image_size=256,
            dims=[48, 64], # Transformer dimensions
            channels=[8, 16, 32, 48, 288] # Channel counts for different stages
        )

    print(model)

    # Use torchinfo.summary for a detailed model summary
    # input_size should match the expected input to the model (batch_size, channels, height, width)
    summary(model, input_size=(1, 3, 256, 256))

    # Calculate FLOPs and MACs using calflops
    input_shape = (1, 3, 256, 256)
    flops, macs, params = calculate_flops(model=model,
                                          input_shape=input_shape,
                                          output_as_string=True,
                                          output_precision=4)
    print("LightBlockNet FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))

    # Example of a forward pass
    # dummy_input = torch.randn(1, 3, 256, 256).to(device)
    # output = model(dummy_input)
    # print("Output shape:", output.shape)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    param_mb = param_bytes / (1024 ** 2)
    print(f"📦 UltraLightEfficentNet_L1 size in parameters: {param_mb:.2f} MB  "
          f"({param_bytes:,} bytes)")
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    buffer_mb = buffer_bytes / (1024 ** 2)
    print(f"📦 UltraLightEfficentNet_L1 size in buffer: {buffer_mb:.2f} MB  "
          f"({param_bytes:,} bytes)")
    in_memory_bytes = param_bytes + buffer_bytes
    in_memory_mb = in_memory_bytes / (1024 ** 2)
    print(f"📦 UltraLightEfficentNet_L1 size in memory: {in_memory_mb:.2f} MB  "
          f"({param_bytes + buffer_bytes:,} bytes)")
    # 3. Print total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🔢 Total parameters: {total_params:,}")