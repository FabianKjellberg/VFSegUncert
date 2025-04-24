import torch
import torch_directml  
from model import TSformerVO

# ✅ Set up device (CPU, CUDA, or AMD DirectML)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch_directml.device()

# ✅ Define dummy input (B=1, C=3, H=192, W=640)
dummy_input = torch.randn(1, 3, 192, 640).to(device)

# ✅ Define model parameters (match your args.pkl or checkpoint)
model_args = {
    "image_size": (192, 640),
    "patch_size": 16,
    "num_classes": 6,
    "embed_dim": 384,
    "depth": 12,
    "num_heads": 6,
    "dim_head": 64,
    "attn_dropout": 0.1,
    "ff_dropout": 0.1,
    "num_frames": 2,
    "attention_type": "divided_space_time",
    "time_only": False
}

# ✅ Initialize the model
model = TSformerVO(**model_args).to(device)
model.eval()

# ✅ Run inference
with torch.no_grad():
    output = model(dummy_input)
    print("🔍 Output shape:", output.shape)
    print("🔢 Pose prediction:", output.cpu().numpy())