import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import TSformerVO
from dataset import KITTIPoseDataset
from config import CONFIG


def plot_trajectory(checkpoint_path, sequence="09"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Testing on device: {device}")

    model_args = {
        "image_size": CONFIG["image_size"],
        "patch_size": CONFIG["patch_size"],
        "num_classes": CONFIG["num_classes"],
        "embed_dim": CONFIG["embed_dim"],
        "depth": CONFIG["depth"],
        "num_heads": CONFIG["num_heads"],
        "dim_head": CONFIG["dim_head"],
        "attn_dropout": CONFIG["attn_dropout"],
        "ff_dropout": CONFIG["ff_dropout"],
        "num_frames": CONFIG["num_frames"],
        "attention_type": CONFIG["attention_type"],
        "time_only": CONFIG["time_only"],
        "input_channels": CONFIG["input_channels"]
    }

    model = TSformerVO(**model_args).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_dataset = KITTIPoseDataset([sequence], CONFIG, train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    pred_poses = []
    true_poses = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs)
            pred_poses.append(preds.cpu().numpy())
            true_poses.append(targets.cpu().numpy())

    pred_poses = np.concatenate(pred_poses, axis=0)
    true_poses = np.concatenate(true_poses, axis=0)

    pred_x = np.cumsum(pred_poses[:, 0])
    pred_z = np.cumsum(pred_poses[:, 2])
    true_x = np.cumsum(true_poses[:, 0])
    true_z = np.cumsum(true_poses[:, 2])

    plt.figure(figsize=(10, 6))
    plt.plot(true_x, true_z, label="Ground Truth", linewidth=2)
    plt.plot(pred_x, pred_z, label="Predicted", linestyle="--")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.legend()
    plt.title(f"Trajectory Comparison (Sequence {sequence})")
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(f"trajectory_plot_sequence_{sequence}.png")
    print(f"✅ Plot saved as trajectory_plot_sequence_{sequence}.png")


if __name__ == "__main__":
    checkpoint_path = CONFIG["resume_checkpoint"]
    plot_trajectory(checkpoint_path, sequence="09")