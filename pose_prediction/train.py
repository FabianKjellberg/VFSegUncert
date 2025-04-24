import os
import sys 


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch_directml
from tqdm import tqdm
from model import TSformerVO
from utils.logger import Logger
from config import CONFIG  
import torch.nn.functional as F
import pickle
import cv2



# ===============================
# Dataset Loader
# ===============================
class KITTISegPoseDataset(Dataset):
    def __init__(self, sequences, image_root, segmentation_root, pose_root, img_size):
        self.data = []
        self.img_size = img_size

        for seq in sequences:
            img_dir = os.path.join(image_root, seq, "image_2")
            seg_dir = os.path.join(segmentation_root, seq, "image_2", "segmentclasses")
            pose_file = os.path.join(pose_root, f"{seq}.txt")

            if not os.path.exists(pose_file):
                continue

            with open(pose_file, 'r') as f:
                lines = f.readlines()
                for i in range(len(lines) - 1):
                    img1 = os.path.join(img_dir, f"{i:06d}.png")
                    img2 = os.path.join(img_dir, f"{i+1:06d}.png")
                    seg1 = os.path.join(seg_dir, f"{i:06d}.npy")
                    seg2 = os.path.join(seg_dir, f"{i+1:06d}.npy")

                    if os.path.exists(img1) and os.path.exists(img2) and os.path.exists(seg1) and os.path.exists(seg2):
                        pose_line = lines[i].strip().split()
                        pose = np.array(pose_line, dtype=np.float32).reshape(3, 4)
                        self.data.append((img1, img2, seg1, seg2, pose))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img1_path, img2_path, seg1_path, seg2_path, pose = self.data[idx]

        def load_and_process(img_path, seg_path):
            img = cv2.imread(img_path)
            img = cv2.resize(img, CONFIG["image_size"])
            seg = np.load(seg_path)
            seg = cv2.resize(seg, CONFIG["image_size"], interpolation=cv2.INTER_NEAREST)

            img = img.astype(np.float32)[:, :, ::-1] / 255.0
            seg = seg.astype(np.float32) / 18.0

            img = torch.from_numpy(img.transpose(2, 0, 1))
            seg = torch.from_numpy(seg).unsqueeze(0)

            return torch.cat((img, seg), dim=0)

        input1 = load_and_process(img1_path, seg1_path)
        input2 = load_and_process(img2_path, seg2_path)
        input_pair = torch.stack([input1, input2], dim=0)

        pose_vector = torch.tensor(self.pose_to_vector(pose), dtype=torch.float32)
        return input_pair, pose_vector

    @staticmethod
    def pose_to_vector(pose):
        return pose.flatten()


# ===============================
# Training
# ===============================
def train():
    #Use DirectML if enabled
    device = torch_directml.device() if CONFIG["use_directml"] else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(" Using device:", device)
    
    logger = Logger(log_dir=os.path.join(CONFIG["checkpoint_dir"], "logs"))

    dataset = KITTISegPoseDataset(
        sequences=CONFIG["sequences"],
        image_root=CONFIG["image_root"],
        segmentation_root=CONFIG["segment_root"],
        pose_root=CONFIG["pose_root"],
        img_size=CONFIG["image_size"]
)

    print(f"📊 Dataset size: {len(dataset)} samples")

    loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])

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
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    loss_fn = nn.MSELoss()

    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0

        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}"):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            inputs = inputs.mean(dim=1)

            preds = model(inputs.float())
            loss = loss_fn(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        logger.log_scalar("Loss/train", total_loss / len(loader), step=epoch)
        logger.next_step()

        if (epoch + 1) % CONFIG["save_interval"] == 0:
            ckpt_path = os.path.join(CONFIG["checkpoint_dir"], f"epoch_{epoch+1}.pth")
            os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")
    logger.close()


if __name__ == "__main__":
    train()