import torch
import torch.nn as nn

class TSformerVO(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, embed_dim, depth,
                 num_heads, dim_head, attn_dropout, ff_dropout, num_frames,
                 attention_type, time_only, input_channels=3):
        super(TSformerVO, self).__init__()

        # Patch embedding layer: convert image to patch tokens
        self.patch_embed = nn.Conv2d(
            in_channels=input_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dropout=attn_dropout,
                dim_feedforward=embed_dim * 4,
                batch_first=False 
            ),
            num_layers=depth
        )

        # Final MLP to predict pose (6D)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        Input: x of shape [B, 3, H, W]
        Output: Pose vector [B, 6]
        """
        B, C, H, W = x.shape

        # Convert to patch tokens
        x = self.patch_embed(x)  # [B, embed_dim, H', W']
        x = x.flatten(2).permute(2, 0, 1)  # [S, B, D]

        # Transformer
        x = self.transformer(x)  # [S, B, D]
        x = x.mean(dim=0)        # [B, D]

        # Predict pose
        out = self.fc(x)         # [B, 6]

        return out


# Optional utility to load model with torch-directml
def load_model_with_weights(model_args, checkpoint_path, device):
    model = TSformerVO(**model_args).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    return model