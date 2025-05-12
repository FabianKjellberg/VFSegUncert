# config.py

CONFIG = {
    # Dataset
    "sequences": ["00", "01", "02", "03", "04", "05"],
    "image_dir": "../datasets/kitti/sequences_png/{}/image_2",
    "segment_dir": "../datasets/kitti/sequences_segmentation/{}/image_2/segmentclasses",
    "pose_dir": "../datasets/kitti/poses/{}.txt",
    "data_root": "../datasets/kitti",

    # Model Architecture
    "image_size": (192, 640),
    "patch_size": 16,
    "num_classes": 15,
    "embed_dim": 384,
    "depth": 12,
    "num_heads": 6,
    "dim_head": 64,
    "attn_dropout": 0.1,
    "ff_dropout": 0.1,
    "num_frames": 5,
    "attention_type": "divided_space_time",
    "time_only": False,
    "input_channels": 15,

    # Training
    "batch_size": 2,
    "epochs": 10,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "checkpoint_dir": "pose_prediction/checkpoints/Model1",
    "save_interval": 1,
    "log_interval": 10,
    "num_workers": 4,
    "image_root": "datasets/kitti/sequences_png",
    "segment_root": "datasets/kitti/sequences_segmentation",
    "pose_root": "datasets/kitti/poses",

    # Hardware
    "use_directml": True
}