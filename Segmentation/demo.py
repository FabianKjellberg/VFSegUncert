import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F
import time
from pp_liteseg import PPLiteSeg


def parse_args():
    parser = argparse.ArgumentParser(description='Run PPLiteSeg on CPU')

    parser.add_argument('--image',
                        help='Test image path',
                        default="mainz_000001_009328_leftImg8bit.png",
                        type=str)
    parser.add_argument('--weights',
                        help='Path to cityscape-pretrained weights',
                        default="ppliteset_pp2torch_cityscape_pretrained.pth",
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    return parser.parse_args()


def colorEncode(labelmap, colors, mode='RGB'):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3), dtype=np.uint8)
    for label in np.unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb = labelmap_rgb + (labelmap == label)[:, :, np.newaxis] * \
                       np.tile(colors[label], (labelmap.shape[0], labelmap.shape[1], 1))
    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb


def main():
    base_size = 512
    wh = 2
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    args = parse_args()

    # Load model
    model = PPLiteSeg()
    print("🔍 Model initialized")
    model.eval()

    # Load weights on CPU
    ckpt = torch.load(args.weights, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)

    # Read image
    img = cv2.imread(args.image)
    imgor = img.copy()
    img = cv2.resize(img, (wh * base_size, base_size))
    image = img.astype(np.float32)[:, :, ::-1] / 255.0
    image -= mean
    image /= std

    # Prepare input tensor
    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image).unsqueeze(0)

    start = time.time()
    with torch.no_grad():
        out = model(image)
    end = time.time()
    print("✅ Inference time:", round(end - start, 2), "s")

    # Postprocess
    out = out[0].squeeze(dim=0)
    out_softmax = F.softmax(out, dim=0)
    out_argmax = torch.argmax(out_softmax, dim=0)
    pred = out_argmax.detach().cpu().numpy().astype(np.int32)

    # Color map
    colors = np.random.randint(0, 255, 19 * 3).reshape(19, 3)
    pred_color = colorEncode(pred, colors).astype(np.uint8)
    pred_color = cv2.resize(pred_color, (imgor.shape[1], imgor.shape[0]))

    # Blend with original image
    im_vis = cv2.addWeighted(imgor, 0.7, pred_color, 0.3, 0)
    cv2.imwrite("results.jpg", im_vis)
    print("✅ Segmentation result saved to results.jpg")

    # === 🎨 Save each class separately ===
    output_dir = "results/classes"
    os.makedirs(output_dir, exist_ok=True)

    cityscapes_classes = [
        "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
        "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
        "truck", "bus", "train", "motorcycle", "bicycle"
    ]

    for class_id in np.unique(pred):
        if class_id < 0 or class_id >= len(cityscapes_classes):
            continue

        class_mask = (pred == class_id).astype(np.uint8) * 255
        class_mask_color = cv2.merge([class_mask] * 3)

        class_mask_color = cv2.resize(class_mask_color, (imgor.shape[1], imgor.shape[0]))
        highlight = cv2.addWeighted(imgor, 0.3, class_mask_color, 0.7, 0)
        label = f"{class_id:03d}_{cityscapes_classes[class_id]}"
        cv2.putText(highlight, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        output_path = os.path.join(output_dir, f"{label}.png")
        cv2.imwrite(output_path, highlight)

    print(f"✅ Saved {len(np.unique(pred))} class masks in {output_dir}")


if __name__ == '__main__':
    main()
