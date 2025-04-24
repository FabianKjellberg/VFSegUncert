import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch_directml
from pp_liteseg import PPLiteSeg


class KITTIDMLSegmenter:
    def __init__(self, base_input_dir, base_output_dir, weights_path, sequence_list, base_size=512):
        self.base_input_dir = base_input_dir
        self.base_output_dir = base_output_dir
        self.sequence_list = sequence_list
        self.weights_path = weights_path
        self.base_size = base_size
        self.wh = 2
        self.device = torch_directml.device()

        # Load model once
        self.model = PPLiteSeg()
        ckpt = torch.load(weights_path, map_location="cpu")
        self.model.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
        self.model.eval().to(self.device)

        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

    def process_image(self, image_path, rel_out_path):
        out_dir_npy = os.path.join(self.base_output_dir, rel_out_path, "segmentclasses")
        out_dir_img = os.path.join(self.base_output_dir, rel_out_path, "image")
        os.makedirs(out_dir_npy, exist_ok=True)
        os.makedirs(out_dir_img, exist_ok=True)

        name = os.path.splitext(os.path.basename(image_path))[0]
        out_npy_path = os.path.join(out_dir_npy, name + ".npy")
        out_png_path = os.path.join(out_dir_img, name + ".png")

        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"❌ Couldn't read image: {image_path}")
                return

            img_resized = cv2.resize(img, (self.wh * self.base_size, self.base_size))
            image = img_resized.astype(np.float32)[:, :, ::-1] / 255.0
            image -= self.mean
            image /= self.std
            image = image.transpose((2, 0, 1))
            image_tensor = torch.from_numpy(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                out = self.model(image_tensor)
                out = out[0].squeeze(0)
                out = F.softmax(out, dim=0)
                pred = torch.argmax(out, dim=0).cpu().numpy().astype(np.int32)

            np.save(out_npy_path, pred)

            # Colored overlay
            colors = np.random.randint(0, 255, 19 * 3).reshape(19, 3)
            seg_img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
            for c in np.unique(pred):
                seg_img[pred == c] = colors[c]
            seg_img = cv2.resize(seg_img, (img.shape[1], img.shape[0]))
            vis = cv2.addWeighted(img, 0.7, seg_img, 0.3, 0)
            cv2.imwrite(out_png_path, vis)

        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")

    def run(self):
        for sequence in self.sequence_list:
            print(f"🚗 Processing sequence {sequence}...")

            input_dir = os.path.join(self.base_input_dir, sequence, "image_2")
            rel_output_dir = os.path.join(sequence, "image_2")

            image_path_list = []
            for file in sorted(os.listdir(input_dir)):
                if file.endswith(".png"):
                    full_path = os.path.join(input_dir, file)
                    image_path_list.append((full_path, rel_output_dir))

            print(f"📸 Found {len(image_path_list)} images in sequence {sequence}. Starting segmentation...")

            for image_path, rel_out_path in tqdm(image_path_list):
                self.process_image(image_path, rel_out_path)

        print("✅ All sequences finished!")


if __name__ == "__main__":
    SEQUENCES = ["02"]  # Set your sequence list here

    INPUT_BASE = "../datasets/kitti/sequences_png"
    OUTPUT_BASE = "../datasets/kitti/sequences_segmentation"
    WEIGHTS = "checkpoints/ppliteset_pp2torch_cityscape_pretrained.pth"

    segmenter = KITTIDMLSegmenter(INPUT_BASE, OUTPUT_BASE, WEIGHTS, SEQUENCES)
    segmenter.run()