import os
import glob
import cv2
import torch
import numpy as np
from scipy.ndimage import gaussian_filter

# --------------------------
# Model load (native YOLOv5)
# --------------------------
from models.yolo import Model
from utils.torch_utils import select_device
from utils.general import non_max_suppression, scale_coords

WEIGHTS = "crowdhuman_yolov5m.pt"
DEVICE = select_device("cpu")

print("Loading YOLOv5 model...")
checkpoint = torch.load(WEIGHTS, map_location=DEVICE, weights_only=False)
model = checkpoint['model'].to(DEVICE).float().eval()

# --------------------------
# Dataset
# --------------------------
root = "real_world"
all_images = glob.glob(os.path.join(root, "**", "images", "*.jpg"), recursive=True)

csv = open(os.path.join(root, "counts.csv"), "w")
csv.write("image,count\n")

# --------------------------
# Inference loop
# --------------------------
for img_path in all_images:
    img0 = cv2.imread(img_path)
    H, W = img0.shape[:2]

    img = cv2.resize(img0, (640, 640))
    img = img[:, :, ::-1].transpose(2, 0, 1)   # BGR → RGB → CHW
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    img = torch.from_numpy(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(img)[0]
    pred = non_max_suppression(pred, 0.25, 0.45)[0]

    points = []
    if pred is not None:
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img0.shape).round()
        for x1, y1, x2, y2, conf, cls in pred:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            points.append((cx, cy))

    # Density map
    density = np.zeros((H, W), dtype=np.float32)
    for (x, y) in points:
        if 0 <= x < W and 0 <= y < H:
            density[y, x] += 1
    density = gaussian_filter(density, sigma=4)

    # Save
    density_dir = os.path.dirname(img_path).replace("images", "density")
    os.makedirs(density_dir, exist_ok=True)

    np.save(os.path.join(
        density_dir,
        os.path.basename(img_path).replace(".jpg", ".npy")
    ), density)

    csv.write(f"{img_path},{len(points)}\n")
    print(f"{img_path} -> {len(points)}")

csv.close()
print("\n DONE: all density maps and counts generated")
