import torch
import cv2

MODEL_PATH = "crowdhuman_yolov5m.pt"
DEVICE = "cpu"

# 1) Load checkpoint the safe way
ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model = ckpt['model'].float().to(DEVICE)
model.eval()

# 2) Prepare an image for inference
def get_head_points(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to network size (640) & normalize
    im = cv2.resize(img_rgb, (640, 640))
    im = im.transpose(2, 0, 1) / 255.0
    im = torch.tensor(im, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # 3) Forward pass
    with torch.no_grad():
        pred = model(im)[0]

    # 4) Convert YOLO output (x1,y1,x2,y2,conf,class) → head centers
    pred = pred.cpu()
    points = []
    for x1, y1, x2, y2, conf, cls in pred:
        if conf < 0.25:     # confidence filter
            continue
        cx = int((x1 + x2) / 2 / 640 * img.shape[1])
        cy = int((y1 + y2) / 2 / 640 * img.shape[0])
        points.append((cx, cy))

    return points
