import torch
import torch.serialization
import numpy.core.multiarray

# allow unpickling old YOLOv5 weight files
torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])

from yolov5 import YOLOv5

MODEL_PATH = "crowdhuman_yolov5m.pt"
DEVICE = "cpu"

# bypass PyTorch 2.6 restriction
model = YOLOv5(MODEL_PATH, DEVICE, autoshape=True)
