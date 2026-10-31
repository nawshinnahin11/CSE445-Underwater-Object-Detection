import subprocess
import sys

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "ultralytics==8.3.221", "imageio-ffmpeg==0.6.0"
])

import os
import json
import math
import shutil
import tempfile
from pathlib import Path
from collections import defaultdict

WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)
os.environ["YOLO_CONFIG_DIR"] = str(WORK / "ultralytics_config")
os.environ["WANDB_MODE"] = "disabled"

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
import ultralytics
import imageio_ffmpeg
from PIL import Image
from IPython.display import display, Video, FileLink
from ultralytics import YOLO, settings

settings.update({key: False for key in [
    "wandb", "mlflow", "comet", "clearml", "neptune", "raytune",
    "dvc", "tensorboard", "hub", "sync"
]})

DATA_ROOT = Path("/kaggle/input/datasets/salmanabir11/cocodatadet")
VIDEO_PATH = "/kaggle/input/datasets/salmanabir11/testtvedio/test.mp4"
CLASS_NAMES = ["fish", "jellyfish", "penguin", "puffin", "shark", "starfish", "stingray"]

assert DATA_ROOT.is_dir(), f"Dataset folder not found: {DATA_ROOT}"
assert torch.cuda.is_available(), "Enable the Kaggle GPU T4 accelerator."
assert "T4" in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)

if VIDEO_PATH:
    video_path = Path(VIDEO_PATH)
else:
    video_candidates = sorted(DATA_ROOT.parent.rglob("test.mp4"))
    if not video_candidates:
        video_candidates = sorted(Path("/kaggle/input").rglob("test.mp4"))
    assert len(video_candidates) == 1, (
        "Attach the Classroom test.mp4 and set VIDEO_PATH to its exact path. "
        f"Matches: {[str(p) for p in video_candidates]}"
    )
    video_path = video_candidates[0]

assert video_path.is_file(), f"Video not found: {video_path}"
capture = cv2.VideoCapture(str(video_path))
try:
    video_fps = capture.get(cv2.CAP_PROP_FPS)
    video_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    readable, first_frame = capture.read()
    assert readable, f"Cannot read video: {video_path}"
    assert math.isfinite(video_fps) and video_fps > 0, f"Invalid FPS: {video_fps}"
finally:
    capture.release()

print({"ultralytics": ultralytics.__version__, "torch": torch.__version__,
       "gpu": torch.cuda.get_device_name(0), "video": str(video_path),
       "fps": video_fps, "frames": video_frame_count})
