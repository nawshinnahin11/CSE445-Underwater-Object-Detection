MODEL_WEIGHTS = "yolo11s.pt"
train_args = {
    "data": str(DATA_YAML),
    "epochs": 20,
    "imgsz": 640,
    "batch": 16,
    "device": 0,
    "amp": True,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.1,
    "cos_lr": True,
    "warmup_epochs": 3.0,
    "warmup_bias_lr": 0.0,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "mosaic": 1.0,
    "close_mosaic": 5,
    "mixup": 0.0,
    "fliplr": 0.5,
    "flipud": 0.0,
    "scale": 0.3,
    "translate": 0.1,
    "degrees": 0.0,
    "hsv_h": 0.015,
    "hsv_s": 0.5,
    "hsv_v": 0.3,
    "patience": 0,
    "seed": 42,
    "deterministic": True,
    "workers": 2,
    "cache": False,
    "pretrained": True,
    "resume": False,
    "val": True,
    "save": True,
    "plots": True,
    "project": str(WORK / "runs"),
    "name": "aquarium_yolo11",
    "exist_ok": False
}
print(json.dumps({"model": MODEL_WEIGHTS, **train_args}, indent=2))
(WORK / "hyperparameters.json").write_text(
    json.dumps({"model": MODEL_WEIGHTS, **train_args}, indent=2), encoding="utf-8"
)

def verify_amp(trainer):
    assert trainer.amp, "AMP was disabled after its check; resolve this before submitting."
    print({"amp_enabled": bool(trainer.amp)})

model = YOLO(MODEL_WEIGHTS)
model.add_callback("on_train_start", verify_amp)
model.train(**train_args)
TRAIN_DIR = Path(model.trainer.save_dir)
BEST_WEIGHTS = TRAIN_DIR / "weights" / "best.pt"
assert BEST_WEIGHTS.is_file(), "Training did not create best.pt"
shutil.copy2(BEST_WEIGHTS, WORK / "best.pt")
print((TRAIN_DIR / "args.yaml").read_text())

history = pd.read_csv(TRAIN_DIR / "results.csv")
history.columns = history.columns.str.strip()

expected_epochs = int(train_args["epochs"])
assert history["epoch"].astype(int).tolist() == list(range(1, expected_epochs + 1)), "Training did not complete the configured epochs"

display(history)
display(Image.open(TRAIN_DIR / "results.png"))
