best_model = YOLO(str(WORK / "best.pt"))
metrics = best_model.val(
    data=str(DATA_YAML), split="test", imgsz=640, batch=16,
    device=0, conf=0.001, iou=0.7, max_det=300,
    plots=True, save_json=False, verbose=True,
    project=str(WORK / "runs"), name="test_evaluation", exist_ok=False
)
summary_df = pd.DataFrame([{
    "mAP@0.5": float(metrics.box.map50),
    "mAP@0.5:0.95": float(metrics.box.map),
    "precision": float(metrics.box.mp),
    "recall": float(metrics.box.mr)
}])
per_class_df = pd.DataFrame({
    "class_id": range(len(CLASS_NAMES)), "class": CLASS_NAMES,
    "test_boxes": [int(class_counts_df.loc["test", name]) for name in CLASS_NAMES],
    "precision": np.nan, "recall": np.nan, "AP@0.5": np.nan, "AP@0.5:0.95": np.nan
})
for metric_index, class_id in enumerate(metrics.box.ap_class_index):
    per_class_df.loc[int(class_id), ["precision", "recall", "AP@0.5", "AP@0.5:0.95"]] = (
        metrics.box.class_result(metric_index)
    )
display(summary_df)
display(per_class_df)
summary_df.to_csv(WORK / "test_metrics.csv", index=False)
per_class_df.to_csv(WORK / "test_per_class_metrics.csv", index=False)
TEST_DIR = Path(metrics.save_dir)
display(Image.open(TEST_DIR / "confusion_matrix.png"))
display(Image.open(TEST_DIR / "confusion_matrix_normalized.png"))

test_images = sorted((YOLO_ROOT / "images" / "test").iterdir())
assert len(test_images) >= 4, "At least four test images are required"
sample_indices = np.random.default_rng(42).choice(len(test_images), size=4, replace=False)
sample_paths = [str(test_images[int(index)]) for index in sample_indices]
predictions = best_model.predict(
    source=sample_paths, imgsz=640, conf=0.25, iou=0.7,
    device=0, save=False, verbose=False
)
PREDICTION_DIR = WORK / "test_predictions"
PREDICTION_DIR.mkdir(exist_ok=True)
for index, result in enumerate(predictions, 1):
    annotated_image = Image.fromarray(result.plot()[:, :, ::-1])
    annotated_image.save(PREDICTION_DIR / f"prediction_{index}.jpg")
    display(annotated_image)
