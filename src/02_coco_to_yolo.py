def convert_split(split, source_root, destination_root, class_names):
    source = source_root / split
    candidates = sorted(source.glob("*.json"))
    coco_files = []
    for candidate in candidates:
        content = json.loads(candidate.read_text(encoding="utf-8-sig"))
        if isinstance(content, dict) and {"images", "annotations", "categories"} <= content.keys():
            coco_files.append((candidate, content))
    assert len(coco_files) == 1, f"Expected one COCO JSON in {source}; found {len(coco_files)}"
    annotation_file, coco = coco_files[0]

    class_indices = {name: index for index, name in enumerate(class_names)}
    categories = {item["id"]: item["name"].strip().lower() for item in coco["categories"]}
    assert len(categories) == len(coco["categories"]), "Duplicate category IDs"
    assert set(class_names) <= set(categories.values()), f"Missing classes in {annotation_file}"
    category_map = {key: class_indices[name] for key, name in categories.items() if name in class_indices}
    images = {item["id"]: item for item in coco["images"]}
    assert len(images) == len(coco["images"]), "Duplicate image IDs"
    annotations = defaultdict(list)
    for annotation in coco["annotations"]:
        assert annotation["image_id"] in images, f"Unknown image ID: {annotation['image_id']}"
        assert annotation["category_id"] in category_map, f"Unknown category: {annotation['category_id']}"
        annotations[annotation["image_id"]].append(annotation)

    images_dir = destination_root / "images" / split
    labels_dir = destination_root / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    class_counts = [0] * len(class_names)
    box_count = 0
    clipped_boxes = 0

    for index, (image_id, info) in enumerate(images.items()):
        filename = Path(info["file_name"].replace("\\", "/"))
        possible_paths = [source / filename, source / "images" / filename,
                          source / "images" / filename.name, source / filename.name]
        existing = list(dict.fromkeys(p.resolve() for p in possible_paths if p.is_file()))
        assert len(existing) == 1, f"Missing or ambiguous image: {info['file_name']}"
        image_path = existing[0]
        width, height = int(info["width"]), int(info["height"])
        assert width > 0 and height > 0, f"Invalid image dimensions: {image_path}"
        with Image.open(image_path) as image:
            assert image.size == (width, height), f"COCO/image size mismatch: {image_path}"
            image.verify()
        target_name = f"{index:06d}"
        shutil.copy2(image_path, images_dir / (target_name + image_path.suffix.lower()))
        lines = []

        for annotation in annotations[image_id]:
            x, y, box_width, box_height = map(float, annotation["bbox"])
            assert all(math.isfinite(v) for v in [x, y, box_width, box_height]), annotation
            assert box_width > 0 and box_height > 0, annotation
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(float(width), x + box_width), min(float(height), y + box_height)
            assert x2 > x1 and y2 > y1, f"Box outside image: {annotation}"
            clipped_boxes += int((x1, y1, x2, y2) != (x, y, x + box_width, y + box_height))
            normalized = [(x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                          (x2 - x1) / width, (y2 - y1) / height]
            assert all(0 <= value <= 1 for value in normalized), annotation
            class_id = category_map[annotation["category_id"]]
            lines.append(str(class_id) + " " + " ".join(f"{value:.10f}" for value in normalized))
            class_counts[class_id] += 1
            box_count += 1

        (labels_dir / (target_name + ".txt")).write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )

    assert box_count == len(coco["annotations"]), "Some boxes were not converted"
    assert len(list(images_dir.iterdir())) == len(images), "Image count mismatch"
    assert len(list(labels_dir.glob("*.txt"))) == len(images), "Label count mismatch"
    return {"split": split, "images": len(images), "boxes": box_count,
            "clipped_boxes": clipped_boxes}, class_counts

YOLO_ROOT = Path(tempfile.mkdtemp(prefix="aquarium_yolo_", dir=WORK))
conversion_rows = []
class_count_rows = []
for split in ["train", "valid", "test"]:
    row, counts = convert_split(split, DATA_ROOT, YOLO_ROOT, CLASS_NAMES)
    conversion_rows.append(row)
    class_count_rows.append({"split": split, **dict(zip(CLASS_NAMES, counts))})

conversion_df = pd.DataFrame(conversion_rows)
class_counts_df = pd.DataFrame(class_count_rows).set_index("split")
print(conversion_df.to_string(index=False))
print(class_counts_df.to_string())
conversion_df.to_csv(WORK / "conversion_counts.csv", index=False)
class_counts_df.to_csv(WORK / "class_counts.csv")

expected_counts = {"train": (448, 3324), "valid": (127, 909), "test": (63, 582)}
for row in conversion_rows:
    assert (row["images"], row["boxes"]) == expected_counts[row["split"]], (
        f"Counts differ from the lab handout: {row}. Check the uploaded Classroom dataset."
    )

DATA_YAML = WORK / "data.yaml"
DATA_YAML.write_text(yaml.safe_dump({
    "path": str(YOLO_ROOT), "train": "images/train", "val": "images/valid",
    "test": "images/test", "nc": len(CLASS_NAMES), "names": CLASS_NAMES
}, sort_keys=False), encoding="utf-8")
print(DATA_YAML.read_text())
