"""Prepare and validate the project's single-class YOLO dataset.

The preparation function below is copied unchanged from Cell 5 of the original
MMA3001 notebook. It filters annotations, preserves target-negative images and
writes a YOLO YAML. The companion notebook checks source parity with the
student's executed notebook before running the software tests.

Use a fresh destination for each archive. A failed preparation may leave partial
files; discard that temporary destination before retrying. Image pixels are
copied without decoding. Label verification does not establish physical seal
integrity, label correctness or independent dataset splits.
"""
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
import math
import yaml

__all__ = ["prepare_unsealed_dataset"]
__docformat__ = "numpy"

def prepare_unsealed_dataset(zip_path, destination):
    """Copy images and write validated single-class YOLO labels from an archive.

    Parameters
    ----------
    zip_path : pathlib.Path
        Original Roboflow YOLOv8 ZIP, read without modification.
    destination : pathlib.Path
        Directory for generated images, labels and YAML.

    Returns
    -------
    tuple
        Generated YAML path and per-split image/box counts.

    Raises
    ------
    ValueError
        Invalid labels, conflicting duplicate entries or unexpected class definitions.
    FileNotFoundError
        A split or matching image label is missing.
    """
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    with ZipFile(zip_path) as archive:
        members = {}
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = info.filename
            parts = PurePosixPath(name)
            if parts.is_absolute() or ".." in parts.parts or "\\" in name:
                raise ValueError(f"Unexpected archive path: {name}")
            if name in members and archive.read(members[name]) != archive.read(info):
                raise ValueError(f"Conflicting duplicate ZIP entry: {name}")
            members[name] = info

        original = yaml.safe_load(archive.read(members["data.yaml"]))
        names = original["names"]
        if isinstance(names, dict):
            names = [names[i] if i in names else names[str(i)] for i in range(len(names))]
        target_id = names.index("unsealed")
        if original["nc"] != len(names):
            raise ValueError("Class count and class names disagree.")

        for split in ("train", "valid", "test"):
            image_dir = destination / split / "images"
            label_dir = destination / split / "labels"
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            image_names = sorted(name for name in members
                                 if name.startswith(f"{split}/images/")
                                 and PurePosixPath(name).suffix.lower() in {".jpg", ".jpeg", ".png"})
            if not image_names:
                raise FileNotFoundError(f"No images found in {split}.")
            target_images = target_boxes = 0
            seen_stems = set()
            for name in image_names:
                image_name = PurePosixPath(name).name
                stem = PurePosixPath(name).stem
                if stem in seen_stems:
                    raise ValueError(f"Ambiguous image stem in {split}: {stem}")
                seen_stems.add(stem)
                label_name = f"{split}/labels/{stem}.txt"
                if label_name not in members:
                    raise FileNotFoundError(f"Missing label: {label_name}")
                selected = []
                label_text = archive.read(members[label_name]).decode("utf-8")
                for number, line in enumerate(label_text.splitlines(), start=1):
                    if not line.strip():
                        continue
                    values = list(map(float, line.split()))
                    if len(values) != 5 or not all(math.isfinite(v) for v in values):
                        raise ValueError(f"Invalid record: {label_name}:{number}")
                    cls, x, y, w, h = values
                    if int(cls) != cls or not 0 <= cls < len(names):
                        raise ValueError(f"Invalid class: {label_name}:{number}")
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                        raise ValueError(f"Invalid box: {label_name}:{number}")
                    if min(x - w/2, y - h/2) < -1e-5 or max(x + w/2, y + h/2) > 1 + 1e-5:
                        raise ValueError(f"Box outside image: {label_name}:{number}")
                    if int(cls) == target_id:
                        selected.append("0 " + " ".join(line.split()[1:]))
                (image_dir / image_name).write_bytes(archive.read(members[name]))
                (label_dir / f"{stem}.txt").write_text("\n".join(selected) + ("\n" if selected else ""))
                target_images += bool(selected)
                target_boxes += len(selected)
            rows.append({"split": split, "images": len(image_names),
                         "images_with_unsealed": target_images,
                         "images_without_unsealed": len(image_names) - target_images,
                         "unsealed_boxes": target_boxes})

    config = {"path": str(destination), "train": "train/images", "val": "valid/images",
              "test": "test/images", "nc": 1, "names": {0: "unsealed"}}
    yaml_path = destination / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return yaml_path, rows
