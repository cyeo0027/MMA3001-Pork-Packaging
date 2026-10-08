"""Software verification with tiny, known-input ZIP fixtures.

These tests check data preparation and CLI behaviour. They do not measure YOLO
accuracy, image-label semantics, GPU timing or generalisation.
"""
import contextlib
import io
import json
import tempfile
import unittest
import warnings
from pathlib import Path
from zipfile import ZipFile
import yaml
from pork_dataset import prepare_unsealed_dataset
from main import main, predict_image, show_results

# A valid 1x1 PNG, used only to verify that archive bytes are preserved.
import base64
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/x8AAusB9Wl6F9sAAAAASUVORK5CYII="
)
NAMES = ["loose-meat", "packaging-error", "twisted-meat", "unsealed", "wrinkle"]


class PreparationTests(unittest.TestCase):
    """Known-input checks for annotation filtering and validation."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def entries(self):
        data = {"nc": 5, "names": NAMES}
        rows = [("data.yaml", yaml.safe_dump(data).encode())]
        for split in ("train", "valid", "test"):
            rows += [
                (f"{split}/images/positive.png", PNG),
                (f"{split}/labels/positive.txt",
                 b"1 0.5 0.5 0.2 0.2\n3 0.5 0.5 0.4 0.4\n3 0.25 0.25 0.1 0.1\n"),
                (f"{split}/images/negative.png", PNG),
                (f"{split}/labels/negative.txt", b"0 0.5 0.5 0.2 0.2\n"),
            ]
        return rows

    def write(self, entries):
        path = self.root / "fixture.zip"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with ZipFile(path, "w") as archive:
                for name, content in entries:
                    archive.writestr(name, content)
        return path

    def replace(self, entries, name, content):
        return [(n, content if n == name else value) for n, value in entries]

    def test_filter_remap_and_keep_negative_images(self):
        archive = self.write(self.entries())
        yaml_path, counts = prepare_unsealed_dataset(archive, self.root / "prepared")
        config = yaml.safe_load(yaml_path.read_text())
        self.assertEqual(config["nc"], 1)
        self.assertEqual(config["names"], {0: "unsealed"})
        self.assertEqual(config["val"], "valid/images")
        for row in counts:
            self.assertEqual(row, {
                "split": row["split"], "images": 2, "images_with_unsealed": 1,
                "images_without_unsealed": 1, "unsealed_boxes": 2,
            })
            base = yaml_path.parent / row["split"]
            self.assertEqual((base / "images/positive.png").read_bytes(), PNG)
            self.assertEqual((base / "images/negative.png").read_bytes(), PNG)
            self.assertEqual((base / "labels/negative.txt").read_text(), "")
            self.assertEqual((base / "labels/positive.txt").read_text(),
                             "0 0.5 0.5 0.4 0.4\n0 0.25 0.25 0.1 0.1\n")

    def test_invalid_records_are_rejected(self):
        invalid = [
            "3 0.5 0.5 0.2", "3 nan 0.5 0.2 0.2", "3 inf 0.5 0.2 0.2",
            "3 0.5 0.5 0 0.2", "3 1.2 0.5 0.2 0.2",
            "3 0.95 0.5 0.2 0.2", "5 0.5 0.5 0.2 0.2",
            "-1 0.5 0.5 0.2 0.2", "2.5 0.5 0.5 0.2 0.2",
            "nonsense",
        ]
        for number, line in enumerate(invalid):
            with self.subTest(line=line):
                entries = self.replace(self.entries(), "train/labels/positive.txt",
                                       line.encode())
                with self.assertRaises(ValueError):
                    prepare_unsealed_dataset(self.write(entries), self.root / f"bad{number}")

    def test_invalid_non_target_record_is_also_rejected(self):
        entries = self.replace(self.entries(), "train/labels/negative.txt",
                               b"0 nan 0.5 0.2 0.2")
        with self.assertRaises(ValueError):
            prepare_unsealed_dataset(self.write(entries), self.root / "prepared")

    def test_missing_label_is_rejected(self):
        entries = [(n, v) for n, v in self.entries() if n != "train/labels/positive.txt"]
        with self.assertRaises(FileNotFoundError):
            prepare_unsealed_dataset(self.write(entries), self.root / "prepared")

    def test_missing_split_is_rejected(self):
        entries = [(n, v) for n, v in self.entries() if not n.startswith("test/")]
        with self.assertRaises(FileNotFoundError):
            prepare_unsealed_dataset(self.write(entries), self.root / "prepared")

    def test_identical_duplicate_yaml_is_accepted(self):
        entries = self.entries()
        entries += [entries[0], entries[0]]
        _, rows = prepare_unsealed_dataset(self.write(entries), self.root / "prepared")
        self.assertEqual(sum(r["images"] for r in rows), 6)

    def test_conflicting_duplicate_yaml_is_rejected(self):
        entries = self.entries() + [("data.yaml", b"nc: 0\nnames: []\n")]
        with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
            prepare_unsealed_dataset(self.write(entries), self.root / "prepared")

    def test_unsafe_archive_paths_are_rejected(self):
        for number, name in enumerate(("../escape.txt", "/absolute.txt", "train\\bad.png")):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "Unexpected archive path"):
                    prepare_unsealed_dataset(self.write(self.entries() + [(name, b"x")]),
                                             self.root / f"unsafe{number}")

    def test_ambiguous_image_stems_are_rejected(self):
        entries = self.entries() + [("train/images/positive.jpg", PNG)]
        with self.assertRaisesRegex(ValueError, "Ambiguous image stem"):
            prepare_unsealed_dataset(self.write(entries), self.root / "prepared")

    def test_class_count_mismatch_is_rejected(self):
        config = yaml.safe_dump({"nc": 4, "names": NAMES}).encode()
        entries = self.replace(self.entries(), "data.yaml", config)
        with self.assertRaisesRegex(ValueError, "Class count"):
            prepare_unsealed_dataset(self.write(entries), self.root / "prepared")

    def test_string_keyed_class_mapping_is_supported(self):
        # The same export can represent IDs as strings rather than integers.
        config = yaml.safe_dump({"nc": 5, "names": {str(i): n for i, n in enumerate(NAMES)}}).encode()
        _, rows = prepare_unsealed_dataset(
            self.write(self.replace(self.entries(), "data.yaml", config)),
            self.root / "prepared")
        self.assertEqual([r["unsealed_boxes"] for r in rows], [2, 2, 2])


class EntryPointTests(unittest.TestCase):
    """Verify the read-only result command and input validation without a model."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_results_command_reads_existing_json_without_changing_it(self):
        record = {"model": "yolov8n", "image_size": 640, "recall": 0.75}
        path = self.root / "final_test_metrics.json"
        path.write_text(json.dumps(record))
        original = path.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["results", "--run-dir", str(self.root)]), 0)
        self.assertEqual(json.loads(output.getvalue()), record)
        self.assertEqual(path.read_bytes(), original)

    def test_missing_results_are_rejected(self):
        with self.assertRaises(FileNotFoundError):
            show_results(self.root)

    def test_non_object_result_is_rejected(self):
        (self.root / "final_test_metrics.json").write_text("[]")
        with self.assertRaises(ValueError):
            show_results(self.root)

    def test_prediction_requires_local_weights(self):
        with self.assertRaises(FileNotFoundError):
            predict_image(self.root / "missing.pt", self.root / "image.png", self.root / "out")

    def test_prediction_requires_existing_image(self):
        weights = self.root / "placeholder.pt"
        weights.write_bytes(b"Only tests the input guard, not model loading.")
        with self.assertRaises(FileNotFoundError):
            predict_image(weights, self.root / "missing.png", self.root / "out")

    def test_invalid_cli_command_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main(["unsupported"])
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
