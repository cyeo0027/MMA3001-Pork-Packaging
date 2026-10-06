"""Command-line entry point for data preparation, saved results and prediction.

Training and model-selection experiments are recorded in
notebooks/MMA3001_training.ipynb. This entry point performs a small CPU-friendly
demonstration or reads completed results. Importing it has no execution side
effects and does not import Ultralytics until prediction is requested.
"""
import argparse
import json
from pathlib import Path
from pork_dataset import prepare_unsealed_dataset


def show_results(run_dir):
    """Read and print the existing final-test JSON without running inference.

    Parameters
    ----------
    run_dir : pathlib.Path
        Experiment folder containing final_test_metrics.json.

    Returns
    -------
    dict
        The recorded final-test metadata and metrics.

    Raises
    ------
    FileNotFoundError
        The saved metrics file is absent.
    ValueError
        JSON decoding fails or the saved JSON is not an object.
    """
    path = Path(run_dir) / "final_test_metrics.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("Expected a JSON object in final_test_metrics.json.")
    print(json.dumps(record, indent=2))
    return record


def predict_image(weights, image, output, device="cpu"):
    """Predict annotated regions in one image using a saved checkpoint.

    Parameters
    ----------
    weights : pathlib.Path
        Local trained YOLOv8n best.pt checkpoint.
    image : pathlib.Path
        Supported input image; Ultralytics validates its contents.
    output : pathlib.Path
        Directory where prediction images are written.
    device : str
        CPU by default, or a CUDA device such as '0'.

    Returns
    -------
    list
        Ultralytics prediction result objects.

    Raises
    ------
    FileNotFoundError
        Checkpoint or input image is absent.
    ImportError
        Ultralytics is not installed.

    Notes
    -----
    The displayed confidence threshold is 0.25. It is a demonstration setting,
    separate from the metrics and settings of the completed final evaluation.
    """
    weights, image = Path(weights), Path(image)
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")
    if not image.is_file():
        raise FileNotFoundError(f"Image not found: {image}")
    from ultralytics import YOLO
    return YOLO(str(weights)).predict(
        source=str(image), imgsz=640, conf=0.25, device=device,
        save=True, project=str(output), name="demo", exist_ok=True,
    )


def main(argv=None):
    """Parse CLI arguments and run a requested operation; return zero on success."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Prepare the single-class dataset.")
    prep.add_argument("--zip", type=Path, required=True)
    prep.add_argument("--destination", type=Path, required=True)
    results = commands.add_parser("results", help="Display saved final-test metrics.")
    results.add_argument("--run-dir", type=Path, default=Path("results/20261005_053211"))
    pred = commands.add_parser("predict", help="Run a single-image demonstration.")
    pred.add_argument("--weights", type=Path,
        default=Path("results/20261005_053211/train/yolov8n/weights/best.pt"))
    pred.add_argument("--image", type=Path, required=True)
    pred.add_argument("--output", type=Path, default=Path("demo_predictions"))
    pred.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    if args.command == "prepare":
        if args.destination.exists() and any(args.destination.iterdir()):
            raise ValueError("Use a new empty destination to avoid stale dataset files.")
        yaml_path, rows = prepare_unsealed_dataset(args.zip, args.destination)
        print(json.dumps({"data_yaml": str(yaml_path), "counts": rows}, indent=2))
    elif args.command == "results":
        show_results(args.run_dir)
    else:
        predict_image(args.weights, args.image, args.output, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
