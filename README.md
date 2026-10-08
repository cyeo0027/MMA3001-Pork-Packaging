# Detecting annotated unsealed pork packaging using YOLOv8

**MMA3001 Numerical Methods and Machine Learning — Yeo Chun Hau**

This project localises regions labelled **unsealed** in pork-packaging images.
It compares YOLOv8n and YOLOv8s, evaluates inference-resolution sensitivity and
tests the selected YOLOv8n model at an input size of 640 pixels. The selected model achieved **79.98%
mAP50 and 75.51% recall** on the supplied test split.

This is a prototype for reviewing annotated visual regions. It does not certify
airtightness, meat freshness or food safety. Images without an unsealed annotation
can contain other packaging defects.

## Review the project

| Item | Location |
| --- | --- |
| Executed training notebook | [notebooks/MMA3001_training.ipynb](notebooks/MMA3001_training.ipynb) |
| Functional command-line entry point | [main.py](main.py) |
| Dataset preparation | [pork_dataset.py](pork_dataset.py) |
| Known-input software tests | [tests/test_project.py](tests/test_project.py) |
| Stored passing test report | [verification/software_test_report.txt](verification/software_test_report.txt) |
| Preparation-function source parity | [verification/source_parity.json](verification/source_parity.json) |
| Actual-archive audit | [verification/actual_dataset_audit.json](verification/actual_dataset_audit.json) |
| HTML source documentation | [docs/main.html](docs/main.html) and [docs/pork_dataset.html](docs/pork_dataset.html) |
| Completed experiment and checkpoints | [results/20261005_053211/](results/20261005_053211/) |
| Dataset and dependency provenance | [SOURCES.md](SOURCES.md) |
| AI use and reflection | [AI_USE.md](AI_USE.md) |
| Project source licence | [LICENSE](LICENSE) |

Use the directory paths above for reproduction and evidence review. The assessed
written report is submitted separately. The original dataset archive is also
kept separately from this repository.

## Quick checks without training

The recorded training environment used Python 3.13.15; the stored software-test
report used Python 3.13.16. In a compatible Python environment, run these commands
from the repository root:

```bash
python -m pip install PyYAML
python main.py results
python -m unittest discover -s tests -v
```

`results` reads the saved final-test JSON. It does not train or run inference.
The software tests use small synthetic archive fixtures with known expected
outputs. The stored Colab report records **17 passing tests**, `OK` and exit
code 0 on 6 October 2026. This verifies selected code behaviours rather than
neural-model accuracy.

For a single-image demonstration:

```bash
python -m pip install -r requirements.txt
python main.py predict --image /path/to/image.jpg --device cpu
```

The default checkpoint is
`results/20261005_053211/train/yolov8n/weights/best.pt`. The preview uses confidence
0.25 and input size 640. This preview threshold is separate from the completed
evaluation's summary precision/recall. The model package version recorded for
the experiment is Ultralytics 8.4.120.

## Data and preparation

The source is Roboflow `hello-2aqe0/pork-rasher-error-packaging`, version 4,
exported in YOLOv8 format. Export metadata records CC BY 4.0; see
[SOURCES.md](SOURCES.md) for attribution. Original class ID 3, `unsealed`, is
remapped to 0. All images are retained, with empty target labels for images
without an unsealed annotation.

| Split | Images | Target-positive images | Target-negative images | Target boxes |
| --- | ---: | ---: | ---: | ---: |
| Training | 3349 | 1055 | 2294 | 1055 |
| Validation | 120 | 81 | 39 | 81 |
| Test | 80 | 49 | 31 | 49 |

Labels contain a class ID and normalised box centre, width and height. Preparation
checks archive paths, conflicting duplicates, class definitions, image/label
pairing, finite records and box bounds before filtering. These checks do not
prove annotation correctness or split independence.

The training record and actual-archive audit identify their input ZIP by this
recorded SHA256:

```text
88cc5b2527db50dc6e39d3d8c8379d1b23a6d8c7dd105303ef6551347ddbc66d
```

For reproduction of the audited input, use the original archive whose hash
matches these records. A matching filename alone does not establish that the
ZIP is the same file.

Prepare the archive in a fresh directory:

```bash
python main.py prepare --zip "/path/Pork Rasher Error-Packaging-.v4i.yolov8.zip" --destination /path/new_prepared_data
```

## Experiments and measured results

Both COCO-pretrained variants were fine-tuned for 30 epochs at `imgsz=640`,
batch 8, seed 42, AdamW and initial learning rate 0.001 on a Tesla T4. Full
settings are in `results/20261005_053211/experiment.json` and the individual
`train/yolov8n/args.yaml` and `train/yolov8s/args.yaml` files. Training logs and
best checkpoints are retained in the corresponding model directories.
The tables below report evaluations of the saved best checkpoints, rather than
the highest value of each metric observed anywhere in the training logs.

### Model comparison on validation data

| Metric | YOLOv8n | YOLOv8s |
| --- | ---: | ---: |
| Precision (%) | 90.88 | 92.37 |
| Recall (%) | 87.65 | 79.01 |
| mAP50 (%) | 90.14 | 86.94 |
| mAP50–95 (%) | 51.57 | 51.28 |
| Training duration (minutes) | 26.71 | 34.05 |
| T4 inference (ms/image) | 5.774 | 9.571 |

YOLOv8n offered higher recall and mAP50, shorter recorded training duration
and lower measured inference time. YOLOv8s had higher precision. Only one seed
was studied; no repeated-run uncertainty estimate or statistical significance
test was performed, so these results do not establish a reliable mAP50–95
advantage across training runs.

### Inference-resolution sensitivity

This study evaluated the **same nano checkpoint**, trained at 640, at two input
sizes. It did not retrain the model at 416.

| Metric | 416 | 640 |
| --- | ---: | ---: |
| Recall (%) | 65.00 | 87.65 |
| mAP50 (%) | 75.59 | 90.14 |
| mAP50–95 (%) | 30.49 | 51.57 |
| T4 inference (ms/image) | 5.167 | 5.305 |

The small recorded timing saving at 416 did not outweigh its lower detection
metrics. Nano at 640 was selected using validation evidence and recorded in
`final_design.json`. Timings here and in the model comparison come from separate
evaluation passes, not repeated benchmark statistics.

### Final evaluation on the supplied test split

| Metric | YOLOv8n at 640 |
| --- | ---: |
| Precision (%) | 87.87 |
| Recall (%) | 75.51 |
| mAP50 (%) | 79.98 |
| mAP50–95 (%) | 46.01 |
| CPU inference (ms/image) | 219.79 |

Source records are `validation_comparison.csv`, `resolution_study.csv` and
`final_test_metrics.json` inside the completed experiment directory. The final
test ran on an Intel Xeon CPU at 2.20 GHz; its timing must be kept separate from
T4 timings. None of these values establishes end-to-end conveyor throughput.

Original confusion matrices, precision–recall curves and annotation/prediction
batches are saved under `final_test/yolov8n/`. Summary precision/recall and plotted
matrix counts can represent different operating settings. Box-matching counts
are not an image-level pass/fail confusion matrix.

## Reproduction and evidence scope

Use [notebooks/MMA3001_training.ipynb](notebooks/MMA3001_training.ipynb) for the
full workflow. The updated notebook retains the original numbered headings and
labels troubleshooting additions with letters:

| Added cell | Purpose |
| --- | --- |
| Training 3A | Restore the latest completed 30-epoch comparison |
| Training 4A | Restore the specific recorded experiment `20261005_053211` |
| Training 12A | Check completed training logs and compare inference resolutions |
| Training 13A | Recover the separate historical three-epoch setup record |
| Training 13B | Display saved final-test annotations, predictions and confusion matrix |
| Companion verification 9A | Restore imports and saved metrics before the final GitHub clone check |

**Do not use Run all indiscriminately.** Cell 4 creates a new experiment, while
Cells 3A and 4A select existing experiments. Cell 13A switches `RUN_DIR` to the
historical setup run. Follow the route table in the updated training notebook.
Training and validation require a GPU in the supplied code. The saved final test
used CPU; Cell 13 reads the cached JSON when present rather than evaluating again.

For deliberate retraining, adapt Drive paths to the reviewer's account, create
a new experiment and skip the selectors for the existing project run. Review
later cells' fixed references to `20261005_053211` before using them for a new run.
Preserve the supplied split membership, matched settings and validation-based
selection; do not tune on the final test data. Exact results and timings can vary
with hardware and numerical behaviour.

The source-parity record contains the notebook and module hashes that were checked.
When replacing the notebook after text edits, refresh this record with companion
verification Cell 4; the preparation-function code was unchanged by the text update.
The actual-archive audit links the recorded input hash to prepared counts.

`verification/file_manifest.json` describes files at the time it was generated.
Refresh it from the reviewed repository files when repackaging; an older manifest
does not automatically describe later documentation or notebook edits.

After the final GitHub edits are committed, use companion verification Cells
**9A → 10** to check a fresh clone and generate `MMA3001_Final_Clone.zip`, including
the `.git` directory. That cell checks the saved-result entry point, reruns tests
and checks for the HTML files; it does not train the detector or publish changes.
The clone-verification report is written inside the downloaded clone, not
automatically committed back to GitHub. Retain the executed verification notebook
with its output separately. The original software-test report, fresh-clone check
and archive are distinct evidence records.

## Limitations

The data contain video-like conveyor frames; independence across recordings or
trays has not been established. There are only 80 test images and 49 target boxes.
Only one seed and a fixed training budget were studied. Early validation peaks
while training loss continued falling warrant investigation of overfitting.
Missed regions, extra detections and imperfect box localisation remain. No
physical leak test, deployment study or repeated timing benchmark was performed.

AI assistance is disclosed in the completed [AI_USE.md](AI_USE.md) reflection
and the written report. AI provided substantial code, writing and analysis
assistance; the student executed the Colab workflow and preserved its measured
results. The student remains responsible for reviewing the submitted claims,
references and engineering interpretation.
