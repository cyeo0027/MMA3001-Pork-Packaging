# MMA3001: Detecting annotated unsealed pork packaging

Individual project by Yeo Chun Hau. The detector localises regions labelled
"unsealed" in Roboflow pork-packaging images. It does not certify airtightness,
meat freshness or food safety. Images without the target annotation can contain
other packaging defects.

## Files
- notebooks/MMA3001_training.ipynb: the student's saved notebook with experiment outputs.
- pork_dataset.py: unchanged preparation function from that notebook's Cell 5.
- main.py: command-line preparation, saved-result display and single-image demonstration.
- tests/test_project.py: known-input software tests using synthetic archive fixtures.
- docs/: generated HTML source documentation.
- results/20261005_053211/: copied completed experiment evidence and best checkpoints.
- verification/: source-parity check, actual-archive audit, test report and runtime metadata.
- SOURCES.md: dataset and dependency provenance.
- AI_USE.md: assistance disclosure notes to review and complete.

## Setup and quick checks
Python 3 is required. For preparation, results display and software tests:

    python -m pip install PyYAML
    python main.py results
    python -m unittest discover -s tests -v

For the single-image model demonstration and training dependencies:

    python -m pip install -r requirements.txt
    python main.py predict --image /path/to/image.jpg --device cpu

The default prediction checkpoint is the saved YOLOv8n best.pt in the results
folder. Preview predictions use confidence 0.25 and 640-pixel input. This preview
threshold is separate from the completed evaluation's summary precision/recall.

## Dataset and reproduction
Obtain the original version-4 YOLOv8 export identified in SOURCES.md. The original
archive is distributed separately because it is about 241 MB. Its expected SHA256:

88cc5b2527db50dc6e39d3d8c8379d1b23a6d8c7dd105303ef6551347ddbc66d

To prepare it in a fresh directory:

    python main.py prepare --zip "/path/Pork Rasher Error-Packaging-.v4i.yolov8.zip" --destination /path/new_prepared_data

Expected images: train 3349, valid 120, test 80.
Expected target boxes: train 1055, valid 81, test 49.
Class ID 3 becomes 0. All images are retained, including target-negative images.

The full training workflow is in notebooks/MMA3001_training.ipynb. Its Drive paths
must be adapted to the reviewer's own project folder. To reproduce the model
comparison, use a NEW experiment folder, 30 epochs, 640-pixel input, batch 8,
seed 42 and both YOLOv8n and YOLOv8s. Use the settings recorded in experiment.json
and train/args.yaml. Existing completed experiments are reused by the notebook;
a fresh folder is necessary for deliberate retraining.

Compare the models on validation data; the sensitivity study evaluates the same
nano checkpoint at 416 and 640 pixels. Final YOLOv8n/640 was selected using
validation evidence before the held-out test. Review the frozen final test results
without using them to adjust the model. Exact rerun results can vary with runtime,
hardware and numerical behaviour.

## Verification and validation
Software verification is in verification/software_test_report.txt. Tests check
class filtering/remapping, negative-image retention, label validity, missing
inputs, duplicate/path handling and CLI behaviour. Fixtures have known expected
outputs. Source parity links the tested preparation function to the saved notebook.
The actual-archive audit records the input hash and the prepared split counts.

Model validation is separate: validation_comparison.csv, resolution_study.csv and
final_test_metrics.json contain the actual measured detection results. Training
and model-comparison timing came from a Tesla T4. The saved final test ran on an
Intel Xeon CPU at 2.20 GHz; its inference timing is recorded separately as CPU
timing. Summary precision/recall and the plotted confusion matrix can use different
operating settings; matrix counts must be labelled with their own settings.

## Limitations
The supplied data contain video-like frames and similar filename prefixes.
Exact-byte checks do not establish independence between recordings or trays.
Near-duplicate/source-group overlap remains a limitation unless provenance
demonstrates independence. There are only 80 test images and 49 target boxes.
Annotations represent visual regions rather than a physical seal-integrity test.
One seed was used, so uncertainty across training seeds was not measured.
