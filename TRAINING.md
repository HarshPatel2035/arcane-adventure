# Gesture model training

Run from this folder:

```bash
./.venv/bin/python train_model.py
```

Or double-click `Train Gesture Model.command`. Training libraries are installed in `.venv`.
To recreate that environment, install `requirements.txt` with Python 3.13.

The script validates `data/gestures.csv`, removes exact duplicates, rejects conflicting labels,
and splits each gesture into contiguous 70% training, 15% validation and 15% test sections.
This reduces interleaving of neighboring video frames. `--split random` provides the alternative
stratified random split. Separate people and separate recording sessions are needed to measure
how well the model generalizes. The default split of your 1,000 rows is 700/150/150.

Training runs on CPU with 50 epochs maximum, early stopping after 10 validation-loss epochs
without improvement, Adam at 0.001, batches of 32, and a 126 → 128 → 64 → 5 network with ReLU
and dropout 0.3. The final layer returns logits for CrossEntropyLoss; inference uses softmax.
See https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html.

Training-only augmentation applies small rotations, noise and reflection with Left/Right slot
swapping. Missing hand slots stay zero. Recentring and scaling preserve the collector's wrist
and hand-size invariants; translation and uniform scale themselves cancel under normalization.
Validation and test features are not augmented.

## Recommended: add an idle class

Without an idle class, any visible hand gets one of the five spell labels. Record idle
examples (relaxed hands, scratching your face, reaching for keys, half-made poses):

```bash
./.venv/bin/python collect_data.py --guided --include-idle
```

or double-click `Collect Idle Gesture.command`. Guided mode collects only unfinished
labels, so it records label 5 up to 200 samples. Then retrain. The game and
`inference.py` treat idle as "no spell".

## Honest accuracy: a separate session

The default split comes from one recording, so the test score is optimistic. Record a
second session (different day, lighting, other hand or another person) into its own
file, keep it out of training, and score the model on it:

```bash
./.venv/bin/python collect_data.py --guided --output data/session2.csv --target 60
./.venv/bin/python train_model.py --holdout data/session2.csv
```

The holdout accuracy, confusion matrix and report are printed and saved under
`holdout` in `reports/training/metrics.json`. A holdout file may cover only some
gestures. If holdout accuracy is much lower than test accuracy, add varied
samples to `data/gestures.csv` and retrain.

## Outputs

- `models/gesture_model.pth`: plain PyTorch state dictionary for the best validation-loss epoch.
- `models/gesture_model.json`: architecture, class IDs/names and preprocessing contract.
- `reports/training/training_curves.png`: training/validation loss and accuracy.
- `reports/training/confusion_matrix.png`: test errors by gesture.
- `reports/training/classification_report.txt`: precision, recall and F1 scores.
- `reports/training/metrics.json`: accuracy, latency, split sizes and dataset fingerprint.
- `reports/training/training_history.csv`: metrics per epoch.
- `reports/training/split_indices.npz`: zero-based source data-row indices for each split.

The verified default run scored 98.7% test accuracy (148/150). This measures your current
recording, not independent new users. Reported latency measures the classifier alone,
excluding webcam capture, MediaPipe detection and game rendering. Ordinary retraining replaces
model/report outputs but does not change the collected dataset.

## Options and troubleshooting

`--epochs 100`, `--patience 15`, `--no-augmentation`, `--seed 42`, `--threads 2`,
`--data PATH`, `--model PATH`, `--reports PATH` and `--holdout PATH` customize training.

If dependencies are missing or Python crashes, use `.venv/bin/python` rather than Anaconda base.
An empty dataset or missing gesture class requires collecting more samples. If accuracy is low,
inspect the confusion matrix, check labels and collect additional varied poses. CPU training is
expected; no GPU or network is needed after setup. The classifier has no idle output unless
label 5 was collected, so recognizing 'no spell' with hands visible needs more idle data later.
