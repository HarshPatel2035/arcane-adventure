"""Train the spellcasting gesture classifier on CPU.

Run: ./.venv/bin/python train_model.py
Dependencies: torch numpy pandas scikit-learn matplotlib seaborn
Uses collect_data.py's 126 normalized features: Left then Right hand slots.
Weights: models/gesture_model.pth; architecture/labels: gesture_model.json.
Reports: reports/training/. Softmax is applied only at inference, not before
CrossEntropyLoss. No webcam or Internet connection is needed for training.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from collect_data import HEADER, LABELS

ROOT = Path(__file__).resolve().parent
FEATURE_CONTRACT = {
    "input_features": 126,
    "hand_slots": ["Left", "Right"],
    "landmarks_per_hand": 21,
    "coordinates": ["x", "y", "z"],
    "mirrored_camera": True,
    "normalization": "subtract wrist; divide by 3D wrist-to-middle-MCP (index 9) distance",
    "missing_hand": "63 zeros",
}


class GestureClassifier(nn.Module):
    """Return logits; use torch.softmax(logits, dim=1) for probabilities."""

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(126, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, features):
        return self.network(features)


def load_data(path):
    table = pd.read_csv(path)
    if list(table.columns) != HEADER:
        raise ValueError("CSV must have columns label, f0, ..., f125 in that order.")
    if table.empty:
        raise ValueError("Dataset is empty. Collect samples first.")
    labels = pd.to_numeric(table['label'], errors='raise').to_numpy(dtype=np.float64)
    if not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all():
        raise ValueError("Labels must be finite integers.")
    if not set(labels).issubset(LABELS):
        raise ValueError("Labels must be in the range 0-5.")
    missing = set(range(5)) - set(labels)
    if missing:
        raise ValueError(f"Missing core gesture labels: {sorted(missing)}")
    features = table.iloc[:, 1:].to_numpy(dtype=np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Features contain missing or nonfinite values.")
    labels = labels.astype(np.int64)
    # Exact duplicates in different splits would inflate measured accuracy.
    seen, keep, duplicates = {}, [], 0
    for index, (row, label) in enumerate(zip(features, labels)):
        key = row.tobytes()
        if key in seen:
            if seen[key] != label:
                raise ValueError("Identical features have conflicting labels. Recheck your data.")
            duplicates += 1
        else:
            seen[key] = label
            keep.append(index)
    if duplicates:
        print(f"Removed {duplicates} exact duplicate samples before splitting.")
    features, labels = features[keep], labels[keep]
    classes, counts = np.unique(labels, return_counts=True)
    if np.any(counts < 10):
        raise ValueError("Each class needs at least 10 unique samples; aim for 200+.")
    return features, labels, classes.tolist(), np.asarray(keep), duplicates


def split_indices(labels, mode, seed):
    """Default: separate contiguous sections of each class's capture session.

    This avoids interleaving adjacent video frames across all three sets.
    Separate people/recording sessions are still needed for stronger evaluation.
    """
    if mode == 'random':
        indices = np.arange(len(labels))
        train, rest = train_test_split(indices, test_size=0.3, stratify=labels,
                                       random_state=seed)
        val, test = train_test_split(rest, test_size=0.5, stratify=labels[rest],
                                     random_state=seed)
        return train, val, test
    sets = [[], [], []]
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        boundary1, boundary2 = int(len(indices) * 0.7), int(len(indices) * 0.85)
        for output, part in zip(sets, (indices[:boundary1],
                                      indices[boundary1:boundary2], indices[boundary2:])):
            output.extend(part.tolist())
    return tuple(np.asarray(part, dtype=np.int64) for part in sets)


def augment_features(features):
    """Small rotations/noise and physical mirroring, only on present hands.

    Scale/translation are already removed by the collector's normalization.
    Reapplying them and normalizing here preserves that same invariant.
    Missing slots remain zero, wrists stay zero and MCP length stays one.
    """
    hands = features.clone().reshape(2, 21, 3)
    present = hands.abs().sum(dim=(1, 2)) > 0
    angle = (torch.rand(()) * 2 - 1) * math.radians(15)
    cos, sin = torch.cos(angle), torch.sin(angle)
    for slot in range(2):
        if not present[slot]:
            continue
        hand = hands[slot]
        x, y = hand[:, 0].clone(), hand[:, 1].clone()
        hand[:, 0], hand[:, 1] = cos * x - sin * y, sin * x + cos * y
        hand *= 0.9 + 0.2 * torch.rand(())
        hand += (torch.rand(3) - 0.5) * 0.04
        hand += torch.randn_like(hand) * 0.01
        hand -= hand[0].clone()
        hand /= torch.linalg.vector_norm(hand[9]).clamp_min(1e-6)
    if torch.rand(()) < 0.5:
        hands[:, :, 0] *= -1
        hands = hands.flip(0)  # Reflection swaps Left/Right hand slots.
    return hands.reshape(126)


class GestureDataset(Dataset):
    def __init__(self, features, labels, augment=False):
        self.features = torch.from_numpy(features.copy())
        self.labels = torch.from_numpy(labels.copy())
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        features = self.features[index]
        if self.augment:
            features = augment_features(features)
        return features, self.labels[index]


def evaluate(model, loader, criterion):
    model.eval()
    loss_sum, predicted, truth = 0.0, [], []
    with torch.inference_mode():
        for features, labels in loader:
            logits = model(features)
            loss_sum += criterion(logits, labels).item() * len(labels)
            predicted.extend(logits.argmax(dim=1).tolist())
            truth.extend(labels.tolist())
    return loss_sum / len(truth), accuracy_score(truth, predicted), truth, predicted


def save_plots(history, matrix, names, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs = np.arange(1, len(history) + 1)
    for axis, metric in zip(axes, ('loss', 'accuracy')):
        for split in ('train', 'val'):
            axis.plot(epochs, [row[f'{split}_{metric}'] for row in history], label=split)
        axis.set(xlabel='Epoch', ylabel=metric.capitalize())
        axis.legend()
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(directory / 'training_curves.png', dpi=160)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(9, 7))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=axis,
                xticklabels=names, yticklabels=names)
    axis.set(xlabel='Predicted gesture', ylabel='Actual gesture', title='Test confusion matrix')
    axis.tick_params(axis='x', rotation=35)
    fig.tight_layout()
    fig.savefig(directory / 'confusion_matrix.png', dpi=160)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/gestures.csv')
    parser.add_argument('--model', type=Path, default=ROOT / 'models/gesture_model.pth')
    parser.add_argument('--reports', type=Path, default=ROOT / 'reports/training')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--split', choices=['blocked', 'random'], default='blocked')
    parser.add_argument('--no-augmentation', action='store_true')
    args = parser.parse_args(argv)
    if min(args.epochs, args.patience, args.batch_size, args.threads) < 1:
        parser.error('Epochs, patience, batch size and threads must be positive.')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error('Learning rate must be positive and finite.')
    if not 0 <= args.seed < 2**32:
        parser.error('Seed must be between 0 and 2**32 - 1.')
    if args.data.resolve() == args.model.resolve():
        parser.error('Model output must differ from the input dataset.')
    return args


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(args.threads, os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True)
    features, original_labels, class_ids, source_rows, duplicates = load_data(args.data)
    labels = np.array([class_ids.index(int(value)) for value in original_labels], dtype=np.int64)
    splits = split_indices(labels, args.split, args.seed)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.reports.mkdir(parents=True, exist_ok=True)
    names = [LABELS[label].split(' / ')[0] for label in class_ids]
    print('Samples per gesture:', dict(zip(names, np.bincount(labels).tolist())), flush=True)
    print(f'CPU training | train/validation/test: {[len(part) for part in splits]}', flush=True)
    print('Evaluation uses one recording dataset; test accuracy may not generalize to new people.', flush=True)
    generator = torch.Generator().manual_seed(args.seed)
    loaders = []
    for index, part in enumerate(splits):
        dataset = GestureDataset(features[part], labels[part],
                                 augment=index == 0 and not args.no_augmentation)
        loaders.append(DataLoader(dataset, batch_size=args.batch_size, shuffle=index == 0,
                                   generator=generator if index == 0 else None, num_workers=0))
    model = GestureClassifier(len(class_ids))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_loss, best_epoch, stale = float('inf'), 0, 0
    best_state, history = None, []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for batch_features, batch_labels in loaders[0]:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            if not torch.isfinite(loss):
                raise ValueError('Training loss is nonfinite. Check landmark values.')
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch_labels)
            correct += (logits.argmax(1) == batch_labels).sum().item()
            total += len(batch_labels)
        val_loss, val_accuracy, _, _ = evaluate(model, loaders[1], criterion)
        history.append(dict(epoch=epoch, train_loss=loss_sum / total,
                            train_accuracy=correct / total, val_loss=val_loss,
                            val_accuracy=val_accuracy))
        print(f'Epoch {epoch:02d}/{args.epochs} | train {correct/total:.1%} | '
              f'validation {val_accuracy:.1%} | val loss {val_loss:.4f}', flush=True)
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= args.patience:
                print(f'Early stopping after {args.patience} epochs without improvement.')
                break
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), args.model)
    test_loss, test_accuracy, truth, predicted = evaluate(model, loaders[2], criterion)
    report = classification_report(truth, predicted, labels=list(range(len(class_ids))),
                                   target_names=names, output_dict=True, zero_division=0)
    report_text = classification_report(truth, predicted, labels=list(range(len(class_ids))),
                                        target_names=names, zero_division=0)
    matrix = confusion_matrix(truth, predicted, labels=list(range(len(class_ids))))
    model.eval()
    example = torch.from_numpy(features[splits[2][:1]])
    with torch.inference_mode():
        for _ in range(20):
            model(example)
        tick = time.perf_counter()
        for _ in range(200):
            model(example)
        latency = (time.perf_counter() - tick) * 1000 / 200
    metadata = dict(architecture={'input': 126, 'hidden': [128, 64], 'dropout': 0.3,
                                  'num_classes': len(class_ids)}, class_ids=class_ids,
                    class_names=names, feature_contract=FEATURE_CONTRACT,
                    best_epoch=best_epoch, seed=args.seed)
    args.model.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    metrics = dict(test_accuracy=test_accuracy, test_loss=test_loss, best_epoch=best_epoch,
                   best_val_loss=best_loss, epochs_run=len(history),
                   model_only_latency_ms=latency, training_seconds=time.perf_counter()-started,
                   split_mode=args.split, split_sizes=[len(part) for part in splits],
                   dataset_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                   exact_duplicates_removed=duplicates, classification_report=report,
                   confusion_matrix=matrix.tolist(),
                   limitation='Same recording dataset; evaluate on new people and separate sessions.')
    (args.reports / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    (args.reports / 'classification_report.txt').write_text(report_text)
    pd.DataFrame(history).to_csv(args.reports / 'training_history.csv', index=False)
    np.savez(args.reports / 'split_indices.npz',
             **{name: source_rows[part] for name, part in zip(('train','validation','test'),splits)})
    save_plots(history, matrix, names, args.reports)
    print('\n' + report_text)
    print(f'Test accuracy: {test_accuracy:.1%} | model-only latency: {latency:.3f} ms')
    print(f'Saved best model: {args.model}\nReports: {args.reports}')
    if test_accuracy < 0.95:
        print('Below the 95% goal: collect more varied examples for confused gestures.')


def main():
    args = parse_args()
    try:
        train(args)
    except KeyboardInterrupt:
        print('\nTraining interrupted. Run again to restart.', file=sys.stderr)
        return 130
    except Exception as exc:
        print(f'Training failed: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
