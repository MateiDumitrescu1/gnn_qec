"""Train or sanity-check the TensorFlow GNN decoder."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from qec_gnn.config import DEFAULT_CONFIG, METRICS_DIR, MODEL_DIR
from qec_gnn.dataset import load_graph_dataset
from qec_gnn.model import build_gnn_model
from qec_gnn.utils import binary_classification_metrics, print_metric_block, save_json


class StopAtAccuracy(tf.keras.callbacks.Callback):
    """Stop tiny-overfit runs once the README acceptance criterion is met."""

    def __init__(self, target: float = 0.95):
        super().__init__()
        self.target = target

    def on_epoch_end(self, epoch, logs=None):
        if logs and logs.get("accuracy", 0.0) >= self.target:
            self.model.stop_training = True


def _scalar(value: Any) -> Any:
    array = np.asarray(value)
    return array.item() if array.shape == () else array.tolist()


def _class_weight(y: np.ndarray) -> dict[int, float] | None:
    y = y.astype(np.uint8)
    counts = np.bincount(y, minlength=2)
    if np.any(counts == 0):
        return None
    total = float(np.sum(counts))
    return {0: total / (2.0 * counts[0]), 1: total / (2.0 * counts[1])}


def train_gnn(
    data_path: Path,
    *,
    output_path: Path | None,
    metrics_path: Path,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    learning_rate: float,
    dropout: float,
    tiny_overfit: bool,
    tiny_examples: int,
    seed: int,
) -> dict[str, Any]:
    tf.keras.utils.set_random_seed(seed)
    data = load_graph_dataset(data_path)

    x = data["X"]
    a = data["A"]
    mask = data["mask"]
    y = data["y"]
    max_nodes = x.shape[1]
    feature_dim = x.shape[2]

    if tiny_overfit:
        indices = np.arange(min(tiny_examples, len(y)))
        train_idx = indices
        val_idx = indices
        test_idx = indices
        epochs = max(epochs, 400)
        hidden_dim = max(hidden_dim, 128)
        dropout = 0.0
    else:
        train_idx = data["train_idx"]
        val_idx = data["val_idx"]
        test_idx = data["test_idx"]

    model = build_gnn_model(
        max_nodes=max_nodes,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        learning_rate=learning_rate,
    )

    callbacks = []
    if not tiny_overfit:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
            )
        )
    else:
        callbacks.append(StopAtAccuracy(target=0.95))

    history = model.fit(
        [x[train_idx], a[train_idx], mask[train_idx]],
        y[train_idx],
        validation_data=([x[val_idx], a[val_idx], mask[val_idx]], y[val_idx]),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
        callbacks=callbacks,
        class_weight=None if tiny_overfit else _class_weight(y[train_idx]),
    )

    train_prob = model.predict([x[train_idx], a[train_idx], mask[train_idx]], batch_size=batch_size, verbose=0)
    test_prob = model.predict([x[test_idx], a[test_idx], mask[test_idx]], batch_size=batch_size, verbose=0)
    train_metrics = binary_classification_metrics(y[train_idx], train_prob[:, 0])
    test_metrics = binary_classification_metrics(y[test_idx], test_prob[:, 0])

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(output_path)

    config = {
        "data_path": str(data_path),
        "model_path": str(output_path) if output_path is not None else None,
        "tiny_overfit": tiny_overfit,
        "epochs_requested": epochs,
        "epochs_ran": len(history.history["loss"]),
        "batch_size": batch_size,
        "hidden_dim": hidden_dim,
        "learning_rate": learning_rate,
        "dropout": dropout,
        "max_nodes": max_nodes,
        "feature_dim": feature_dim,
        "seed": seed,
    }
    for key in (
        "distance",
        "rounds",
        "p",
        "shots",
        "k_neighbors",
        "truncation_rate",
        "empty_graph_fraction",
    ):
        if key in data:
            config[key] = _scalar(data[key])

    payload = {
        "kind": "gnn",
        "config": config,
        "train": train_metrics,
        "test": test_metrics,
        "history": {key: [float(v) for v in values] for key, values in history.history.items()},
    }
    save_json(metrics_path, payload)

    print_metric_block("GNN train", train_metrics)
    print_metric_block("GNN test", test_metrics)
    print(f"metrics = {metrics_path}")
    if output_path is not None:
        print(f"model = {output_path}")
    return payload


def parse_args() -> argparse.Namespace:
    cfg = DEFAULT_CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--hidden-dim", type=int, default=cfg.hidden_dim)
    parser.add_argument("--learning-rate", type=float, default=cfg.learning_rate)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--tiny-overfit", action="store_true")
    parser.add_argument("--tiny-examples", type=int, default=128)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--metrics-out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out is None and not args.tiny_overfit:
        args.out = MODEL_DIR / f"gnn_{args.data.stem}.keras"
    metrics_path = args.metrics_out or METRICS_DIR / (
        f"gnn_tiny_overfit_{args.data.stem}.json" if args.tiny_overfit else f"gnn_{args.data.stem}.json"
    )
    train_gnn(
        args.data,
        output_path=args.out,
        metrics_path=metrics_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        tiny_overfit=args.tiny_overfit,
        tiny_examples=args.tiny_examples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
