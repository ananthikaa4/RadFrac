"""Run SimCLR self-supervised pretraining, then fine-tune for fracture screening.

Pretraining uses **no labels**. To keep the evaluation airtight it runs on the
training split only, so no test image ever influences the representation.

The point of the experiment is the comparison it enables: an identical
architecture trained (a) from random init on the labelled data alone, versus
(b) contrastively pretrained then fine-tuned. With small labelled sets the
latter should need fewer labels to reach the same performance.

Usage::

    python -m scripts.ssl_pretrain --ssl-epochs 20 --finetune-epochs 15
    python -m scripts.ssl_pretrain --label-fraction 0.25   # low-label regime
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.preprocessing import (  # noqa: E402
    compute_class_weights,
    load_splits,
    make_tf_dataset,
    splits_exist,
)


def _subsample(samples, fraction: float, seed: int):
    """Stratified subsample to simulate a scarce-label regime."""
    if fraction >= 1.0:
        return samples
    rng = np.random.default_rng(seed)
    by_class: dict[int, list] = {}
    for s in samples:
        by_class.setdefault(s.label, []).append(s)
    out = []
    for label, items in by_class.items():
        k = max(2, int(round(len(items) * fraction)))
        idx = rng.choice(len(items), size=min(k, len(items)), replace=False)
        out.extend(items[i] for i in idx)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="SimCLR pretrain + fine-tune")
    ap.add_argument("--ssl-epochs", type=int, default=20)
    ap.add_argument("--finetune-epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--label-fraction", type=float, default=1.0,
                    help="fraction of labelled training data for fine-tuning")
    ap.add_argument("--baseline", action="store_true",
                    help="also train the same architecture from scratch for comparison")
    args = ap.parse_args()

    import tensorflow as tf

    from src.ssl_pretrain import (
        build_encoder,
        build_finetune_model,
        build_projection_head,
        build_simclr,
        make_contrastive_dataset,
    )

    config.ensure_output_dirs()
    if not splits_exist():
        raise SystemExit("No splits. Run 'python -m src.preprocessing' first.")
    train, val, test = load_splits()
    tf.keras.utils.set_random_seed(config.RANDOM_SEED)

    # ---------------- Stage 1: contrastive pretraining (no labels) ----------
    print(f"[ssl] contrastive pretraining on {len(train)} UNLABELLED train images "
          f"({args.ssl_epochs} epochs, batch {args.batch_size})")
    if len(train) < args.batch_size * 2:
        args.batch_size = max(4, len(train) // 4)
        print(f"[ssl] small dataset - reduced batch size to {args.batch_size}")

    ds = make_contrastive_dataset(train, batch_size=args.batch_size)
    encoder = build_encoder()
    head = build_projection_head(encoder.output_shape[-1])
    simclr = build_simclr(encoder, head, temperature=args.temperature)
    simclr.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    hist = simclr.fit(ds, epochs=args.ssl_epochs, verbose=2)
    ssl_losses = [float(v) for v in hist.history.get("loss", [])]

    encoder.save(config.SSL_ENCODER_FILE)
    print(f"[ssl] encoder saved -> {config.SSL_ENCODER_FILE}")

    # ---------------- Stage 2: supervised fine-tuning -----------------------
    ft_train = _subsample(train, args.label_fraction, config.RANDOM_SEED)
    print(f"[ssl] fine-tuning on {len(ft_train)} labelled images "
          f"({args.label_fraction:.0%} of train)")

    model = build_finetune_model(encoder)
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=5, restore_best_weights=True)]
    model.fit(
        make_tf_dataset(ft_train, training=True),
        validation_data=make_tf_dataset(val, training=False),
        epochs=args.finetune_epochs,
        class_weight=compute_class_weights(ft_train),
        callbacks=callbacks, verbose=2,
    )
    model.save(config.SSL_FINETUNED_FILE)
    print(f"[ssl] fine-tuned model saved -> {config.SSL_FINETUNED_FILE}")

    # ---------------- Evaluate on the held-out test split -------------------
    from src.evaluate import compute_metrics
    from src.thresholds import best_threshold_f1

    y_val = np.array([s.label for s in val])
    p_val = model.predict(make_tf_dataset(val, training=False, shuffle=False),
                          verbose=0).reshape(-1)
    t = best_threshold_f1(y_val, p_val)          # operating point from val only

    y_test = np.array([s.label for s in test])
    p_test = model.predict(make_tf_dataset(test, training=False, shuffle=False),
                           verbose=0).reshape(-1)
    ssl_metrics = compute_metrics(y_test, p_test, threshold=t)
    print(f"\n[ssl] fine-tuned test: acc={ssl_metrics['accuracy']:.3f} "
          f"f1={ssl_metrics['f1']:.3f} roc_auc={ssl_metrics['roc_auc']} (t={t:.3f})")

    payload = {
        "ssl_epochs": args.ssl_epochs,
        "ssl_loss_first": ssl_losses[0] if ssl_losses else None,
        "ssl_loss_last": ssl_losses[-1] if ssl_losses else None,
        "ssl_loss_curve": ssl_losses,
        "label_fraction": args.label_fraction,
        "n_labelled_used": len(ft_train),
        "threshold_from_val": t,
        "pretrained_finetuned_test": ssl_metrics,
        "pretraining_data": "train split only (no labels, no test leakage)",
    }

    # ---------------- Optional matched baseline -----------------------------
    if args.baseline:
        print(f"\n[baseline] same architecture from random init on the same "
              f"{len(ft_train)} labelled images")
        tf.keras.utils.set_random_seed(config.RANDOM_SEED)
        fresh = build_finetune_model(build_encoder(name="scratch_encoder"))
        fresh.fit(
            make_tf_dataset(ft_train, training=True),
            validation_data=make_tf_dataset(val, training=False),
            epochs=args.finetune_epochs,
            class_weight=compute_class_weights(ft_train),
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", mode="max", patience=5,
                restore_best_weights=True)],
            verbose=2,
        )
        bp_val = fresh.predict(make_tf_dataset(val, training=False, shuffle=False),
                               verbose=0).reshape(-1)
        bt = best_threshold_f1(y_val, bp_val)
        bp_test = fresh.predict(make_tf_dataset(test, training=False, shuffle=False),
                                verbose=0).reshape(-1)
        base_metrics = compute_metrics(y_test, bp_test, threshold=bt)
        payload["scratch_baseline_test"] = base_metrics
        payload["scratch_threshold_from_val"] = bt
        print(f"[baseline] test: acc={base_metrics['accuracy']:.3f} "
              f"f1={base_metrics['f1']:.3f} roc_auc={base_metrics['roc_auc']}")
        print(f"\nComparison at {args.label_fraction:.0%} labels: "
              f"SSL f1={ssl_metrics['f1']:.3f} vs scratch f1={base_metrics['f1']:.3f}")

    out = config.METRICS_DIR / "ssl_results.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nresults -> {out}")


if __name__ == "__main__":
    main()
