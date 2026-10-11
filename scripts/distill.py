"""Distil a trained teacher into a small student for low-resource hardware.

Reports the accuracy/size trade-off explicitly, because that trade-off is the
entire deployment argument: how much performance is given up for how much
smaller a model, and does the student beat the same architecture trained on
hard labels alone?

Usage::

    python -m scripts.distill --teacher transfer --epochs 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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


def _bench_latency(model, n: int = 25) -> float:
    """Mean single-image inference latency in ms (CPU-relevant metric)."""
    x = np.random.rand(1, *config.IMAGE_SIZE, config.IMAGE_CHANNELS).astype("float32")
    model.predict(x, verbose=0)                    # warm up
    start = time.perf_counter()
    for _ in range(n):
        model.predict(x, verbose=0)
    return (time.perf_counter() - start) / n * 1000.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Knowledge distillation")
    ap.add_argument("--teacher", choices=["cnn", "transfer"], default="transfer")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--temperature", type=float, default=4.0)
    ap.add_argument("--width", type=float, default=0.5)
    ap.add_argument("--no-baseline", action="store_true")
    args = ap.parse_args()

    import tensorflow as tf

    from src.distillation import (
        build_student,
        compile_student_for_distillation,
        make_distillation_dataset,
        model_footprint,
    )
    from src.evaluate import compute_metrics
    from src.thresholds import best_threshold_f1

    config.ensure_output_dirs()
    if not splits_exist():
        raise SystemExit("No splits. Run 'python -m src.preprocessing' first.")
    train, val, test = load_splits()
    tf.keras.utils.set_random_seed(config.RANDOM_SEED)

    teacher_path = (config.CNN_MODEL_FILE if args.teacher == "cnn"
                    else config.TRANSFER_MODEL_FILE)
    if not teacher_path.exists():
        raise SystemExit(f"Teacher artifact missing: {teacher_path}. Train it first.")
    teacher = tf.keras.models.load_model(teacher_path)
    print(f"[distill] teacher={args.teacher} {model_footprint(teacher)}")

    train_ds = make_tf_dataset(train, training=True)
    val_ds = make_tf_dataset(val, training=False)
    y_val = np.array([s.label for s in val])
    y_test = np.array([s.label for s in test])

    # ---- teacher reference on test -----------------------------------------
    tp_val = teacher.predict(make_tf_dataset(val, training=False, shuffle=False),
                             verbose=0).reshape(-1)
    t_thr = best_threshold_f1(y_val, tp_val)
    tp_test = teacher.predict(make_tf_dataset(test, training=False, shuffle=False),
                              verbose=0).reshape(-1)
    teacher_metrics = compute_metrics(y_test, tp_test, threshold=t_thr)

    # ---- distillation ------------------------------------------------------
    student = build_student(width=args.width)
    print(f"[distill] student {model_footprint(student)}")
    compile_student_for_distillation(student, alpha=args.alpha,
                                     temperature=args.temperature)
    print("[distill] precomputing teacher soft targets ...")
    d_train = make_distillation_dataset(train, teacher, training=True)
    d_val = make_distillation_dataset(val, teacher, training=False)
    student.fit(
        d_train, validation_data=d_val, epochs=args.epochs,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=6, restore_best_weights=True)],
        verbose=2,
    )
    # Recompile with a standard loss so the saved artifact loads anywhere.
    student.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                    loss="binary_crossentropy",
                    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    student.save(config.STUDENT_MODEL_FILE)
    print(f"[distill] student saved -> {config.STUDENT_MODEL_FILE}")

    sp_val = student.predict(make_tf_dataset(val, training=False, shuffle=False),
                             verbose=0).reshape(-1)
    s_thr = best_threshold_f1(y_val, sp_val)
    sp_test = student.predict(make_tf_dataset(test, training=False, shuffle=False),
                              verbose=0).reshape(-1)
    student_metrics = compute_metrics(y_test, sp_test, threshold=s_thr)

    payload = {
        "teacher": {
            "key": args.teacher, **model_footprint(teacher),
            "test": teacher_metrics, "threshold_from_val": t_thr,
            "latency_ms": round(_bench_latency(teacher), 2),
        },
        "student_distilled": {
            "width": args.width, **model_footprint(student),
            "test": student_metrics, "threshold_from_val": s_thr,
            "latency_ms": round(_bench_latency(student), 2),
            "alpha": args.alpha, "temperature": args.temperature,
        },
    }

    # ---- matched baseline: same student, hard labels only ------------------
    if not args.no_baseline:
        print("\n[baseline] same student architecture on hard labels only")
        tf.keras.utils.set_random_seed(config.RANDOM_SEED)
        plain = build_student(width=args.width)
        plain.fit(
            train_ds, validation_data=val_ds, epochs=args.epochs,
            class_weight=compute_class_weights(train),
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", mode="max", patience=6,
                restore_best_weights=True)],
            verbose=2,
        )
        bp_val = plain.predict(make_tf_dataset(val, training=False, shuffle=False),
                               verbose=0).reshape(-1)
        b_thr = best_threshold_f1(y_val, bp_val)
        bp_test = plain.predict(make_tf_dataset(test, training=False, shuffle=False),
                                verbose=0).reshape(-1)
        payload["student_hard_labels_only"] = {
            **model_footprint(plain),
            "test": compute_metrics(y_test, bp_test, threshold=b_thr),
            "threshold_from_val": b_thr,
        }

    out = config.METRICS_DIR / "distillation.json"
    out.write_text(json.dumps(payload, indent=2))

    # ---- summary -----------------------------------------------------------
    t_inf = payload["teacher"]
    s_inf = payload["student_distilled"]
    ratio = t_inf["parameters"] / max(s_inf["parameters"], 1)
    print("\n" + "=" * 70)
    print(f"{'model':<26}{'params':>10}{'MB':>8}{'f1':>8}{'auc':>8}{'ms':>8}")
    print("-" * 70)
    print(f"{'teacher (' + args.teacher + ')':<26}{t_inf['parameters']:>10}"
          f"{t_inf['approx_float32_mb']:>8.2f}{t_inf['test']['f1']:>8.3f}"
          f"{(t_inf['test']['roc_auc'] or 0):>8.3f}{t_inf['latency_ms']:>8.1f}")
    print(f"{'student (distilled)':<26}{s_inf['parameters']:>10}"
          f"{s_inf['approx_float32_mb']:>8.2f}{s_inf['test']['f1']:>8.3f}"
          f"{(s_inf['test']['roc_auc'] or 0):>8.3f}{s_inf['latency_ms']:>8.1f}")
    if "student_hard_labels_only" in payload:
        b = payload["student_hard_labels_only"]
        print(f"{'student (hard labels)':<26}{b['parameters']:>10}"
              f"{b['approx_float32_mb']:>8.2f}{b['test']['f1']:>8.3f}"
              f"{(b['test']['roc_auc'] or 0):>8.3f}{'-':>8}")
    print("=" * 70)
    print(f"Compression: {ratio:.1f}x fewer parameters than the teacher.")
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
