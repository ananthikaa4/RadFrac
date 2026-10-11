"""Run the duplicate / near-duplicate leakage audit. Read-only on models & data.

Outputs (all NEW files - nothing existing is modified):
  outputs/duplicate_audit/duplicate_groups.csv    every image in a multi-image group
  outputs/duplicate_audit/audit_report.json       all statistics
  outputs/duplicate_audit/examples/*.png          visual side-by-side proof
  data/splits_grouped/{train,val,test}.csv        leakage-safe group-aware split

Usage::

    python -m scripts.duplicate_audit
    python -m scripts.duplicate_audit --no-split      # audit only
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.dup_audit import (  # noqa: E402
    CORR_STRONG,
    build_groups,
    candidate_pairs,
    load_records,
    verify_pairs,
)
from src.preprocessing import load_splits, splits_exist  # noqa: E402

OUT_DIR = config.OUTPUT_DIR / "duplicate_audit"
EX_DIR = OUT_DIR / "examples"
GROUPED_SPLIT_DIR = config.DATA_ROOT / "splits_grouped"


# --------------------------------------------------------------------------- #
# Visual proof
# --------------------------------------------------------------------------- #
def render_examples(records, groups, confirmed, limit: int = 12) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    EX_DIR.mkdir(parents=True, exist_ok=True)
    by_pair = {(p["i"], p["j"]): p for p in confirmed}

    # Prioritise: cross-split groups first, then strongest correlation.
    multi = [(gid, idxs) for gid, idxs in groups.items() if len(idxs) > 1]

    def priority(item):
        _gid, idxs = item
        splits = {records[i].split for i in idxs}
        labels = {records[i].label for i in idxs}
        cross = len(splits) > 1
        conflict = len(labels) > 1
        return (not cross, not conflict, -len(idxs))

    multi.sort(key=priority)
    written = 0
    for gid, idxs in multi[:limit]:
        idxs = idxs[:4]
        fig, axes = plt.subplots(1, len(idxs), figsize=(3.1 * len(idxs), 3.6))
        if len(idxs) == 1:
            axes = [axes]
        for ax, i in zip(axes, idxs):
            r = records[i]
            with Image.open(r.filepath) as im:
                ax.imshow(np.asarray(im.convert("L")), cmap="gray")
            ax.set_title(f"{r.name}\n{config.CLASS_NAMES[r.label]} · {r.split}",
                         fontsize=8)
            ax.axis("off")

        scores = [by_pair[(a, b)] for a in idxs for b in idxs
                  if (a, b) in by_pair]
        desc = ""
        if scores:
            top = max(scores, key=lambda d: d["score"])
            desc = f"  |  best match: {top['kind']}, corr={top['score']}, {top['transform']}"
        splits = sorted({records[i].split for i in idxs})
        labels = sorted({config.CLASS_NAMES[records[i].label] for i in idxs})
        fig.suptitle(f"group {gid}  |  splits: {'+'.join(splits)}  |  "
                     f"labels: {'+'.join(labels)}{desc}", fontsize=9)
        fig.tight_layout()
        fig.savefig(EX_DIR / f"group_{gid:06d}.png", dpi=110)
        plt.close(fig)
        written += 1
    return written


# --------------------------------------------------------------------------- #
# Group-aware split
# --------------------------------------------------------------------------- #
def make_group_split(records, groups, test_size=config.TEST_SIZE,
                     val_size=config.VAL_SIZE, seed=config.RANDOM_SEED):
    """Assign whole groups to splits, keeping class balance close to target.

    Greedy: shuffle groups, then place each group into whichever split is
    furthest below its quota for that group's majority class. Because a group is
    never divided, no underlying film can appear in two splits.
    """
    rng = np.random.default_rng(seed)
    gids = list(groups)
    rng.shuffle(gids)

    n_total = len(records)
    targets = {
        "train": (1.0 - test_size - val_size) * n_total,
        "val": val_size * n_total,
        "test": test_size * n_total,
    }
    # Per-class quotas keep balance rather than just total counts.
    class_totals = collections.Counter(r.label for r in records)
    quota = {
        sp: {lab: targets[sp] / n_total * cnt for lab, cnt in class_totals.items()}
        for sp in targets
    }
    filled = {sp: collections.Counter() for sp in targets}
    assignment: dict[int, str] = {}

    for gid in gids:
        idxs = groups[gid]
        counts = collections.Counter(records[i].label for i in idxs)
        major = counts.most_common(1)[0][0]
        # Deficit = how far this split still is from its quota for that class.
        best_sp = max(targets, key=lambda sp: (quota[sp][major] - filled[sp][major]))
        assignment[gid] = best_sp
        for lab, c in counts.items():
            filled[best_sp][lab] += c

    split_records = {sp: [] for sp in targets}
    for gid, sp in assignment.items():
        for i in groups[gid]:
            split_records[sp].append(records[i])
    return split_records, assignment


def write_group_split(split_records, assignment, groups, records) -> None:
    GROUPED_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for sp in ("train", "val", "test"):
        with open(GROUPED_SPLIT_DIR / f"{sp}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["filepath", "label"])
            for r in split_records[sp]:
                w.writerow([r.filepath, r.label])

    # Persist the grouping itself so the split is reproducible / auditable.
    gid_of = {}
    for gid, idxs in groups.items():
        for i in idxs:
            gid_of[i] = gid
    with open(GROUPED_SPLIT_DIR / "groups.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filepath", "label", "group_id", "assigned_split"])
        for r in records:
            g = gid_of[r.index]
            w.writerow([r.filepath, r.label, g, assignment[g]])

    summary = {
        "counts": {sp: len(v) for sp, v in split_records.items()},
        "class_names": config.CLASS_NAMES,
        "group_aware": True,
        "note": ("Whole duplicate groups are assigned to exactly one split, so no "
                 "underlying film appears in more than one split."),
    }
    for sp, v in split_records.items():
        c = collections.Counter(r.label for r in v)
        summary[f"{sp}_per_class"] = {config.CLASS_NAMES[k]: c[k] for k in sorted(c)}
    (GROUPED_SPLIT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Duplicate / near-duplicate audit")
    ap.add_argument("--no-split", action="store_true",
                    help="audit only; do not write a group-aware split")
    ap.add_argument("--examples", type=int, default=12)
    args = ap.parse_args()

    if not splits_exist():
        raise SystemExit("No splits found. Run 'python -m src.preprocessing' first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train, val, test = load_splits()
    splits = {"train": train, "val": val, "test": test}
    print(f"dataset: train={len(train)} val={len(val)} test={len(test)} "
          f"total={len(train) + len(val) + len(test)}")

    print("\n[1/5] hashing (sha256 + dihedral-canonical dHash/pHash) ...")
    records = load_records(splits)

    print("\n[2/5] exact-duplicate pass ...")
    by_sha: dict[str, list[int]] = {}
    for r in records:
        by_sha.setdefault(r.sha256, []).append(r.index)
    exact_groups = {k: v for k, v in by_sha.items() if len(v) > 1}
    n_exact_extra = sum(len(v) - 1 for v in exact_groups.values())
    print(f"    byte-identical groups: {len(exact_groups)} "
          f"({n_exact_extra} redundant files)")

    print("\n[3/5] perceptual candidate search ...")
    pairs = candidate_pairs(records)
    print(f"    candidate pairs: {len(pairs)}")

    print("\n[4/5] pixel verification of candidates ...")
    confirmed = verify_pairs(records, pairs)
    n_exact_pairs = sum(1 for c in confirmed if c["kind"] == "exact")
    n_near_pairs = sum(1 for c in confirmed if c["kind"] == "near")
    print(f"    confirmed pairs: {len(confirmed)} "
          f"(exact={n_exact_pairs}, near={n_near_pairs}); "
          f"rejected {len(pairs) - len(confirmed)} as merely similar")

    print("\n[5/5] grouping + leakage statistics ...")
    groups = build_groups(records, confirmed)
    multi = {g: i for g, i in groups.items() if len(i) > 1}

    split_of = {r.index: r.split for r in records}
    label_of = {r.index: r.label for r in records}

    cross_tr_te = cross_tr_val = cross_val_te = 0
    label_conflicts = 0
    test_with_train_dup = set()
    val_with_train_dup = set()
    for gid, idxs in multi.items():
        sps = {split_of[i] for i in idxs}
        labs = {label_of[i] for i in idxs}
        if len(labs) > 1:
            label_conflicts += 1
        if "train" in sps and "test" in sps:
            cross_tr_te += 1
            test_with_train_dup.update(i for i in idxs if split_of[i] == "test")
        if "train" in sps and "val" in sps:
            cross_tr_val += 1
            val_with_train_dup.update(i for i in idxs if split_of[i] == "val")
        if "val" in sps and "test" in sps:
            cross_val_te += 1

    pct_test = len(test_with_train_dup) / max(len(test), 1) * 100
    pct_val = len(val_with_train_dup) / max(len(val), 1) * 100

    # --- CSV of every grouped image ---------------------------------------
    with open(OUT_DIR / "duplicate_groups.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["group_id", "filename", "filepath", "class_label", "split",
                    "sha256_prefix", "dhash", "phash", "group_size",
                    "group_splits", "group_has_label_conflict"])
        for gid, idxs in sorted(multi.items(), key=lambda kv: -len(kv[1])):
            sps = "+".join(sorted({split_of[i] for i in idxs}))
            conflict = len({label_of[i] for i in idxs}) > 1
            for i in idxs:
                r = records[i]
                w.writerow([gid, r.name, r.filepath, config.CLASS_NAMES[r.label],
                            r.split, r.sha256[:12], int(r.dhash), int(r.phash),
                            len(idxs), sps, conflict])

    # --- pair-level detail -------------------------------------------------
    with open(OUT_DIR / "confirmed_pairs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file_a", "split_a", "label_a", "file_b", "split_b", "label_b",
                    "kind", "similarity", "transform"])
        for c in sorted(confirmed, key=lambda d: -d["score"]):
            a, b = records[c["i"]], records[c["j"]]
            w.writerow([a.name, a.split, config.CLASS_NAMES[a.label],
                        b.name, b.split, config.CLASS_NAMES[b.label],
                        c["kind"], c["score"], c["transform"]])

    # --- transform breakdown (augmentation evidence) -----------------------
    transforms = collections.Counter(c["transform"] for c in confirmed)

    print(f"    duplicate groups (size>1): {len(multi)}")
    print(f"    groups crossing train/test: {cross_tr_te}")
    print(f"    groups crossing train/val : {cross_tr_val}")
    print(f"    groups crossing val/test  : {cross_val_te}")
    print(f"    groups with LABEL CONFLICT: {label_conflicts}")
    print(f"    TEST images with a train near-duplicate: "
          f"{len(test_with_train_dup)}/{len(test)} ({pct_test:.2f}%)")
    print(f"    VAL  images with a train near-duplicate: "
          f"{len(val_with_train_dup)}/{len(val)} ({pct_val:.2f}%)")

    n_examples = render_examples(records, groups, confirmed, args.examples)
    print(f"\n    wrote {n_examples} visual example(s) -> {EX_DIR}")

    report = {
        "dataset_size": len(records),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "exact_duplicate_groups": len(exact_groups),
        "exact_duplicate_redundant_files": n_exact_extra,
        "candidate_pairs_screened": len(pairs),
        "confirmed_pairs_total": len(confirmed),
        "confirmed_pairs_exact": n_exact_pairs,
        "confirmed_pairs_near": n_near_pairs,
        "candidate_pairs_rejected": len(pairs) - len(confirmed),
        "duplicate_groups": len(multi),
        "largest_group_size": max((len(v) for v in multi.values()), default=0),
        "groups_crossing_train_test": cross_tr_te,
        "groups_crossing_train_val": cross_tr_val,
        "groups_crossing_val_test": cross_val_te,
        "groups_with_label_conflict": label_conflicts,
        "test_images_with_train_duplicate": len(test_with_train_dup),
        "test_percent_affected": round(pct_test, 3),
        "val_images_with_train_duplicate": len(val_with_train_dup),
        "val_percent_affected": round(pct_val, 3),
        "transform_breakdown": dict(transforms),
        "thresholds": {
            "hamming_max": 8, "corr_accept": 0.93, "corr_strong": CORR_STRONG,
            "note": "pairs are only confirmed by pixel correlation, never by hash alone",
        },
    }

    if not args.no_split:
        split_records, assignment = make_group_split(records, groups)
        write_group_split(split_records, assignment, groups, records)
        moved = sum(1 for r in records
                    if assignment[next(g for g, i in groups.items() if r.index in i)]
                    != r.split) if False else None   # expensive; computed below
        # cheaper: map index -> group once
        gid_of = {}
        for gid, idxs in groups.items():
            for i in idxs:
                gid_of[i] = gid
        moved = sum(1 for r in records if assignment[gid_of[r.index]] != r.split)
        report["group_aware_split"] = {
            "counts": {sp: len(v) for sp, v in split_records.items()},
            "per_class": {
                sp: {config.CLASS_NAMES[k]: v for k, v in
                     sorted(collections.Counter(r.label for r in recs).items())}
                for sp, recs in split_records.items()
            },
            "images_that_change_split": moved,
            "percent_images_moved": round(moved / len(records) * 100, 2),
            "written_to": str(GROUPED_SPLIT_DIR),
        }
        print(f"\n    group-aware split -> {GROUPED_SPLIT_DIR}")
        for sp, recs in split_records.items():
            c = collections.Counter(r.label for r in recs)
            print(f"      {sp:<6} {len(recs):<6} " +
                  "  ".join(f"{config.CLASS_NAMES[k]}={c[k]}" for k in sorted(c)))
        print(f"      images changing split: {moved} "
              f"({moved / len(records) * 100:.2f}%)")

    (OUT_DIR / "audit_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nreport -> {OUT_DIR / 'audit_report.json'}")
    print(f"groups -> {OUT_DIR / 'duplicate_groups.csv'}")
    print(f"pairs  -> {OUT_DIR / 'confirmed_pairs.csv'}")


if __name__ == "__main__":
    main()
