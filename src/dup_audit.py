"""Duplicate / near-duplicate leakage audit.

Why this matters: RadFrac's stratified split shuffles *files*. If the dataset
contains augmented or re-encoded copies of the same underlying radiograph, copies
of one film can land in train AND test, and the model is effectively tested on
images it memorised. Every headline metric would then be optimistic.

Detection is layered, cheapest and most certain first:

1. **SHA-256 of the file bytes** - byte-identical files. Zero false positives.
2. **Perceptual hashes** (dHash + pHash, 64-bit each) - survive re-encoding,
   resizing and mild brightness/contrast change. Used only to generate
   *candidates*, never as the final verdict.
3. **Pixel verification** on every candidate pair - normalised cross-correlation
   over the 8 dihedral transforms (4 rotations x optional flip) plus a small
   rotation sweep. A pair is only called a duplicate if correlation clears a
   high bar, which is what stops "two similar-looking wrists" from being counted
   as the same film.

Groups are then formed by union-find over confirmed pairs, so a chain of
transformed copies collapses into one group representing one underlying film.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config

# --------------------------------------------------------------------------- #
# Tunables
# --------------------------------------------------------------------------- #
HASH_SIDE = 8                 # dHash/pHash grid -> 64 bits
THUMB = 64                    # verification thumbnail size
HAMMING_MAX = 8               # candidate cut-off on 64-bit hashes
CORR_ACCEPT = 0.93            # normalised-correlation bar for "same film"
CORR_STRONG = 0.985           # near-certain (used to label examples)
SMALL_ROTATIONS = (-8, -4, 4, 8)


@dataclass
class ImageRecord:
    index: int
    filepath: str
    label: int
    split: str
    sha256: str = ""
    dhash: np.uint64 = np.uint64(0)
    phash: np.uint64 = np.uint64(0)
    thumb: np.ndarray | None = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return Path(self.filepath).name


# --------------------------------------------------------------------------- #
# Hashing
# --------------------------------------------------------------------------- #
def _bits_to_uint64(bits: np.ndarray) -> np.uint64:
    packed = np.uint64(0)
    for b in bits.ravel().astype(bool):
        packed = np.uint64(packed << np.uint64(1)) | np.uint64(1 if b else 0)
    return packed


def dhash(gray: np.ndarray, side: int = HASH_SIDE) -> np.uint64:
    """Difference hash: compares each pixel with its right neighbour."""
    import cv2

    small = cv2.resize(gray, (side + 1, side), interpolation=cv2.INTER_AREA)
    return _bits_to_uint64(small[:, 1:] > small[:, :-1])


def phash(gray: np.ndarray, side: int = HASH_SIDE) -> np.uint64:
    """DCT-based perceptual hash (robust to scaling / mild tone changes)."""
    import cv2

    big = cv2.resize(gray, (side * 4, side * 4), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(big.astype("float32"))
    block = dct[:side, :side].copy()
    block[0, 0] = 0.0                      # drop DC (overall brightness)
    return _bits_to_uint64(block > np.median(block))


def _dihedral(gray: np.ndarray):
    """The 8 dihedral variants of an image."""
    out = []
    for flip in (False, True):
        g = np.fliplr(gray) if flip else gray
        for k in range(4):
            out.append(np.rot90(g, k))
    return out


def canonical_hashes(gray: np.ndarray) -> tuple[np.uint64, np.uint64]:
    """Transform-invariant hashes: the minimum over all dihedral variants.

    Taking the min makes a 90/180/270-degree rotation or a mirror of the same
    film collapse to the SAME hash, so flipped/rotated copies become exact hash
    matches rather than needing a fuzzy search.
    """
    ds, ps = [], []
    for v in _dihedral(gray):
        v = np.ascontiguousarray(v)
        ds.append(int(dhash(v)))
        ps.append(int(phash(v)))
    return np.uint64(min(ds)), np.uint64(min(ps))


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_records(splits: dict[str, list]) -> list[ImageRecord]:
    """Hash every image once, recording split membership."""
    import cv2
    from PIL import Image

    records: list[ImageRecord] = []
    idx = 0
    for split_name, samples in splits.items():
        for s in samples:
            records.append(ImageRecord(index=idx, filepath=s.filepath,
                                       label=s.label, split=split_name))
            idx += 1

    for i, rec in enumerate(records):
        raw = Path(rec.filepath).read_bytes()
        rec.sha256 = hashlib.sha256(raw).hexdigest()
        with Image.open(rec.filepath) as im:
            gray = np.asarray(im.convert("L"), dtype="uint8")
        rec.thumb = cv2.resize(gray, (THUMB, THUMB), interpolation=cv2.INTER_AREA)
        rec.dhash, rec.phash = canonical_hashes(rec.thumb)
        if (i + 1) % 2000 == 0:
            print(f"    hashed {i + 1}/{len(records)}", flush=True)
    return records


# --------------------------------------------------------------------------- #
# Candidate search
# --------------------------------------------------------------------------- #
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _hamming_matrix_chunk(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamming distance between two uint64 arrays -> (len(a), len(b)) uint8."""
    x = np.bitwise_xor(a[:, None], b[None, :]).astype(">u8")
    bytes_view = x.view(np.uint8).reshape(*x.shape, 8)
    return _POPCOUNT[bytes_view].sum(axis=-1).astype(np.uint8)


def candidate_pairs(records: list[ImageRecord], hamming_max: int = HAMMING_MAX,
                    chunk: int = 512) -> set[tuple[int, int]]:
    """Pairs whose dHash OR pHash are within ``hamming_max`` bits."""
    d = np.array([int(r.dhash) for r in records], dtype=np.uint64)
    p = np.array([int(r.phash) for r in records], dtype=np.uint64)
    n = len(records)
    pairs: set[tuple[int, int]] = set()

    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        dd = _hamming_matrix_chunk(d[start:stop], d)
        pp = _hamming_matrix_chunk(p[start:stop], p)
        close = (dd <= hamming_max) | (pp <= hamming_max)
        rows, cols = np.nonzero(close)
        for r, c in zip(rows, cols):
            i, j = start + int(r), int(c)
            if i < j:
                pairs.add((i, j))
        if (stop // chunk) % 4 == 0:
            print(f"    candidate scan {stop}/{n}", flush=True)
    return pairs


# --------------------------------------------------------------------------- #
# Pixel verification
# --------------------------------------------------------------------------- #
def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation in [-1, 1]; brightness/contrast invariant."""
    af = a.astype("float32").ravel()
    bf = b.astype("float32").ravel()
    af -= af.mean()
    bf -= bf.mean()
    na, nb = np.linalg.norm(af), np.linalg.norm(bf)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(np.dot(af, bf) / (na * nb))


CROP_SCALES = (0.95, 0.90, 0.85, 0.80)
CORR_ACCEPT_CROP = 0.95       # stricter bar for crop matches (see note below)


def _centre_crop_resize(img: np.ndarray, scale: float) -> np.ndarray:
    import cv2

    h, w = img.shape
    ch, cw = int(round(h * scale)), int(round(w * scale))
    top, left = (h - ch) // 2, (w - cw) // 2
    return cv2.resize(img[top:top + ch, left:left + cw], (w, h),
                      interpolation=cv2.INTER_LINEAR)


def best_correlation(a: np.ndarray, b: np.ndarray,
                     try_small_rotations: bool = True,
                     try_crops: bool = True) -> tuple[float, str]:
    """Best NCC of ``a`` against all plausible transforms of ``b``.

    Returns (score, transform description). This is the step that separates a
    genuine transformed copy from merely similar anatomy.

    Crop trials use a STRICTER acceptance bar. Cropping and rescaling discards
    context and can inflate correlation between two merely-similar films, so a
    crop match must clear ``CORR_ACCEPT_CROP`` rather than ``CORR_ACCEPT``. The
    returned description is prefixed "crop" so the caller can apply that bar.
    """
    import cv2

    best, how = -1.0, "none"
    names = []
    for flip in (False, True):
        for k in range(4):
            names.append(("flip+" if flip else "") + f"rot{k * 90}")
    for variant, nm in zip(_dihedral(b), names):
        s = _ncc(a, np.ascontiguousarray(variant))
        if s > best:
            best, how = s, nm

    if try_small_rotations and best < CORR_STRONG:
        h, w = b.shape
        centre = (w / 2.0, h / 2.0)
        for ang in SMALL_ROTATIONS:
            m = cv2.getRotationMatrix2D(centre, ang, 1.0)
            rot = cv2.warpAffine(b, m, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
            s = _ncc(a, rot)
            if s > best:
                best, how = s, f"rot{ang}deg"

    # Crop/zoom variants, in both directions (a may be the crop of b or vice versa).
    if try_crops and best < CORR_STRONG:
        for scale in CROP_SCALES:
            s = _ncc(a, _centre_crop_resize(b, scale))
            if s > best:
                best, how = s, f"crop{int(scale * 100)}(b)"
            s = _ncc(_centre_crop_resize(a, scale), b)
            if s > best:
                best, how = s, f"crop{int(scale * 100)}(a)"
    return best, how


def verify_pairs(records: list[ImageRecord], pairs: set[tuple[int, int]],
                 accept: float = CORR_ACCEPT) -> list[dict]:
    """Confirm or reject each candidate pair by pixel correlation."""
    confirmed: list[dict] = []
    total = len(pairs)
    for n, (i, j) in enumerate(sorted(pairs), 1):
        a, b = records[i], records[j]
        if a.sha256 == b.sha256:
            confirmed.append({"i": i, "j": j, "score": 1.0, "transform": "identical",
                              "kind": "exact"})
        else:
            score, how = best_correlation(a.thumb, b.thumb)
            # Crop matches must clear a stricter bar (see best_correlation).
            bar = CORR_ACCEPT_CROP if how.startswith("crop") else accept
            if score >= bar:
                confirmed.append({"i": i, "j": j, "score": round(score, 4),
                                  "transform": how, "kind": "near"})
        if n % 5000 == 0:
            print(f"    verified {n}/{total}", flush=True)
    return confirmed


# --------------------------------------------------------------------------- #
# Grouping
# --------------------------------------------------------------------------- #
class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_groups(records: list[ImageRecord], confirmed: list[dict]) -> dict[int, list[int]]:
    """Union-find over confirmed pairs -> {group_id: [record indices]}."""
    uf = UnionFind(len(records))
    for pair in confirmed:
        uf.union(pair["i"], pair["j"])
    groups: dict[int, list[int]] = {}
    for r in records:
        groups.setdefault(uf.find(r.index), []).append(r.index)
    return groups
