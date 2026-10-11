# RadFrac — AI-Assisted Fracture Screening & Triage

**Target deployment context:** rural / under-resourced primary healthcare in India.

> ⚠️ **RadFrac is a screening and triage assistant, not a diagnostic authority
> and not a medical device.** Every output requires sign-off by a qualified
> clinician. A negative screen does not exclude fracture.

---

## 1. Problem statement

India has roughly **20,000–22,000 practising radiologists for ~1.4 billion
people** (about 1 per 100,000), well below WHO-adequate levels. Around **70–80%
of radiologists work in Tier-1 metros**, while rural areas — home to about
**65% of the population** — are served by only **27% of the country's doctors**.

This is **not primarily an equipment problem.** Many rural Primary Health Centres
(PHCs) and Community Health Centres (CHCs) already have X-ray machines. It is a
**specialist interpretation gap**: the clinician on site is typically a general
physician, not a radiologist. When a film is taken, they must either judge it
without radiology-specific training, or send it to a city and wait — sometimes
days — delaying treatment.

### Core positioning

RadFrac **does not compete with a radiologist.** It fills the gap that currently
has **no expert input at all**. The doctor remains the final authority in every
case.

The differentiator is not "AI vs human" but that AI does a **categorically
different kind of work**:

| RadFrac can | A human cannot (practically) |
|---|---|
| Quantify its own uncertainty numerically (MC-Dropout sampling) | Express reliability as a calibrated number |
| Stay perfectly consistent regardless of fatigue or caseload | Guarantee fatigue-independent consistency |
| Provide instant structured screening where no specialist exists | Be physically present at every rural PHC |

---

## 2. Target customers

**Primary — rural PHC/CHC general physicians & nurses** with an X-ray machine but
no on-site radiologist. They need a second-opinion screen to decide whether to
refer, at the point of care.

**Secondary — emergency / busy radiology departments**, where radiologists review
every case but the queue is in arrival order rather than urgency order. RadFrac
**re-sorts the worklist**; it does not replace sequential human review and makes
no claim to "comprehend 500 X-rays at once".

**Tertiary — radiology students / junior residents**, using the explainability
output to compare their own reads against model attention.

**Explicitly out of scope:** replacing a radiologist in a well-staffed hospital;
legal/insurance diagnostic use; direct-to-patient self-diagnosis.

### How this shaped the build

| Requirement | Consequence in the code |
|---|---|
| Offline / low-connectivity is a must | No cloud calls at inference; stdlib SQLite; models load from local disk |
| Non-specialist user (possibly a nurse) | UI answers *verdict → trust → what to hand over*, not a data dashboard |
| The report **is** the referral document | `src/referral_report.py` is a first-class deliverable, not an export afterthought |
| Modest, non-GPU hardware | MobileNetV2 backbone + knowledge distillation to a **0.03 MB** student |
| Single-patient real-time is the primary flow | `/predict` is the main path; batch/queue mode is secondary |

---

## 3. Architecture

```
                      ┌─────────────────────────────────────────┐
  Streamlit UI  ─────►│                                         │
  (app/demo_app.py)   │   SERVICE LAYER  (src/service.py)       │
                      │                                         │
  FastAPI       ─────►│   validity gate → predict → Grad-CAM    │──► SQLite
  (api/main.py)       │   → MC-dropout → cross-check → note     │   (src/db.py)
                      │   → persist                             │
                      └──────────────────┬──────────────────────┘
                                         │
        ┌────────────────┬───────────────┼────────────────┬──────────────┐
        ▼                ▼               ▼                ▼              ▼
   model_cnn.py   model_transfer.py  model_svm.py   uncertainty.py  gradcam.py
                                                    validity.py     explain_dual.py
                                                    disagreement.py screening_note.py
```

Validation and orchestration live in the **service layer**, so the UI, the API,
and any future frontend share identical behaviour (no duplicated rules).

| Responsibility | File |
|---|---|
| Env-driven config (paths, thresholds, costs) | `src/config.py` |
| Data loading, stratified splits, tf.data | `src/preprocessing.py` |
| Model 1 — custom CNN | `src/model_cnn.py` |
| Model 2 — MobileNetV2 transfer | `src/model_transfer.py` |
| Model 3 — HOG+Canny → SVM/RF | `src/features_classical.py`, `src/model_svm.py` |
| MC-Dropout uncertainty (4.1) | `src/uncertainty.py` |
| Self-supervised contrastive pretraining (4.2) | `src/ssl_pretrain.py` |
| Knowledge distillation (4.3) | `src/distillation.py` |
| Cost-sensitive thresholds + calibration (4.4) | `src/threshold_analysis.py` |
| Dual explainability + agreement (4.5) | `src/explain_dual.py`, `src/gradcam.py` |
| Referral PDF (4.6) | `src/referral_report.py` |
| Model-disagreement flagging (4.7) | `src/disagreement.py` |
| Input-validity gate (4.8) | `src/validity.py` |
| NL screening note (4.9) | `src/screening_note.py` |
| Service layer (6.1) | `src/service.py` |
| HTTP API (6.2) | `api/main.py` |
| Persistence (6.3) | `src/db.py` |
| Synthetic-data guard | `src/dataset_status.py` |

---

## 4. Setup

```bash
cd radfrac
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on Unix
pip install -r requirements.txt
```

Quieter/faster TensorFlow start (optional):
`$env:TF_CPP_MIN_LOG_LEVEL="3"; $env:TF_ENABLE_ONEDNN_OPTS="0"`

---

## 5. Getting the real dataset  ← **action needed from you**

**RadFrac currently ships synthetic placeholder images only.** They exist to
verify the pipeline runs and are *not* valid for any clinical claim. Every
surface (CLI, UI, PDF) hard-warns while synthetic data is active.

### Recommended primary dataset

[Bone Fracture Multi-Region X-ray Data](https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data)
— ~9,246 train / 828 val / 506 test, pre-split into fractured / not-fractured
folders across several anatomical regions.

```bash
# 1. kaggle.com → Settings → API → "Create New API Token"
#    save kaggle.json to  C:\Users\<you>\.kaggle\kaggle.json
# 2. download + normalise into data/raw/{fractured,not_fractured}/
python -m scripts.fetch_kaggle_dataset
# 3. rebuild everything on real data
python -m src.preprocessing
python -m src.train --model all
python -m src.evaluate
python -m scripts.threshold_analysis
```

`fetch_kaggle_dataset.py` tolerates the varied folder spellings and
train/val/test nesting these dataset versions use, flattens them, and removes
the synthetic marker so the warnings clear automatically.

**Alternatives**
- [FracAtlas](https://www.kaggle.com/datasets/tommyngx/fracatlas) — ships a
  filename→label CSV (has bounding boxes, useful for a future detection
  extension). Use `scripts/ingest_real_dataset.py --csv ... --image-dir ...`.
- [Mendeley X-ray Bone Fracture Dataset](https://data.mendeley.com/datasets/8d9kn57pdj/1)
  (DOI [10.17632/8d9kn57pdj.1](https://doi.org/10.17632/8d9kn57pdj.1)) —
  hospital-sourced, simple/comminuted categories.

**Cited for context, not usable for this task**
- [IN-CXR](https://nirt.res.in/html/xray.html) — open Indian chest X-ray data
  from the 2019–21 National TB Prevalence Survey. Indian-population imaging data
  exists, but for TB screening, not fractures.
- **ICMR–IISc / ARTPARK** — an active national effort to build gold-standard
  Indian-population medical imaging datasets, explicitly because most AI tools
  deployed in India today are trained on non-Indian data.

Fracture *appearance* is not population-specific, so non-India-specific training
images are scientifically acceptable here; the India-specific angle belongs to
the **deployment context and customer framing**, not the training pixels.

---

## 6. Running it

### Point-of-care UI (primary flow)

```bash
streamlit run app/demo_app.py
```

Run it from the repository root so `.streamlit/config.toml` (light base theme) is
picked up. The landing screen uses a photographic backdrop loaded from
`app/assets/home_bg.jpg`; drop the image in there and it is inlined automatically.
Without it the theme falls back to a CSS gradient. See `app/assets/README.md`.

### HTTP API

```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
# interactive docs at http://127.0.0.1:8000/docs
```

| Method & path | Purpose |
|---|---|
| `GET /health` | liveness + which models loaded + synthetic flag |
| `GET /models` | models, thresholds, test metrics |
| `POST /predict` | one image → prediction, uncertainty, disagreement, note |
| `POST /predict/batch` | several images + urgency-ordered `triage_order` |
| `GET /predict/{id}` | stored record |
| `GET /predict/{id}/gradcam` | attention heatmap PNG |
| `GET /predict/{id}/report` | referral PDF |
| `GET /predictions` | recent history |
| `GET /triage` | worklist re-sorted by urgency |
| `POST /cases`, `GET /cases/{id}` | group several X-rays per encounter |

```bash
curl -X POST http://127.0.0.1:8000/predict \
     -F "file=@xray.png" -F "model=cnn"
```

> **Security:** the API ships with **no authentication**. That is acceptable for
> localhost / single-machine clinic use, but it must sit behind an auth layer and
> TLS before being exposed to any network.

### Full pipeline from scratch

```bash
python -m scripts.make_synthetic_data --n_per_class 150 --clean  # or fetch real data
python -m src.preprocessing
python -m src.train --model all
python -m src.evaluate
python -m scripts.threshold_analysis          # cost-optimal thresholds + calibration
python -m scripts.generate_gradcam_samples --n 3
python -m scripts.explain_agreement --n 3     # Grad-CAM vs occlusion agreement
python -m scripts.ssl_pretrain --label-fraction 0.25 --baseline
python -m scripts.distill --teacher transfer
python -m scripts.generate_report
```

### Configuration (6.5)

Everything is environment-overridable, so the same code runs on a laptop, a
server, or a clinic machine:

| Variable | Default | Meaning |
|---|---|---|
| `RADFRAC_DATA_ROOT` | `../data` | dataset root |
| `RADFRAC_OUTPUT_ROOT` | `./outputs` | models, metrics, figures |
| `RADFRAC_DB_PATH` | `outputs/radfrac.db` | SQLite file |
| `RADFRAC_CASE_STORE` | `outputs/cases` | per-case images/PDFs |
| `RADFRAC_DEFAULT_MODEL` | `cnn` | model used when unspecified |
| `RADFRAC_MC_PASSES` | `40` | MC-Dropout passes |
| `RADFRAC_UNCERTAINTY_STD` | `0.15` | sd above which we flag uncertainty |
| `RADFRAC_FN_COST` / `RADFRAC_FP_COST` | `5.0` / `1.0` | clinical cost weights |
| `RADFRAC_DISAGREEMENT_SPREAD` | `0.30` | probability spread that flags disagreement |

---

## 7. Methodology

**Preprocessing.** RGB, resized to 128×128, scaled to [0,1]. The identical code
path (`load_image_array` → `preprocess_for_dl`) runs at training *and* inference,
so there is no train/serve skew.

**Splitting.** One stratified split into train/val/test. The **test split is
never** used for model selection, hyperparameter search, or threshold tuning.
All three models share identical partitions.

**Operating points.** Thresholds are chosen on **validation** data only, in
preference order: cost-optimal → F1-optimal → 0.5. The active source is surfaced
in the UI and the PDF.

**Why cost-sensitive?** At a screening step a **false negative (missed fracture)
is far costlier than a false positive** — a false alarm costs one radiologist
review, a miss can cost a mismanaged injury. Maximising accuracy or F1 assumes
symmetric costs, which is wrong here. We minimise
`FN×5 + FP×1` (weights configurable) and also report the threshold achieving a
target recall.

---

## 8. Results

All figures below are **real measured outputs** on the **synthetic** dataset
(150 images/class → train 180 / val 60 / test 60). They demonstrate that the
pipeline is correct. **They are not clinical performance figures.**

### 8.1 Three-model comparison (test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|:--:|:--:|:--:|:--:|:--:|
| Custom CNN | 0.967 | — | — | 0.968 | 0.997 |
| Transfer (MobileNetV2) | 0.950 | — | — | 0.952 | **1.000** |
| Classical (HOG+Canny+SVM) | **1.000** | 1.000 | 1.000 | **1.000** | **1.000** |

### 8.2 Cost-sensitive thresholds (chosen on val, reported on test)

| Model | Cost-optimal t | Test FN | Test FP | Recall | Precision |
|---|:--:|:--:|:--:|:--:|:--:|
| CNN | 0.030 | 2 | 1 | 0.933 | 0.966 |
| Transfer | 0.400 | **0** | 4 | **1.000** | 0.882 |
| Classical | 0.005 | **0** | 13 | **1.000** | 0.698 |

The cost weighting behaves exactly as intended: it pushes thresholds down to
eliminate missed fractures, accepting more false alarms in exchange.

### 8.3 Calibration (4.4) — an important negative result

| Model | ECE | Brier | Verdict |
|---|:--:|:--:|---|
| CNN | 0.468 | 0.457 | **poorly calibrated** |
| Transfer | 0.274 | 0.093 | **poorly calibrated** |
| Classical | **0.032** | **0.008** | well calibrated |

**This matters clinically.** The two neural models rank cases well (AUC ≈ 1.0)
but their raw sigmoid outputs are *not* probabilities — the CNN's operating
threshold is 0.03, so a "4%" score is actually a positive screen. The classical
SVM is the only model whose confidence can honestly be read as a probability.
The UI therefore always shows the threshold alongside the score, and the report
records both. Presenting the CNN's raw number to a clinician as "confidence"
would be misleading, and the calibration analysis is what exposed that.

### 8.4 Self-supervised pretraining (4.2) — low-label regime

SimCLR contrastive pretraining on the unlabelled training split, then fine-tuned
on **25% of the labels**, versus the identical architecture trained from scratch
on the same 25%:

| Setup | Test F1 |
|---|:--:|
| SimCLR pretrained → fine-tuned | **0.983** |
| Same architecture, from scratch | 0.675 |

NT-Xent loss fell 4.11 → ~3.4 over 25 epochs. This is the intended result:
self-supervision buys the most when labels are scarce, which is precisely the
situation for real fracture datasets.

### 8.5 Knowledge distillation (4.3) — deployment footprint

| Model | Params | Size | Test F1 | ROC-AUC |
|---|--:|--:|:--:|:--:|
| Teacher (MobileNetV2) | 2,259,265 | 8.62 MB | 0.952 | 1.000 |
| **Student (distilled)** | **7,289** | **0.03 MB** | **0.877** | 0.939 |
| Student (hard labels only) | 7,289 | 0.03 MB | 0.723 | 0.746 |

**310× fewer parameters**, and distillation clearly beats training the same small
network on hard labels alone (F1 0.877 vs 0.723). Measured single-image CPU
latency did *not* improve much (≈91 ms vs ≈98 ms) because per-call framework
overhead dominates at this input size — an honest caveat: the win here is
**memory/storage footprint**, not raw speed.

### 8.6 Dual explainability agreement (4.5)

Grad-CAM (gradient-based) vs occlusion sensitivity (black-box) on the same images:

| | Value |
|---|---|
| Conclusive cases | 3 / 6 |
| Agreement on conclusive cases | **3 / 3 (100%)** |
| Mean Pearson r | 0.772 |
| Mean top-region IoU | 0.588 |

The 3 inconclusive cases were confident negatives, where occlusion has no score
to reduce and both maps are flat. Those are reported as **inconclusive rather
than agreeing** — counting them would have inflated the rate to a meaningless
100%/6.

---

## 9. Safety features built into the application

| Feature | Behaviour |
|---|---|
| **Input-validity gate (4.8)** | Rejects all-black/all-white/no-contrast images with reasons (HTTP 422); warns on colour photos, odd aspect ratios, low resolution |
| **Grad-CAM coherence check** | Flags attention that is empty or floods the frame → manual review |
| **Model-disagreement flag (4.7)** | Flags a verdict split *or* agreement with >0.30 probability spread |
| **Uncertainty flag (4.1)** | Flags MC-Dropout sd ≥ 0.15 as genuinely unreliable |
| **Synthetic-data guard** | Hard banner in CLI, UI and PDF while synthetic data is active |
| **Never a bare verdict** | Every note ends with a clinician-sign-off requirement; negatives advise referral if suspicion persists |

---

## 10. Testing

```bash
python -m pytest            # 72 tests
```

Covers: splitting/leakage, preprocessing consistency, classical features,
model build/save/load, Grad-CAM, synthetic detection, DB round-trips and triage
ordering, validity gating, MC-Dropout (including that BatchNorm stays in
inference mode), disagreement logic, note generation, cost sweeps, calibration
detection, threshold-source precedence, occlusion attribution, distillation loss,
the full service layer, and every API endpoint including its failure modes.

---

## 11. Known limitations

- **Training data is not India-specific.** A genuinely deployable
  India-targeted version needs locally-sourced data, which does not yet exist as
  an open dataset. This is a recognised national gap — see the **ICMR–IISc /
  ARTPARK** initiative — not an oversight in this project's scope.
- **Results above are from synthetic data.** Real radiographs are required
  before any performance claim. The synthetic task is trivially separable, which
  is why scores are near-perfect.
- **Binary classifier only** — no localisation, severity grading, or fracture
  typing. FracAtlas bounding boxes are the natural extension (detection).
- **Small labelled sets** in public fracture datasets limit generalisation
  across bone types, machines and populations.
- **Neural models are poorly calibrated** (§8.3) — their scores are ranking
  signals, not probabilities.
- **Distillation reduces size, not latency** at this input size (§8.5).
- **No authentication on the API** (§6).
- **RadFrac is not a diagnostic authority.** Screening/triage support only,
  always requiring human sign-off.

## 12. Descoped / future work

Considered and deliberately not built: interactive occlusion tool (demo value,
low priority — though the occlusion *engine* exists in `src/explain_dual.py`);
multimodal fusion with patient context (age, injury mechanism); multi-language
report output (Tamil/Hindi), which would directly reinforce the rural-access
motivation; fracture localisation via bounding boxes; prospective clinical
validation.

---

## 13. Advanced models extension (19MAM56 CAT II)

Three gradient-boosting models were added **on top of** the existing system.
The SVM, MobileNetV2, their ensemble, thresholds and the headline verdict are
unchanged; the boosters run alongside and appear in the UI's analysis step and
in their own result panel.

| Model | Key | Notes |
|---|---|---|
| XGBoost | `xgboost` | level-wise histogram boosting |
| LightGBM | `lightgbm` | leaf-wise histogram boosting |
| CatBoost | `catboost` | ordered boosting, symmetric trees |
| Boosting committee | `boost_committee` | mean of the three boosters' probabilities |

### Pipeline

```
image ─► centre-crop 72% ─► 128×128 ─┬─► HOG (1,764) + Canny grid (16) ─┐
                                     └─► MobileNetV2 GAP embedding (1,280)┴─► 3,060-d hybrid vector
     ─► StandardScaler ─► PCA(k) ─► XGBoost | LightGBM | CatBoost ─► P(fractured) ─► validated threshold
```

* **Feature engineering / dimensionality reduction**: hybrid hand-crafted + deep
  features, then PCA. `k ∈ {64,128,256}` is a tuned hyperparameter. t-SNE and UMAP
  projections are in the figures.
* **Hyperparameter optimisation**: Optuna TPE (Bayesian) on validation ROC-AUC
  with early stopping on validation; search spaces in
  [`configs/boosting.yaml`](configs/boosting.yaml).
* **Experiment tracking**: every trial is an MLflow nested run
  (`outputs/mlruns`) and is mirrored to `outputs/experiments/trials_<model>.csv`.
  View it with `pip install mlflow` then `mlflow ui --backend-store-uri outputs/mlruns`.
* **Operating point**: the same clinical rule as the existing models, applied to
  validation only (FN×5 + FP×1, FPR ≤ 0.25). The test split is scored once.
* **Benchmarking**: [`docs/BENCHMARK.md`](docs/BENCHMARK.md) compares against the
  published DenseNet201 baseline (Aldhyani et al., *Front. Med.* 2025), our
  leaky-split reproduction and the leakage-free results, with bootstrap CIs,
  paired ΔAUC tests, McNemar tests and error analysis.

| File | Role |
|---|---|
| `src/features_hybrid.py` | hybrid features, path remapping, per-split cache |
| `src/model_boosting.py` | pipeline builders, persistence, `BoostingSuite` for serving |
| `scripts/train_boosting.py` | Optuna + MLflow tuning, thresholds, registry |
| `scripts/advanced_analysis.py` | 14 figures, `benchmark.json`, `docs/BENCHMARK.md` |
| `configs/boosting.yaml` | features, search spaces, tracking, operating point |
| `tests/test_boosting.py` | unit tests for the extension |

### Reproduce (inside Docker; TensorFlow is needed for the embeddings)

```bash
docker build -t radfrac:latest .
docker run --rm -v "$PWD/../data:/data:ro" -v "$PWD/outputs:/app/outputs" -u root \
    radfrac:latest python -m scripts.train_boosting
docker run --rm -v "$PWD/../data:/data:ro" -v "$PWD/outputs:/app/outputs" -u root \
    radfrac:latest python -m scripts.advanced_analysis
docker build -t radfrac:latest .        # rebuild so the image contains the boosters
```

Figures land in `outputs/figures/advanced/`, the registry (best params,
thresholds, metrics, learning curves) in `outputs/models/boosting/registry.json`.

### API

`POST /predict` accepts `model=xgboost|lightgbm|catboost|boost_committee`. Every
response carries an `advanced` block with all booster scores and their agreement,
whichever model is primary. `/health` lists `advanced_models`; `/models` lists
their test metrics.

### Docker & cloud

```bash
docker compose up -d --build      # UI http://localhost:8501, API http://localhost:8000/docs
```

AWS (ECR + EC2) steps: [`deploy/aws/README.md`](deploy/aws/README.md).
