# PathSnap — GPS Map Matching & Route Reconstruction

PathSnap reconstructs the roads a vehicle likely travelled on from noisy GPS observations. It combines a LightGBM classifier with a Hidden Markov Model (HMM) and Viterbi decoding, then displays the results in an interactive map dashboard.

Built as a machine-learning hackathon prototype, it includes data preparation, model comparison, route decoding, a FastAPI service, and a Leaflet frontend.

## How it works

```text
GPS track + trip road network
            |
            v
Find candidate road segments within 50 metres
            |
            v
Compute 42 candidate features
            |
            v
LightGBM scores each candidate
            |
            +----> ML-only: best candidate per GPS point
            |
            v
HMM + Viterbi: select a plausible sequence of road segments
            |
            v
Route geometry + evaluation metrics + interactive map
```

The classifier estimates which road segment fits each GPS observation. The sequence decoder also considers road-network distance, GPS displacement, speed, and direction to reduce implausible jumps between roads.

The reference route is used for training labels and evaluation/display. It is excluded from inference features and decoding inputs.

## Implemented features

- Dataset loading, coordinate projection, audits, and exploratory maps.
- Spatial candidate search and 42-feature engineering pipeline.
- Training and comparison of Logistic Regression, Random Forest, Extra Trees, LightGBM, and MLP models.
- Optuna hyperparameter search using training trips only.
- HMM/Viterbi route decoding and comparison with independent per-point predictions.
- Single-trip inference through a Python CLI and FastAPI.
- Leaflet dashboard with route layers, accuracy metrics, and inference timing.
- Unit and integration test suites.

## Reported results

Saved evaluation results cover **10 student-test trips and 19,133 labelled GPS points**, using the deployed LightGBM model.

| Metric | LightGBM only | LightGBM + HMM/Viterbi |
|---|---:|---:|
| Top-1 point accuracy | 89.52% | **94.15%** |
| Direct continuity | 95.80% | **98.31%** |
| Continuity within 3 arcs | 98.53% | **99.92%** |
| Mean route arc precision | 91.01% | **97.56%** |
| Mean route arc recall | 89.68% | **90.77%** |
| Mean route length F1 | 94.76% | **98.44%** |

HMM decoding improves overall point accuracy by **4.63 percentage points**. These are saved experimental results, not a guarantee for new GPS datasets.

- **Point accuracy:** proportion of evaluated GPS points assigned to the reference-labelled road segment.
- **Continuity:** how consistently consecutive predictions follow connected roads.
- **Route arc precision/recall:** overlap between predicted and reference road-segment sets.
- **Route length F1:** length-weighted overlap; it does not measure exact route ordering.

Point accuracy is aggregated across labelled points. Route metrics shown above are means across trips.

Sources: [HMM evaluation](results/metrics/hmm_evaluation.json) and [model comparison](results/metrics/model_comparison.json).

### Model selection and tuning

LightGBM was selected using five-fold cross-validation grouped by trip on **80 training trips**, achieving **88.85% CV Top-1 accuracy**. Student-test trips were not used for model selection.

A separate 12-trial Optuna experiment used three-fold grouped cross-validation on a capped training subset. Its best score was **89.00%**, compared with **88.76%** for the baseline under that tuning protocol. The tuned configuration **has not replaced the deployed model**; full retraining and grouped-CV confirmation remain necessary. These tuning scores use a different protocol from the five-fold model comparison.

See [Optuna results](results/metrics/optuna_lightgbm.json).

## Quick start

Run the following commands in Windows PowerShell.

### 1. Clone and install

The repository records Python 3.12 as the version used for its results.

```powershell
git clone https://github.com/raviteja2028tech/ml-hack.git
cd ml-hack
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Add the dataset

Extract the official `Student folder.zip` so the directory structure is:

```text
data/raw/Student folder/
├── train/
│   └── <trip_id>/
└── student_test/
    └── <trip_id>/
```

Raw and processed datasets are not tracked in Git. The demo requires the original trip files, including GPS tracks and road networks. Reference routes are needed for evaluation metrics.

The default deployed model is `artifacts/models/lightgbm.joblib`.

### 3. Start the application

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open:

- Dashboard: <http://localhost:8000/ui/>
- API documentation: <http://localhost:8000/docs>
- Health check: <http://localhost:8000/health>

Choose a student-test trip and click **Run Map Matching**.

## Understanding the dashboard

The dashboard displays the selected trip, ML-only and HMM point accuracy, inference time, and a detailed metric comparison.

| Map layer | Appearance | Meaning |
|---|---|---|
| GPS observations | Blue | Original GPS measurements |
| ML-only prediction | Orange dashed | Highest-scoring road candidate for each point |
| HMM / Viterbi prediction | Green | Sequence-aware road predictions |
| Ground truth | Purple | Reference route for comparison |
| Points changed by HMM | Red circles | Points where HMM changed the ML-only assignment |

Red circles indicate changed predictions, not necessarily errors. Closely matching routes may overlap visually; toggle individual layers to compare them.

The metrics apply to the selected trip, so they can differ from the aggregate results above. Inference time varies with trip size, hardware, and caching.

## Two-laptop demo

On the backend laptop:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

On another laptop on the same network, open `http://<BACKEND_IP>:8000/ui/`. The backend must be reachable on port 8000.

Alternatively, serve the frontend separately:

```powershell
python -m http.server 5500 --directory src\phase8_ui
```

Open `http://localhost:5500/?api=http://<BACKEND_IP>:8000`, or enter the backend address in the dashboard and click **Connect**.

See [the demo guide](docs/DEMO_GUIDE.md) for network setup details.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Model availability and service status |
| GET | `/trips` | Available student-test trips |
| POST | `/predict` | Run inference for an existing student-test trip |
| GET | `/metrics` | Saved model comparison, HMM evaluation, and tuning results |
| GET | `/ui/` | Interactive dashboard |

Example request:

```powershell
Invoke-RestMethod -Method Post `
  -Uri 'http://localhost:8000/predict' `
  -ContentType 'application/json' `
  -Body '{"trip_id":"00000052","include_geometry":true}'
```

The prediction response includes route geometry when requested, evaluation metrics, and timing information. Repeated requests may be served from the backend cache.

Optional environment variables:

- `PATHSNAP_MODEL`: override the default model bundle path.
- `PATHSNAP_CORS_ORIGINS`: comma-separated allowed frontend origins; the default is `*`.

## CLI and training

Run one trip and save its result:

```powershell
.\.venv\Scripts\python.exe -m src.phase6_pipeline.end_to_end_matcher 00000052 --out results/demo_00000052.json
```

Evaluate HMM decoding across the student-test trips:

```powershell
.\.venv\Scripts\python.exe -m src.phase5_hmm.evaluate_hmm
```

Rebuild features, train models, or run tuning:

```powershell
.\.venv\Scripts\python.exe scripts\build_phase3_dataset.py
.\.venv\Scripts\python.exe -m src.phase4_models.train_models
.\.venv\Scripts\python.exe -m src.phase4_models.bayesian_optimization --trials 12
```

These commands require the dataset and can regenerate datasets, model files, or evaluation artifacts.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_structure tests.test_phase3 tests.test_phase4 tests.test_phase5_hmm tests.test_phase6_7_pipeline_api
```

Some checks require local data and trained artifacts. Consult the test output for skipped checks and missing prerequisites.

## Repository layout

```text
api/main.py                  FastAPI entry point
src/
  phase1_eda/                 Loading, projection, audits, and exploratory maps
  phase2_preprocessing/      Preprocessing package
  phase3_features/           Candidate search, features, labels, dataset building
  phase4_models/             Model training, evaluation, and Optuna tuning
  phase5_hmm/                Road transitions, Viterbi decoding, evaluation
  phase6_pipeline/           End-to-end inference and CLI
  phase7_api/                FastAPI implementation
  phase8_ui/                 HTML/CSS/JavaScript dashboard and Leaflet assets
artifacts/models/            Saved model bundles
results/metrics/             Saved evaluation and tuning results
results/maps/                Exploratory maps
data/                       Local datasets (not tracked)
scripts/                    Dataset, demo, and reporting utilities
tests/                      Unit and integration tests
docs/                       Guides and project documentation
reports/                    Reports and logs
```

## Scope and limitations

- The API selects existing student-test trips by ID; it does not currently provide an arbitrary GPS-file upload endpoint.
- Map matching requires the corresponding road network. This is route reconstruction, not turn-by-turn navigation or destination planning.
- A point with no candidate road segment is explicitly marked unmatched. Missing candidates, time gaps, and unreachable transitions can leave route discontinuities.
- Model inference runs locally. Online basemap tiles require internet access; Leaflet assets are bundled with the frontend.
- The dashboard and API are intended for local or LAN demonstrations. The current service has no authentication layer.
- Saved results describe the supplied dataset; performance on other regions or GPS collection conditions has not been established here.
