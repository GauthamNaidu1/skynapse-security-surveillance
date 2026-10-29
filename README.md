# Hyderabad Skynapse: Smart City Surveillance & Crime Risk Prediction

An auditable, full-stack security and surveillance platform integrating spatial-temporal crime risk classification, persistent incident logging, and multimodal CCTV threat analysis.

---

## 1. Project Overview

Hyderabad Skynapse combines an empirical machine learning pipeline with a modern surveillance operations dashboard:
* **Spatio-Temporal Risk Predictor**: Formulates crime risk as a multi-class severity prediction task evaluated with chronological holdouts and calibrated baselines.
* **Persistent Geospatial Dashboard**: Interactive Leaflet map displaying verified historical crime records from SQLite with a distinct, clearly labeled demo simulation mode.
* **Multimodal Surveillance Interface**: Video analysis integration leveraging pre-trained action recognition (Google MoViNet-A0) and zero-shot foundation models (Google Gemini 3.1 Pro).

---

## 2. System Architecture

```text
+---------------------------------------------------------------------------------+
|                                 CLIENT BROWSER                                  |
|  React 19 + TypeScript + Vite + Tailwind CSS + Leaflet (CARTO Tiles) + Recharts |
|                                                                                 |
|   [ Incident Monitor ]    [ Crime Analytics ]    [ AI Predictor ]   [ Vision ]  |
|    - Historical Mode       - Dynamic Metrics      - Threat Level     - MoViNet  |
|    - Simulated Feed        - Recharts Trends      - Fallback Badge   - Gemini   |
+---------------------------------------------------------------------------------+
                                       |
                   HTTP Requests (Port 3000 /api/*)
                                       v
+---------------------------------------------------------------------------------+
|                              NODE / EXPRESS BACKEND                             |
|                               (server.ts)                                       |
|                                                                                 |
|   GET /api/history        -> Reads historical records from SQLite               |
|   GET /api/live-incident  -> Returns historical head or tagged simulation       |
|   GET /api/model-info     -> Reads verified metrics from crime_metrics.json     |
|   POST /api/predict       -> Proxies to Python ML service (with fallback)       |
|   POST /api/analyze-video -> Executes local MoViNet-A0 inference                |
|   POST /api/analyze-footage -> Invokes Gemini 3.1 Pro multimodal API            |
+---------------------------------------------------------------------------------+
           |                                                      |
           v                                                      v
+-----------------------+                              +--------------------------+
|    SQLITE DATABASE    |                              |   PYTHON ML SERVICE      |
|     (skynapse.db)     |                              |   (python/serve.py:8000) |
|                       |                              |                          |
|  Tables:              |                              |  - ColumnTransformer     |
|   - incidents         |                              |  - StandardScaler        |
|   - predictions       |                              |  - OneHotEncoder         |
|   - training_logs     |                              |  - crime_pipeline.joblib |
+-----------------------+                              +--------------------------+
```

---

## 3. Dataset & Preprocessing

* **Dataset Source**: Hyderabad Crime Dataset ([Kaggle: `mikrokosmos1/crimedata`](https://www.kaggle.com/datasets/mikrokosmos1/crimedata)).
* **Target Variable**: Crime Severity Level ($y \in \{0, 1, 2\}$):
  * **Low (0)**: Traffic violations, suspicious activity, minor property disputes.
  * **Medium (1)**: Theft, burglary, vandalism, pickpocketing.
  * **High (2)**: Robbery, assault, fighting, arson, chain snatching.
* **Leakage-Free Temporal Splitting**:
  * Sorted chronologically into **70% Train / 15% Validation / 15% Test**.
  * Eliminates future-to-past information leakage present in random cross-validation.
* **Feature Engineering**:
  * `hour_sin` & `hour_cos`: Continuous cyclical sine/cosine representation ($\sin(2\pi \cdot \text{hour}/24)$, $\cos(2\pi \cdot \text{hour}/24)$).
  * `day_of_week` & `is_weekend`: Calendar temporal indicators.
  * `trailing_30d_density`: Computed strictly over historical events in the window $[T - 30\text{ days}, T)$ using binary search (`np.searchsorted`) per sector.
  * `Area`: One-hot encoded using categories discovered exclusively from the training split.

---

## 4. Model Evaluation & Benchmark Results

All evaluation metrics are calculated on the unseen **15% chronological test set** and exported to [`results/crime_metrics.json`](file:///d:/AU%20Academics/2%272%20IMP%20Docs/hyderabad-skynapse/results/crime_metrics.json) (never hardcoded):

### Evaluation Protocol
1. **Baselines**:
   * `DummyClassifier(strategy='most_frequent')` (Majority Class Baseline)
   * `DummyClassifier(strategy='stratified', random_state=42)`
   * `LogisticRegression(class_weight='balanced', random_state=42)`
2. **Candidates**:
   * `DecisionTreeClassifier(class_weight='balanced', random_state=42)`
   * `KNeighborsClassifier()`
   * `RandomForestClassifier(class_weight='balanced', random_state=42)`
   * Soft `VotingClassifier` combining top tuned estimators.
3. **Hyperparameter Selection**: Tuned strictly on the validation set using **Macro-F1** to prevent majority-class bias.
4. **Held-out Test Metrics**:

| Model | Test Accuracy | Balanced Accuracy | Macro-F1 | Outperforms Majority Baseline? |
| :--- | :---: | :---: | :---: | :---: |
| **Dummy (Most Frequent)** | Baseline | Baseline | Baseline | N/A (Naive Reference) |
| **Dummy (Stratified)** | Script-run | Script-run | Script-run | No |
| **Logistic Regression** | Script-run | Script-run | Script-run | Script-computed |
| **Decision Tree** | Script-run | Script-run | Script-run | Script-computed |
| **K-Nearest Neighbors** | Script-run | Script-run | Script-run | Script-computed |
| **Random Forest** | Script-run | Script-run | Script-run | Script-computed |
| **Voting Classifier (Soft)** | Script-run | Script-run | Script-run | Script-computed |

> **Note**: To populate the table with exact run numbers, place your dataset CSV in `data/crime_data.csv` and execute `py python/prepare_data.py` followed by `py python/evaluate_models.py`. The resulting metrics and normalized confusion matrix heatmap will be saved to `results/`.

---

## 5. System Limitations & Scientific Disclosures

1. **Pretrained-Only Video Threat Detection**:
   * Video analysis utilizes an off-the-shelf, pre-trained Google MoViNet-A0 model (trained on Kinetics-600) and Gemini 3.1 Pro zero-shot prompt classification.
   * **No custom training, fine-tuning, or quantitative benchmark on CCTV crime footage was performed by the author.** Action index mappings in `python/video_analysis.py` are demonstration heuristics.
2. **Simulated vs. Historical Feeds**:
   * In **"Historical Data"** mode, the application displays verified historical records persisted in SQLite (`skynapse.db`).
   * In **"Demo Simulation"** mode, incidents are generated synthetically via random perturbations to demonstrate dashboard dynamics. The UI clearly labels each mode with a badge.
3. **Algorithmic Bias and Ethical Risks in Crime Prediction**:
   * Historical crime records inherently reflect historical policing practices and selective reporting rather than true underlying crime frequency.
   * Predicting crime hotspots risks creating self-reinforcing feedback loops (over-policing neighborhoods, increasing recorded arrests, artificially validating model bias).
   * These models must not be deployed for punitive or automated patrol dispatch without comprehensive algorithmic fairness audits and demographic parity constraints.

---

## 6. Reproduction Guide

### Prerequisites
* Python 3.11 with `scikit-learn`, `pandas`, `numpy`, `joblib`, `matplotlib`
* Node.js 18+ and `npm`

### Step 1: Install Dependencies
```bash
# Node dependencies
npm install

# Python dependencies (for full ASGI FastAPI serving)
pip install scikit-learn pandas numpy joblib matplotlib fastapi uvicorn
```

### Step 2: Prepare Dataset
Download the crime CSV from [Kaggle: mikrokosmos1/crimedata](https://www.kaggle.com/datasets/mikrokosmos1/crimedata) and place it at:
```bash
data/crime_data.csv
```
Run chronological splitting and feature extraction:
```bash
py python/prepare_data.py
```

### Step 3: Run Model Evaluation
```bash
py python/evaluate_models.py
```
* Tunes candidates on validation data.
* Evaluates on held-out test data.
* Exports `models/crime_pipeline.joblib`, `results/crime_metrics.json`, and `results/confusion_matrix.png`.

### Step 4: Run Application Services

**Terminal 1 (Python Inference Microservice)**:
```bash
py python/serve.py
```
*(Runs on `http://127.0.0.1:8000`)*

**Terminal 2 (Express Web Application)**:
```bash
npm run dev
```
*(Runs on `http://localhost:3000`)*

* **AI Predictor (`/predictor`)**: Queries `python/serve.py` and displays the `ML Model (crime_pipeline.joblib)` badge (with automatic fallback to `heuristic fallback` if the Python service is offline).
* **Crime Analytics (`/analytics`)**: Dynamically renders the benchmark results directly from `results/crime_metrics.json`.
