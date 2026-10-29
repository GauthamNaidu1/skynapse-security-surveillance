# Empirical Machine Learning Evaluation & Serving Audit (v2)

**Audit Date**: October 30, 2026  
**Repository**: [`GauthamNaidu1/skynapse-security-surveillance`](file:///d:/AU%20Academics/2%272%20IMP%20Docs/hyderabad-skynapse)  
**Primary Artifacts**: [`results/crime_metrics.json`](file:///d:/AU%20Academics/2%272%20IMP%20Docs/hyderabad-skynapse/results/crime_metrics.json), [`results/robustness.json`](file:///d:/AU%20Academics/2%272%20IMP%20Docs/hyderabad-skynapse/results/robustness.json), and measured terminal logs.

---

## 1. Core Statistical Conclusion

> **Empirical Finding**: **No model reliably beats chance.**
>
> 1. **Permutation Test ($p = 0.5473$)**: Shuffling training labels 200 times showed that 109 out of 200 random noise models ($54.5\%$) achieved a test Macro-F1 equal to or higher than the trained model ($0.3107$). The model is statistically indistinguishable from random noise ($p \ge 0.05$).
> 2. **Worse than Stratified Chance**: A 100-seed Stratified Dummy baseline averages a Macro-F1 of **$0.3285$** and a Balanced Accuracy of **$0.3307$**, both exceeding the trained K-Nearest Neighbors model (**$0.3107$** and **$0.3212$** respectively).
> 3. **Below Naive Guessing on Raw Accuracy**: Always predicting the majority class yields **$52.67\%$** accuracy, whereas the trained model achieves only **$42.67\%$** ($95\%\text{ CI: }[0.3467, 0.5067]$).
> 4. **High-Severity Crime Detection Failure**: The model correctly identified only 1 out of 23 high-severity incidents ($4.35\%$ recall, $95\%\text{ CI: }[0.0000, 0.1429]$), which is statistically indistinguishable from zero.

---

## 2. Test Set Evaluation: Models vs Baselines with 95% Confidence Intervals

Every metric below was evaluated on the unseen 15% chronological test set ($N=150$, dates: May 1, 2025 to September 25, 2025).  
*Source files: Point estimates from [`results/crime_metrics.json`](file:///d:/AU%20Academics/2%272%20IMP%20Docs/hyderabad-skynapse/results/crime_metrics.json#L190-L373); 95% bootstrap confidence intervals (1,000 resamples) from [`results/robustness.json`](file:///d:/AU%20Academics/2%272%20IMP%20Docs/hyderabad-skynapse/results/robustness.json#L27-L115).*

| Model Architecture | Test Macro-F1 (95% CI) | Balanced Accuracy (95% CI) | Raw Accuracy (95% CI) | Source File Citation |
| :--- | :---: | :---: | :---: | :--- |
| **Dummy (Most Frequent)** | **0.2300**<br>`[0.2079, 0.2500]` | **0.3333**<br>`[0.3333, 0.3333]` | **0.5267**<br>`[0.4532, 0.6000]` | [`results/crime_metrics.json:L194`](file:///d:/AU%20Academics/2%272%20IMP%20Docs/hyderabad-skynapse/results/crime_metrics.json#L194), [`results/robustness.json:L30-54`](file:///d:/AU%20Academics/2%272%20IMP%20Docs/hyderabad-skynapse/results/robustness.json#L30-L54) |
| **Dummy (Stratified, Single Seed 42)** | **0.3813**<br>`[0.3048, 0.4570]` | **0.3846**<br>`[0.3143, 0.4615]` | **0.4600**<br>`[0.3800, 0.5467]` | [`results/crime_metrics.json:L220`](file:///d:/AU%20Academics/2%272%20IMP%20Docs/hyderabad-skynapse/results/crime_metrics.json#L220), [`results/robustness.json:L59-83`](file:///d:/AU%20Academics/2%272%20IMP%20Docs/hyderabad-skynapse/results/robustness.json#L59-L83) |
| **Logistic Regression** | **0.2292**<br>`[0.1659, 0.3003]` | **0.2710**<br>`[0.1926, 0.3571]` | **0.2400**<br>`[0.1733, 0.3133]` | [`results/crime_metrics.json:L246`](file:///d:/AU%20Academics/2%272%20IMP%20Docs/hyderabad-skynapse/results/crime_metrics.json#L246), [`results/robustness.json:L88-112`](file:///d:/AU%20Academics/2%272%20IMP%20Docs/hyderabad-skynapse/results/robustness.json#L88-L112) |
| **Decision Tree (depth=5)** | **0.2546**<br>`[0.1912, 0.3165]` | **0.2732**<br>`[0.2007, 0.3437]` | **0.3333**<br>`[0.2533, 0.4000]` | [`results/crime_metrics.json:L272`](file:///d:/AU%20Academics/2%272%20IMP%20Docs/hyderabad-skynapse/results/crime_metrics.json#L272), [`results/robustness.json:L117-141`](file:///d:/AU%20Academics/2%272%20IMP%20Docs/hyderabad-skynapse/results/robustness.json#L117-L141) |
| **K-Nearest Neighbors (Winner)** | **0.3107**<br>`[0.2423, 0.3811]` | **0.3212**<br>`[0.2577, 0.3838]` | **0.4267**<br>`[0.3467, 0.5067]` | [`results/crime_metrics.json:L298`](file:///d:/AU%20Academics/2%272%20IMP%20Docs/hyderabad-skynapse/results/crime_metrics.json#L298), [`results/robustness.json:L146-170`](file:///d:/AU%20Academics/2%272%20IMP%20Docs/hyderabad-skynapse/results/robustness.json#L146-L170) |
| **Random Forest** | **0.3323**<br>`[0.2554, 0.4138]` | **0.3330**<br>`[0.2597, 0.4135]` | **0.4000**<br>`[0.3265, 0.4867]` | [`results/crime_metrics.json:L324`](file:///d:/AU%20Academics/2%272%20IMP%20Docs/hyderabad-skynapse/results/crime_metrics.json#L324), [`results/robustness.json:L175-199`](file:///d:/AU%20Academics/2%272%20IMP%20Docs/hyderabad-skynapse/results/robustness.json#L175-L199) |
| **Voting Classifier (Soft)** | **0.2878**<br>`[0.2250, 0.3510]` | **0.2944**<br>`[0.2323, 0.3576]` | **0.3800**<br>`[0.3000, 0.4533]` | [`results/crime_metrics.json:L350`](file:///d:/AU%20Academics/2%272%20IMP%20Docs/hyderabad-skynapse/results/crime_metrics.json#L350), [`results/robustness.json:L204-228`](file:///d:/AU%20Academics/2%272%20IMP%20Docs/hyderabad-skynapse/results/robustness.json#L204-L228) |

### 100-Seed Stratified Dummy Benchmark
*Source: [`results/robustness.json:L230-264`](file:///d:/AU%20Academics/2%272%20IMP%20Docs/hyderabad-skynapse/results/robustness.json#L230-L264)*
To prevent seed cherry-picking, the Stratified DummyClassifier was evaluated across 100 different random seeds:
- **Macro-F1**: Mean = **$0.3285 \pm 0.0396$** (95% range: `[0.2495, 0.4088]`)
- **Balanced Accuracy**: Mean = **$0.3307 \pm 0.0396$** (95% range: `[0.2526, 0.4100]`)
- **Raw Accuracy**: Mean = **$0.3928 \pm 0.0405$** (95% range: `[0.3165, 0.4635]`)
- **Takeaway**: The average chance Macro-F1 ($0.3285$) exceeds the trained model ($0.3107$).

### Per-Class Recall with 95% Confidence Intervals
*Source: [`results/crime_metrics.json:L375-394`](file:///d:/AU%20Academics/2%272%20IMP%20Docs/hyderabad-skynapse/results/crime_metrics.json#L375-L394), [`results/robustness.json:L700-750`](file:///d:/AU%20Academics/2%272%20IMP%20Docs/hyderabad-skynapse/results/robustness.json#L700-L750)*
- **Low Severity (Class 0, $N=79$)**:
  - Precision: **$0.5455$**, Recall: **$0.6076$** ($48/79$), F1: **$0.5749$**
  - Recall 95% CI: `[0.4937, 0.7215]`
- **Medium Severity (Class 1, $N=48$)**:
  - Precision: **$0.2632$**, Recall: **$0.3125$** ($15/48$), F1: **$0.2857$**
  - Recall 95% CI: `[0.1875, 0.4444]`
- **High Severity (Class 2, $N=23$)**:
  - Precision: **$0.2000$**, Recall: **$0.0435$** ($1/23$), F1: **$0.0714$**
  - Recall 95% CI: **`[0.0000, 0.1429]`**
  - **Takeaway**: High severity recall touches zero at the 95% confidence boundary ($95.65\%$ false-negative rate).

---

## 3. Permutation Test: Null Distribution & Empirical p-value

*Source: [`results/robustness.json:L660-698`](file:///d:/AU%20Academics/2%272%20IMP%20Docs/hyderabad-skynapse/results/robustness.json#L660-L698)*

To evaluate whether the model learned any genuine signal, the training labels were randomly shuffled 200 times. A fresh K-Nearest Neighbors pipeline was trained on each shuffled dataset and evaluated on the un-shuffled test set:

- **Real Model Test Macro-F1**: **$0.3107$**
- **Null Distribution Mean $\pm$ Std**: **$0.3130 \pm 0.0327$**
- **Null Median**: **$0.3129$**
- **Null 95th Percentile**: **$0.3668$**
- **Null Maximum**: **$0.4010$**
- **Permutations Equal to or Greater than Real**: **109 out of 200** ($54.5\%$)
- **Empirical Monte Carlo $p$-value**:
  $$p = \frac{109 + 1}{200 + 1} = \mathbf{0.5473}$$
- **Statistical Significance ($\alpha = 0.05$)**: **Not Significant ($p \ge 0.05$)**.

The model's performance is consistent with training on randomly permuted labels.

---

## 4. Rolling-Origin Expanding Evaluation (5 Chronological Folds)

*Source: [`results/robustness.json:L266-658`](file:///d:/AU%20Academics/2%272%20IMP%20Docs/hyderabad-skynapse/results/robustness.json#L266-L658)*

Evaluation across 5 expanding chronological windows over the 1,000-record dataset ($N=150$ per test fold). Hyperparameters were tuned strictly within each fold's inner training partition:

| Fold Index | Training Window | Test Window | Majority Dummy F1 | Stratified Dummy F1 | KNN Test F1 | Random Forest F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | $0 \to 250$ | $250 \to 400$ | 0.2162 | 0.3541 | 0.3256 | 0.2859 |
| **Fold 2** | $0 \to 400$ | $400 \to 550$ | 0.2182 | 0.2982 | 0.3028 | 0.3291 |
| **Fold 3** | $0 \to 550$ | $550 \to 700$ | 0.2182 | 0.3951 | 0.3266 | 0.3401 |
| **Fold 4** | $0 \to 700$ | $700 \to 850$ | 0.2058 | 0.3421 | 0.2668 | 0.2764 |
| **Fold 5** | $0 \to 850$ | $850 \to 1000$ | 0.2300 | 0.2867 | 0.3350 | 0.4025 |

### Summary Over All 5 Folds:
- **Dummy (Most Frequent)**: Macro-F1 = **$0.2177 \pm 0.0077$** | Balanced Accuracy = **$0.3333 \pm 0.0000$** | Raw Accuracy = **$0.4853 \pm 0.0254$**
- **Dummy (Stratified)**: Macro-F1 = **$0.3352 \pm 0.0491$** | Balanced Accuracy = **$0.3396 \pm 0.0522$** | Raw Accuracy = **$0.3880 \pm 0.0364$**
- **K-Nearest Neighbors**: Macro-F1 = **$0.3114 \pm 0.0247$** | Balanced Accuracy = **$0.3339 \pm 0.0184$** | Raw Accuracy = **$0.4413 \pm 0.0290$**
- **Random Forest**: Macro-F1 = **$0.3268 \pm 0.0450$** | Balanced Accuracy = **$0.3289 \pm 0.0438$** | Raw Accuracy = **$0.3627 \pm 0.0331$**

Across temporal windows, the Stratified Dummy baseline ($0.3352$) outperformed K-Nearest Neighbors ($0.3114$).

---

## 5. Serving Checks, Feature Alignment, and Measured Latencies

*Source: Measured terminal outputs from active test runs on Express (`localhost:3000`) and FastAPI (`127.0.0.1:8000`).*

### Real API Responses via Express Gateway (`POST /api/predict`)

1. **Request 1** (`hour=23`, `density=85`, `location_id=1` [Charminar]):
   ```json
   {
     "threat_level": 1,
     "confidence": 0.6,
     "probabilities": [0.4, 0.6, 0.0],
     "classes": ["Low", "Medium", "High"],
     "source": "ML Model (crime_pipeline.joblib)",
     "is_fallback": false,
     "cluster_id": 1
   }
   ```
2. **Request 2** (`hour=14`, `density=40`, `location_id=4` [Gachibowli]):
   ```json
   {
     "threat_level": 1,
     "confidence": 0.6,
     "probabilities": [0.4, 0.6, 0.0],
     "classes": ["Low", "Medium", "High"],
     "source": "ML Model (crime_pipeline.joblib)",
     "is_fallback": false,
     "cluster_id": 1
   }
   ```
3. **Request 3** (`hour=2`, `density=15`, `location_id=2` [Banjara Hills]):
   ```json
   {
     "threat_level": 1,
     "confidence": 0.6,
     "probabilities": [0.2, 0.6, 0.2],
     "classes": ["Low", "Medium", "High"],
     "source": "ML Model (crime_pipeline.joblib)",
     "is_fallback": false,
     "cluster_id": 1
   }
   ```

### Circuit-Breaker Fallback Response (FastAPI Service Terminated)
When the Python microservice was stopped, repeating Request 1 returned:
```json
{
  "threat_level": 1,
  "confidence": 0.4360587002096436,
  "cluster_id": 3,
  "probabilities": [0.01677, 0.25367, 0.43606, 0.29350],
  "source": "heuristic fallback",
  "is_fallback": true
}
```

### Feature Alignment Audit
- **Pipeline Training Expectation**: 15 features in exact order (`pipe.feature_names_in_` in [`python/evaluate_models.py`](file:///d:/AU%20Academics/2%272%20IMP%20Docs/hyderabad-skynapse/python/evaluate_models.py#L81-L120)):
  `['hour_sin', 'hour_cos', 'day_of_week', 'is_weekend', 'trailing_30d_density', 'area_Gachibowli', 'area_Secunderabad', 'area_Ameerpet', 'area_Kukatpally', 'area_LB Nagar', 'area_Charminar', 'area_Jubilee Hills', 'area_Madhapur', 'area_Banjara Hills', 'area_Begumpet']`.
- **Serving Construction**: [`python/serve.py:format_features`](file:///d:/AU%20Academics/2%272%20IMP%20Docs/hyderabad-skynapse/python/serve.py#L28-L56) constructs 26 columns, including `'Area'` and 10 location indicators that were absent from the 700-sample training partition.
- **Alignment Finding**: Because inputs are formatted as a `pandas.DataFrame`, `ColumnTransformer` aligns by column name and filters extraneous columns. When a request specifies an unseen sector (e.g. `'Hitech City'`), all 10 trained sector indicators evaluate to 0, functioning as an implicit reference baseline category.

### Latency Measurements (50 Sequential Requests per Mode)
*Measured via `System.Diagnostics.Stopwatch`*:
- **ML Model Mode (FastAPI Microservice)**:
  - Median: **13.10 ms**
  - 95th Percentile: **28.44 ms**
  - Mean: $57.10\text{ ms}$, Min: $10.74\text{ ms}$, Max: $2132.81\text{ ms}$ (initial socket/pool connection)
- **Heuristic Fallback Mode (FastAPI Offline)**:
  - Median: **4.18 ms**
  - 95th Percentile: **5.58 ms**
  - Mean: $46.89\text{ ms}$, Min: $3.09\text{ ms}$, Max: $2129.03\text{ ms}$ (initial socket connection)

---

## 6. Claims I Must NOT Make

1. **DO NOT claim the model beats chance or naive baselines**:
   * The permutation test yielded $p = 0.5473$, showing the model is statistically indistinguishable from label-shuffled noise.
   * Stratified dummy guessing averages a higher Macro-F1 ($0.3285$) than the model ($0.3107$).
2. **DO NOT claim high accuracy**:
   * Raw accuracy is $42.67\%$, which is $10$ percentage points lower than a zero-intelligence dummy always predicting Low severity ($52.67\%$).
3. **DO NOT claim operational reliability for emergency dispatch**:
   * High-severity crime recall is $4.35\%$ ($95\%\text{ CI: }[0.0000, 0.1429]$), missing $95.65\%$ of critical emergencies.
4. **DO NOT claim trained computer vision models**:
   * Action recognition uses an off-the-shelf MoViNet-A0 model with Kinetics-600 weights and Gemini 3.1 Pro zero-shot prompts. No custom video training was conducted.
5. **DO NOT claim live camera streaming**:
   * The platform operates on 1,000 historical records and a labeled simulation generator.

---

## 7. Resume Bullets (Empirically Grounded)

* **Bullet 1 (Methodology & Statistical Rigor)**:
  > Evaluated multi-class crime risk classification across 1,000 historical records under a 70/15/15 chronological holdout; established via permutation testing ($p = 0.5473$) and 100-seed dummy baselines that tabular spatio-temporal features yielded test Macro-F1 ($0.3107$, $95\%\text{ CI: }[0.2423, 0.3811]$) indistinguishable from label noise, demonstrating that no tested model reliably beats chance.

* **Bullet 2 (Systems Architecture & Serving Integration)**:
  > Developed an Express and TypeScript surveillance gateway integrated with a FastAPI machine learning microservice, implementing an automated circuit-breaker fallback that handles service disconnects (median latency: $13.10\text{ ms}$ for model inferences, $4.18\text{ ms}$ for heuristic fallbacks).

* **Bullet 3 (Empirical Auditing & Transparent UI)**:
  > Audited data pipelines to eliminate data leakage in trailing 30-day incident densities, serializing test metrics dynamically from JSON files and designing a dashboard that clearly distinguishes verified historical records from simulated demo feeds.

---

## 8. Interview Questions & Grounded Answers

### Q1: "What did your evaluation metrics show about the viability of predicting crime severity from this dataset?"
> **Answer**:  
> "Our evaluation revealed that no model reliably beats chance. In our chronological test set ($N=150$), K-Nearest Neighbors achieved a test Macro-F1 of $0.3107$, whereas an un-tuned Stratified Dummy baseline averaged $0.3285$ across 100 random seeds. Furthermore, a 200-iteration permutation test yielded an empirical $p$-value of $0.5473$, meaning over half of models trained on randomly shuffled labels matched or exceeded our score. Tabular features such as hour and sector had Pearson correlations below $|r| \le 0.061$ with crime severity, providing insufficient signal to predict high-severity incidents, where test recall was only $4.35\%$ ($95\%\text{ CI: }[0.0000, 0.1429]$)."

### Q2: "Why is your model's test accuracy only 42.7% when naive guessing achieves 52.7%?"
> **Answer**:  
> "The dataset exhibits class imbalance, with Low severity representing $52.7\%$ of test records. A naive majority-class classifier achieves $52.67\%$ accuracy by predicting Low for every sample, but scores $0\%$ recall on Medium and High crimes. While our classifier attempted to predict minority categories, its balanced accuracy was $0.3212$ ($95\%\text{ CI: }[0.2577, 0.3838]$), which fails to exceed the naive chance threshold of $0.3333$. Reporting accuracy alone would mask the fact that the model missed $95.65\%$ of high-severity crimes and did not outperform simple chance."
