---

# Federated Learning Pipeline Explained Using the Code

This document explains **how federated learning operates step by step using the pipeline code in this repository**. The code orchestrates a full federated learning workflow using the **FLuKE framework**, the **Adult dataset**, and **four simulated hospitals (clients)**.

The pipeline runs **three federated experiments** with different local training settings and evaluates the resulting models using performance, fairness, cost, and runtime metrics.

---

# 1. Utility Functions

The pipeline begins by defining helper functions used later.

```python
def safe_read_csv(path):
```

This function safely loads CSV files produced by the FLuKE framework.

Why this matters in federated learning:

After each experiment, FLuKE generates several output artifacts:

* `postfit_metrics.csv`
* `comm_costs.csv`

These files contain the **evaluation metrics and communication statistics** generated during training.

The pipeline reads these files to compute final results.

---

# 2. Fairness Metrics

The next functions compute fairness metrics.

```python
def compute_spd(df, sensitive_col="sex", target_col="income")
```

This computes **Statistical Parity Difference (SPD)**.

SPD measures the difference between the probability of predicting **high income** for different groups:

[
SPD = P(Y=1|Female) - P(Y=1|Male)
]

---

```python
def compute_eod(df, sensitive_col="sex", target_col="income")
```

This computes **Equal Opportunity Difference (EOD)**.

EOD measures the difference in **true positive rates** across groups.

---

```python
def add_fairness_metrics(postfit_df, df_full)
```

This function attaches fairness metrics to the federated learning results.

Important observation:

The fairness metrics are calculated from the **original dataset**, meaning the bias comes from the **dataset itself**, not from the training process.

This explains why the values remain constant across runs.

---

# 3. Directory Configuration

The pipeline then defines where the project artifacts are stored.

```python
root_dir = "/storage/fl-lab"
data_dir = os.path.join(root_dir, "data")
config_dir = os.path.join(root_dir, "config")
runs_dir = os.path.join(root_dir, "runs")
```

These directories contain:

| Directory | Purpose                        |
| --------- | ------------------------------ |
| data      | datasets                       |
| config    | experiment configuration files |
| runs      | results of federated training  |

Each run produces its own folder:

```
runs/run_1
runs/run_2
runs/run_3
```

Inside each folder, FLuKE stores metrics and communication statistics.

---

# 4. Dataset Loading

The pipeline loads the **Adult dataset**.

```python
df_adult = pd.read_csv(adult_csv)
```

Output example:

```
Adult dataset loaded: 27145 samples, 14 features
```

The code identifies:

```
Sensitive column: sex
Target column: income
```

These columns are used later to compute fairness metrics.

---

# 5. Baseline Dataset Fairness

Before federated learning starts, the pipeline computes **baseline fairness metrics**.

```python
spd_value = compute_spd(df_adult)
eod_value = compute_eod(df_adult)
```

Output example:

```
Dataset SPD (baseline): -0.1991
Dataset EOD (baseline): -0.1991
```

Interpretation:

* female individuals receive positive predictions less frequently than males
* the dataset contains **gender bias**

Federated learning does **not remove this bias automatically**.

---

# 6. YAML Generation for Each Experiment

Federated learning experiments are configured using **YAML configuration files**.

The pipeline generates temporary YAML files dynamically.

```python
def update_tmp_yaml(run_id, local_epochs=5):
```

Two files are created per run:

```
tmp_exp_X.yaml
tmp_alg_X.yaml
```

---

## Experiment Configuration

The experiment configuration defines the **federated learning protocol**.

```python
exp_cfg["protocol"]["n_clients"] = 4
exp_cfg["protocol"]["n_rounds"] = 20
```

Meaning:

| Parameter        | Value       |
| ---------------- | ----------- |
| Clients          | 4 hospitals |
| Federated rounds | 20          |

---

### Data Distribution

```python
exp_cfg["data"]["distribution"]["name"] = "dir"
exp_cfg["data"]["distribution"]["beta"] = 1.0
```

This means:

The dataset is split across hospitals using a **Dirichlet distribution**.

Each hospital receives a **different subset of patients**.

This simulates **real-world data heterogeneity**.

---

## Algorithm Configuration

The algorithm YAML defines the model and training parameters.

```python
alg_cfg["hyperparameters"]["client"]["local_epochs"] = local_epochs
```

Local epochs control how long each hospital trains locally before sending updates.

Three values are tested:

| Run   | Local Epochs |
| ----- | ------------ |
| Run 1 | 5            |
| Run 2 | 10           |
| Run 3 | 20           |

---

The model used is:

```python
Adult_LogReg
```

A logistic regression model designed for the Adult dataset.

---

# 7. Launching the Federated Learning Process

The pipeline then runs the federated training using FLuKE.

```python
def run_fluke(run_id, tmp_exp_yaml, tmp_alg_yaml):
```

The command executed is:

```
fluke federation tmp_exp.yaml tmp_alg.yaml
```

This launches the **entire federated learning process**.

FLuKE internally performs:

1. dataset partitioning across clients
2. server initialization
3. federated training rounds
4. model aggregation
5. evaluation

---

# 8. Federated Experiment Loop

The core of the pipeline is the experiment loop.

```python
for run_id, local_epochs in zip(range(1,4), [5,10,20]):
```

Three federated experiments are executed.

---

Example output:

```
Running experiment 1
Parameters -> Local epochs: 5, Clients: 4, Rounds: 20
```

Each experiment runs the full federated training process.

---

# 9. What Happens Inside Each Federated Round

Inside FLuKE, the following process happens (conceptually):

### Step 1 — Server Initialization

The server creates the initial global model.

```
W0
```

---

### Step 2 — Model Broadcast

The server sends the model to all hospitals.

```
Server → Clients
```

---

### Step 3 — Local Training

Each hospital trains locally using its private dataset.

```
Hospital A trains on dataset A
Hospital B trains on dataset B
Hospital C trains on dataset C
Hospital D trains on dataset D
```

Training lasts for:

```
local_epochs
```

---

### Step 4 — Local Updates

Each hospital produces an updated model.

```
W_A
W_B
W_C
W_D
```

---

### Step 5 — Model Upload

Hospitals send model updates back to the server.

```
Clients → Server
```

Only **model parameters** are transmitted.

Patient data remains local.

---

### Step 6 — Federated Averaging

The server combines the models:

```
W_global = average(W_A, W_B, W_C, W_D)
```

This produces the **new global model**.

---

### Step 7 — Next Round

The process repeats for:

```
20 rounds
```

---

# 10. Reading Experiment Results

After training finishes, the pipeline loads the metrics.

```python
postfit = safe_read_csv(postfit_path)
comm = safe_read_csv(comm_path)
```

Files used:

| File                | Purpose             |
| ------------------- | ------------------- |
| postfit_metrics.csv | performance metrics |
| comm_costs.csv      | communication cost  |

---

The pipeline extracts:

```
accuracy
macro_f1
micro_f1
SPD
EOD
communication cost
runtime
```

---

# 11. Per-Run Results

Each run produces a summary line.

Example:

```
Run 1 completed
Accuracy: 0.8738
Macro F1: 0.7785
SPD: -0.1991
Runtime: 69s
```

These values show the **performance of the final global model**.

---

# 12. Final Experiment Summary

After all runs complete, the pipeline computes averages.

```python
df_runs = pd.DataFrame(results)
```

Example results:

| Local Epochs | Accuracy | Macro F1 |
| ------------ | -------- | -------- |
| 5            | 0.8738   | 0.7785   |
| 10           | 0.8711   | 0.7666   |
| 20           | 0.8686   | 0.7584   |

Observation:

Increasing local epochs slightly **reduces global accuracy**.

This occurs due to **client model divergence**.

---

# 13. Communication Cost

The pipeline measures total communication.

Example:

```
4920 per run
```

Communication cost depends on:

* number of clients
* number of rounds
* model size

Since these remain constant, the cost is identical across runs.

---

# 14. Runtime Analysis

Runtime increases with local epochs.

Example:

| Local Epochs | Runtime |
| ------------ | ------- |
| 5            | ~69s    |
| 10           | ~70s    |
| 20           | ~98s    |

More local training increases computation time.

---

# 15. Privacy Preservation

Privacy is preserved because:

* hospitals **never share raw datasets**
* only **model parameters** are transmitted
* each hospital trains locally

Data flow in the pipeline:

```
Local dataset
    ↓
Local model training
    ↓
Model parameters sent to server
    ↓
Server aggregates updates
```

Thus patient data never leaves the hospital.

---

# 16. Overall Federated Workflow

The pipeline implements the following full workflow:

```
Load dataset
      ↓
Compute baseline fairness
      ↓
Generate experiment YAML
      ↓
Run federated learning (FLuKE)
      ↓
Clients train locally
      ↓
Server aggregates models
      ↓
Repeat for 20 rounds
      ↓
Evaluate model
      ↓
Store results
      ↓
Repeat experiment with new local epochs
      ↓
Compute final summary
```

---

# Key Take-Away

This pipeline demonstrates a **complete horizontal federated learning system** where:

* multiple institutions collaboratively train a model
* raw data never leaves the local clients
* a central server coordinates training
* results are evaluated using performance, fairness, communication cost, and runtime metrics

It provides a realistic simulation of **multi-hospital collaborative machine learning**.
