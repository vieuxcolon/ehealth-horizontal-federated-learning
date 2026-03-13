# ===============================================================
# Adult-Only HFL Pipeline — 4 Clients, Dirichlet (beta=1.0), FedAvg
# Suppressed FLuKE internal messages, only per-run summary + final results
# ===============================================================

import os
import subprocess
import pandas as pd
import yaml
import time
import warnings
warnings.filterwarnings("ignore")

# -------------------------------
# Step 0 — Utility Functions
# -------------------------------
def safe_read_csv(path):
    if os.path.exists(path):
        df = pd.read_csv(path)
        if not df.empty:
            return df
    return None

# -------------------------------
# Step 0b — Fairness Metrics
# -------------------------------
def compute_spd(df, sensitive_col="sex", target_col="income"):
    priv = df[df[sensitive_col] == "Male"]
    unpriv = df[df[sensitive_col] == "Female"]
    if len(priv) == 0 or len(unpriv) == 0:
        return None
    return round((unpriv[target_col] == ">50K").mean() - (priv[target_col] == ">50K").mean(), 4)

def compute_eod(df, sensitive_col="sex", target_col="income"):
    priv = df[df[sensitive_col] == "Male"]
    unpriv = df[df[sensitive_col] == "Female"]
    if len(priv) == 0 or len(unpriv) == 0:
        return None
    return round((unpriv[target_col] == ">50K").mean() - (priv[target_col] == ">50K").mean(), 4)

def add_fairness_metrics(postfit_df, df_full, sensitive_col="sex", target_col="income"):
    if postfit_df is None:
        return postfit_df
    postfit_df["spd"] = [compute_spd(df_full, sensitive_col, target_col)] * len(postfit_df)
    postfit_df["eod"] = [compute_eod(df_full, sensitive_col, target_col)] * len(postfit_df)
    return postfit_df

# -------------------------------
# Step 1 — Directories & Dataset
# -------------------------------
root_dir = "/storage/fl-lab"  # Persistent Paperspace storage
data_dir = os.path.join(root_dir, "data")
config_dir = os.path.join(root_dir, "config")
runs_dir = os.path.join(root_dir, "runs")
adult_csv = os.path.join(data_dir, "adult", "adult.csv")

# -------------------------------
# Step 2 — Load Adult dataset
# -------------------------------
if not os.path.exists(adult_csv):
    raise FileNotFoundError(f"Adult dataset not found at {adult_csv}")

df_adult = pd.read_csv(adult_csv)
print(f"Adult dataset loaded: {df_adult.shape[0]} samples, {df_adult.shape[1]-1} features")

sensitive_col = "sex"
target_col = "income"
print(f"Detected sensitive column: {sensitive_col}")
print(f"Detected target column: {target_col}")

# Convert numeric columns if necessary
if df_adult[sensitive_col].dtype != object:
    df_adult[sensitive_col] = df_adult[sensitive_col].map({0: "Female", 1: "Male"})
if df_adult[target_col].dtype != object:
    df_adult[target_col] = df_adult[target_col].map({0: "<=50K", 1: ">50K"})

# Baseline fairness
spd_value = compute_spd(df_adult, sensitive_col, target_col)
eod_value = compute_eod(df_adult, sensitive_col, target_col)
print(f"Dataset SPD (baseline): {spd_value}")
print(f"Dataset EOD (baseline): {eod_value}")

# -------------------------------
# Step 3 — YAML generation per run
# -------------------------------
def update_tmp_yaml(run_id, local_epochs=5):
    exp_yaml = os.path.join(config_dir, "exp.yaml")
    alg_yaml = os.path.join(config_dir, "fedavg.yaml")
    tmp_exp_yaml = os.path.join(config_dir, f"tmp_exp_{run_id}.yaml")
    tmp_alg_yaml = os.path.join(config_dir, f"tmp_alg_{run_id}.yaml")

    # Experiment YAML
    with open(exp_yaml) as f:
        exp_cfg = yaml.safe_load(f)
    exp_cfg["protocol"]["n_clients"] = 4
    exp_cfg["protocol"]["eligible_perc"] = 1.0
    exp_cfg["protocol"]["n_rounds"] = 20
    exp_cfg["logger"]["log_dir"] = os.path.join(runs_dir, f"run_{run_id}")
    exp_cfg["data"]["dataset"]["name"] = "adult"
    exp_cfg["data"]["distribution"]["name"] = "dir"
    exp_cfg["data"]["distribution"]["beta"] = 1.0
    exp_cfg["compute_fairness_metrics"] = True

    with open(tmp_exp_yaml, "w") as f:
        yaml.dump(exp_cfg, f)

    # Algorithm YAML
    with open(alg_yaml) as f:
        alg_cfg = yaml.safe_load(f)
    alg_cfg["hyperparameters"]["client"]["local_epochs"] = local_epochs
    input_dim = df_adult.shape[1]-1
    alg_cfg["hyperparameters"]["model"] = "Adult_LogReg"
    alg_cfg["hyperparameters"]["net_args"] = {"input_dim": input_dim}

    with open(tmp_alg_yaml, "w") as f:
        yaml.dump(alg_cfg, f)

    return tmp_exp_yaml, tmp_alg_yaml

# -------------------------------
# Step 4 — Run FLuKE (suppressed output)
# -------------------------------
def run_fluke(run_id, tmp_exp_yaml, tmp_alg_yaml):
    cmd = ["fluke", "federation", tmp_exp_yaml, tmp_alg_yaml]
    start_time = time.time()
    subprocess.run(cmd, cwd=root_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return round(time.time() - start_time, 3)

# -------------------------------
# Step 5 — Execute experiments
# -------------------------------
results = []
for run_id, local_epochs in zip(range(1,4), [5,10,20]):
    tmp_exp_yaml, tmp_alg_yaml = update_tmp_yaml(run_id, local_epochs)

    # --- Start line with experiment path + parameters
    print(f"\nRunning experiment {run_id}: fluke federation {tmp_exp_yaml} {tmp_alg_yaml}")
    print(f"    Parameters -> Local epochs: {local_epochs}, Clients: 4, Rounds: 20, Distribution: Dirichlet (beta=1.0)")

    runtime = run_fluke(run_id, tmp_exp_yaml, tmp_alg_yaml)

    postfit_path = os.path.join(runs_dir, f"run_{run_id}/postfit_metrics.csv")
    comm_path = os.path.join(runs_dir, f"run_{run_id}/comm_costs.csv")
    postfit = safe_read_csv(postfit_path)
    comm = safe_read_csv(comm_path)
    postfit = add_fairness_metrics(postfit, df_adult)

    accuracy = postfit["accuracy"].mean() if postfit is not None else None
    macro_f1 = postfit["macro_f1"].mean() if postfit is not None else None
    micro_f1 = postfit["micro_f1"].mean() if postfit is not None else None
    spd = postfit["spd"].tolist()[-1] if postfit is not None else None
    eod = postfit["eod"].tolist()[-1] if postfit is not None else None
    total_comm = comm["comm_costs"].sum() if comm is not None else None

    results.append({
        "run_id": run_id,
        "local_epochs": local_epochs,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "spd": spd,
        "eod": eod,
        "total_comm_cost": total_comm,
        "runtime_sec": runtime
    })

    # --- End line with per-run metrics
    print(f"Run {run_id} completed | Local epochs: {local_epochs} | "
          f"Accuracy: {accuracy:.4f} | Macro F1: {macro_f1:.4f} | "
          f"SPD: {spd:.4f} | EOD: {eod:.4f} | Runtime: {runtime:.1f}s")

# -------------------------------
# Step 6 — Final Results Summary
# -------------------------------
df_runs = pd.DataFrame(results)
experiment_summary = {
    "runs": len(df_runs),
    "avg_accuracy": df_runs["accuracy"].mean(),
    "avg_macro_f1": df_runs["macro_f1"].mean(),
    "avg_micro_f1": df_runs["micro_f1"].mean(),
    "spd_mean": df_runs["spd"].mean(),
    "eod_mean": df_runs["eod"].mean(),
    "total_comm_cost": df_runs["total_comm_cost"].sum(),
    "avg_runtime_sec": df_runs["runtime_sec"].mean(),
    "total_runtime_sec": df_runs["runtime_sec"].sum()
}
df_experiment = pd.DataFrame([experiment_summary])

print("\n=== Metrics Summary Per Run ===")
print(df_runs.round(4))
print("\n=== Experiment Summary ===")
print(df_experiment.round(4))


# -------------------------------
# Step 8 — Experiment Summary
# -------------------------------
experiment_summary = {
    "runs": len(df_runs),
    "avg_accuracy": df_runs["accuracy"].mean(),
    "avg_macro_f1": df_runs["macro_f1"].mean(),
    "avg_micro_f1": df_runs["micro_f1"].mean(),
    "spd_mean": df_runs["spd"].mean(),
    "eod_mean": df_runs["eod"].mean(),
    "total_comm_cost": df_runs["total_comm_cost"].sum(),
    "avg_runtime_sec": df_runs["runtime_sec"].mean(),
    "total_runtime_sec": df_runs["runtime_sec"].sum()
}
df_experiment = pd.DataFrame([experiment_summary])
print("\n=== Experiment Summary ===")
print(df_experiment.round(4))

# -------------------------------
# Step 9 — Metrics Summary by Type
# -------------------------------
summary_by_type = {
    "utility_metrics": {
        "avg_runtime_sec": df_runs["runtime_sec"].mean(),
        "total_runtime_sec": df_runs["runtime_sec"].sum()
    },
    "cost_metrics": {
        "total_comm_cost_per_run": df_runs["total_comm_cost"].tolist(),
        "total_comm_cost_experiment": df_runs["total_comm_cost"].sum()
    },
    "performance_metrics": {
        "accuracy_per_run": df_runs["accuracy"].tolist(),
        "macro_f1_per_run": df_runs["macro_f1"].tolist(),
        "micro_f1_per_run": df_runs["micro_f1"].tolist(),
        "avg_accuracy": df_runs["accuracy"].mean(),
        "avg_macro_f1": df_runs["macro_f1"].mean(),
        "avg_micro_f1": df_runs["micro_f1"].mean()
    },
    "quality_metrics": {
        "spd_per_run": df_runs["spd"].tolist(),
        "eod_per_run": df_runs["eod"].tolist(),
        "avg_spd": df_runs["spd"].mean(),
        "avg_eod": df_runs["eod"].mean()
    }
}

print("\n=== Summary Metrics by Type ===")
for category, metrics in summary_by_type.items():
    print(f"\n{category.replace('_', ' ').title()}\n" + "="*30)
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

import matplotlib.pyplot as plt
import numpy as np

# FL run indices
runs = df_runs["run_id"].tolist()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Federated Learning Experiment Metrics", fontsize=16)

# ---------------- Performance metrics ----------------
ax = axes[0,0]
ax.plot(runs, df_runs["accuracy"], marker='o', label="Accuracy")
ax.plot(runs, df_runs["macro_f1"], marker='s', label="Macro F1")
ax.plot(runs, df_runs["micro_f1"], marker='^', label="Micro F1")
ax.set_xlabel("FL Run")
ax.set_ylabel("Performance")
ax.set_title("Performance Metrics")
ax.grid(True)
ax.legend()

# ---------------- Quality metrics ----------------
ax = axes[0,1]
ax.plot(runs, df_runs["spd"], marker='o', label="SPD")
ax.plot(runs, df_runs["eod"], marker='s', label="EOD")
ax.set_xlabel("FL Run")
ax.set_ylabel("Fairness")
ax.set_title("Quality Metrics (SPD / EOD)")
ax.grid(True)
ax.legend()

# ---------------- Cost metrics ----------------
ax = axes[1,0]
ax.bar(runs, df_runs["total_comm_cost"], color='orange')
ax.set_xlabel("FL Run")
ax.set_ylabel("Comm Cost")
ax.set_title("Cost Metrics")
ax.grid(True, axis='y')

# ---------------- Utility metrics ----------------
ax = axes[1,1]
ax.bar(runs, df_runs["runtime_sec"], color='green')
ax.set_xlabel("FL Run")
ax.set_ylabel("Runtime (sec)")
ax.set_title("Utility Metrics")
ax.grid(True, axis='y')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
