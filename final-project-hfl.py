# ===============================================================
# Adult HFL Pipeline — Multi-Experiment, IID vs Non-IID
# Computes Accuracy, F1, SPD, EOD, Communication, Runtime per FL Round
# Fluke messages are suppressed
# ===============================================================

import os
import subprocess
import pandas as pd
import yaml
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Step 0 — Utility Functions
# -------------------------------
def check_binary(bin_name):
    if shutil.which(bin_name) is None:
        raise RuntimeError(f"Required binary '{bin_name}' not found in PATH.")

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
    if sensitive_col not in df.columns or target_col not in df.columns:
        return None
    priv = df[df[sensitive_col] == "Male"]
    unpriv = df[df[sensitive_col] == "Female"]
    if len(priv) == 0 or len(unpriv) == 0:
        return None
    p_priv = (priv[target_col] == ">50K").mean()
    p_unpriv = (unpriv[target_col] == ">50K").mean()
    return round(p_unpriv - p_priv, 4)

def compute_eod(df, sensitive_col="sex", target_col="income"):
    if sensitive_col not in df.columns or target_col not in df.columns:
        return None
    priv = df[df[sensitive_col] == "Male"]
    unpriv = df[df[sensitive_col] == "Female"]
    if len(priv) == 0 or len(unpriv) == 0:
        return None
    tpr_priv = (priv[target_col] == ">50K").mean()
    tpr_unpriv = (unpriv[target_col] == ">50K").mean()
    return round(tpr_unpriv - tpr_priv, 4)

# -------------------------------
# Step 1 — Directories and Repo
# -------------------------------
root_dir = "/content"
repo_dir = os.path.join(root_dir, "fl-lab")
data_dir = os.path.join(repo_dir, "data")
config_dir = os.path.join(repo_dir, "config")
runs_dir = os.path.join(repo_dir, "runs")
adult_csv = os.path.join(data_dir, "adult", "adult.csv")
repo_url = "https://git.liris.cnrs.fr/nbenarba/fl-lab.git"

if not os.path.exists(repo_dir):
    print(f"Cloning FLuKE repo from {repo_url}")
    subprocess.run(["git", "clone", repo_url, repo_dir], check=True)

# -------------------------------
# Step 2 — Load Adult Dataset
# -------------------------------
if os.path.exists(adult_csv):
    df_adult = pd.read_csv(adult_csv)
else:
    raise FileNotFoundError(f"Adult dataset not found at {adult_csv}")

# Detect columns
sensitive_candidates = ["sex", "gender"]
target_candidates = ["income", "target", "label", "class"]
sensitive_col = next((c for c in sensitive_candidates if c in df_adult.columns), None)
target_col = next((c for c in target_candidates if c in df_adult.columns), None)

if df_adult[sensitive_col].dtype != object:
    df_adult[sensitive_col] = df_adult[sensitive_col].map({0:"Female",1:"Male"})
if df_adult[target_col].dtype != object:
    df_adult[target_col] = df_adult[target_col].map({0:"<=50K",1:">50K"})

# -------------------------------
# Step 3 — Update YAML per Run
# -------------------------------
def update_tmp_yaml(run_id, local_epochs=5, iid=True):
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
    exp_cfg["data"]["distribution"]["name"] = "dir" if not iid else "iid"
    exp_cfg["data"]["distribution"]["beta"] = 1.0
    exp_cfg["compute_fairness_metrics"] = True

    with open(tmp_exp_yaml, "w") as f:
        yaml.dump(exp_cfg, f)

    # Algorithm YAML
    with open(alg_yaml) as f:
        alg_cfg = yaml.safe_load(f)
    input_dim = df_adult.shape[1]-1
    alg_cfg["hyperparameters"]["client"]["local_epochs"] = local_epochs
    alg_cfg["hyperparameters"]["model"] = "Adult_LogReg"
    alg_cfg["hyperparameters"]["net_args"] = {"input_dim": input_dim}

    with open(tmp_alg_yaml, "w") as f:
        yaml.dump(alg_cfg, f)

    return tmp_exp_yaml, tmp_alg_yaml

# -------------------------------
# Step 4 — Run FLuKE (suppress messages)
# -------------------------------
def run_fluke(run_id, tmp_exp_yaml, tmp_alg_yaml):
    check_binary("fluke")
    cmd = ["fluke", "federation", tmp_exp_yaml, tmp_alg_yaml]
    start_time = time.time()
    subprocess.run(cmd, cwd=repo_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return round(time.time() - start_time, 3)

# -------------------------------
# Step 5 — Add fairness metrics
# -------------------------------
def add_fairness_metrics(postfit_df, df_full, compute_fairness=True):
    if not compute_fairness or postfit_df is None:
        return postfit_df
    postfit_df["spd"] = [compute_spd(df_full, sensitive_col, target_col)]*len(postfit_df)
    postfit_df["eod"] = [compute_eod(df_full, sensitive_col, target_col)]*len(postfit_df)
    return postfit_df

# -------------------------------
# Step 6 — Execute experiments
# -------------------------------
def execute_experiment(run_id, local_epochs=5, iid=True):
    tmp_exp_yaml, tmp_alg_yaml = update_tmp_yaml(run_id, local_epochs, iid)
    runtime = run_fluke(run_id, tmp_exp_yaml, tmp_alg_yaml)
    postfit_path = os.path.join(runs_dir, f"run_{run_id}/postfit_metrics.csv")
    comm_path = os.path.join(runs_dir, f"run_{run_id}/comm_costs.csv")
    postfit = safe_read_csv(postfit_path)
    comm = safe_read_csv(comm_path)
    postfit = add_fairness_metrics(postfit, df_adult)
    results = {
        "run_id": run_id,
        "local_epochs": local_epochs,
        "accuracy_per_round": postfit["accuracy"].tolist() if postfit is not None else None,
        "macro_f1_per_round": postfit["macro_f1"].tolist() if postfit is not None else None,
        "micro_f1_per_round": postfit["micro_f1"].tolist() if postfit is not None else None,
        "spd_per_round": postfit["spd"].tolist() if postfit is not None else None,
        "eod_per_round": postfit["eod"].tolist() if postfit is not None else None,
        "total_comm_cost": comm["comm_costs"].sum() if comm is not None else None,
        "runtime_sec": runtime
    }
    return results

# -------------------------------
# Step 7 — Define Experiments
# -------------------------------
experiment_params = [
    {"run_id": 1, "local_epochs": 5},
    {"run_id": 2, "local_epochs": 10},
    {"run_id": 3, "local_epochs": 20},
    {"run_id": 4, "local_epochs": 5},
    {"run_id": 5, "local_epochs": 10},
    {"run_id": 6, "local_epochs": 20},
]

results_iid, results_noniid = [], []

for params in experiment_params:
    print(f"\nExecuting IID experiment {params['run_id']}")
    results_iid.append(execute_experiment(params['run_id'], params['local_epochs'], iid=True))
    print(f"\nExecuting non-IID experiment {params['run_id']}")
    results_noniid.append(execute_experiment(params['run_id'], params['local_epochs'], iid=False))

# -------------------------------
# Step 8 — Plot per-Experiment Metrics vs FL rounds
# -------------------------------
metrics = ["accuracy_per_round", "macro_f1_per_round", "micro_f1_per_round", "spd_per_round", "eod_per_round"]

for exp_iid, exp_noniid in zip(results_iid, results_noniid):
    rounds = np.arange(1, len(exp_iid["accuracy_per_round"])+1)
    fig, axes = plt.subplots(3,2, figsize=(14,12))
    fig.suptitle(f"Experiment {exp_iid['run_id']} (Local Epochs={exp_iid['local_epochs']})\nIID vs Non-IID", fontsize=16)

    for ax, metric in zip(axes.flatten(), metrics):
        ax.plot(rounds, exp_iid[metric], label="IID", color='blue')
        ax.plot(rounds, exp_noniid[metric], label="Non-IID", color='red', alpha=0.7)
        ax.set_xlabel("FL Round")
        ax.set_ylabel(metric.replace("_per_round","").upper())
        ax.set_title(f"{metric.replace('_per_round','').upper()} vs FL Rounds")
        ax.grid(True)
        ax.legend()

    for ax in axes.flatten()[len(metrics):]:
        ax.axis('off')
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.show()
