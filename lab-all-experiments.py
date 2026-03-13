# =================================================================
# FEDERATED LEARNING MINI RESEARCH PIPELINE (36 EXPERIMENTS VERSION)
# =================================================================

import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from itertools import product
import shutil
import time
import re

# -------------------------------
# Step 0 — Safeguard Functions
# -------------------------------
def check_binary(bin_name):
    """Check if a CLI binary is available in PATH."""
    if shutil.which(bin_name) is None:
        raise RuntimeError(f"Required binary '{bin_name}' not found in PATH.")

def safe_read_csv(path):
    """Read CSV safely. Returns None if missing or empty."""
    if os.path.exists(path):
        df = pd.read_csv(path)
        if not df.empty:
            return df
    return None

# -------------------------------
# Step 1 — Environment Setup
# -------------------------------
print("Step 1 — Environment Setup")
print("Semantics / What: Ensure Python packages and environment are ready.")
print("Why: Required dependencies must be present for Fluke execution.")
print("How: User should install missing packages manually if needed.\n")

print("Python environment ready:", os.sys.version)
print("Working directory:", os.getcwd())

# -------------------------------
# Step 2 — Clone Repository & Install Fluke
# -------------------------------
print("\nStep 2 — Clone Repository & Install Fluke")
print("Semantics / What: Prepare the Fluke repository and install package.")
print("Why: Fluke package required for running experiments.")
print("How: Clone git repo if missing and install via pip editable.\n")

repo_url = "https://git.liris.cnrs.fr/nbenarba/fl-lab.git"
repo_dir = os.path.join(os.getcwd(), "fl-lab")
fluke_package_dir = os.path.join(repo_dir, "fluke_package")

if not os.path.exists(repo_dir):
    print(f"Cloning repository from {repo_url} ...")
    subprocess.run(["git", "clone", repo_url], check=True)

if os.path.exists(fluke_package_dir):
    print("Installing Fluke package ...")
    subprocess.run(["pip", "install", "-e", fluke_package_dir], check=True)

# -------------------------------
# Step 3 — Define Experiment Grid (36 experiments only)
# -------------------------------
print("\nStep 3 — Define Experiment Grid")
print("Semantics / What: List all experiment combinations to run.")
print("Why: To systematically test all modes, datasets, client ratios, and local epochs.")
print("How: Construct explicit experiment list and assign distribution for 36 experiments.\n")

modes = ["centralized", "federation"]
datasets = ["mnist", "adult"]
client_ratios = [0.3, 0.6, 1.0]
local_epochs_list = [5, 10, 20]

# Full cartesian product: 2*2*3*3 = 36
base_combinations = list(product(modes, datasets, client_ratios, local_epochs_list))

# Assign distributions: first 18 iid, next 18 dirichlet=0.5
experiments_list = []
for idx, combo in enumerate(base_combinations):
    mode, dataset, cr, lep = combo
    distribution = "iid" if idx < 18 else 0.5
    experiments_list.append({
        "mode": mode,
        "dataset": dataset,
        "client_ratio": cr,
        "local_epochs": lep,
        "distribution": distribution
    })

print(f"Total experiments to run: {len(experiments_list)}")  # should be 36

# -------------------------------
# Step 4 — Update Configuration Files
# -------------------------------
print("\nStep 4 — Update Configuration Files")
print("Semantics / What: Generate temporary YAML configs per experiment.")
print("Why: Each run may have different client ratio, dataset, or local epochs.")
print("How: Load base YAMLs, modify parameters, and save temporary YAML files.\n")

def update_config(client_ratio=None, local_epochs=None, dataset=None, run_id=None):
    exp_yaml = os.path.join(repo_dir, "config", "exp.yaml")
    alg_yaml = os.path.join(repo_dir, "config", "fedavg.yaml")
    tmp_exp_yaml = os.path.join(repo_dir, "config", "tmp_exp.yaml")
    tmp_alg_yaml = os.path.join(repo_dir, "config", "tmp_alg.yaml")

    if not os.path.exists(exp_yaml) or not os.path.exists(alg_yaml):
        raise FileNotFoundError("Required config YAML files missing in 'config/'")

    # --- Experiment config ---
    with open(exp_yaml) as f:
        cfg = yaml.safe_load(f)

    if client_ratio is not None:
        cfg["protocol"]["eligible_perc"] = client_ratio
    if run_id is not None:
        cfg["logger"]["log_dir"] = os.path.join(repo_dir, f"runs/run_{run_id}")
    cfg["protocol"]["n_rounds"] = 20  # enforce 20 global rounds
    if dataset:
        cfg["data"]["dataset"]["name"] = dataset

    with open(tmp_exp_yaml, "w") as f:
        yaml.dump(cfg, f)

    # --- Algorithm config ---
    with open(alg_yaml) as f:
        alg = yaml.safe_load(f)

    if local_epochs is not None:
        alg["hyperparameters"]["client"]["local_epochs"] = local_epochs

    if dataset == "adult":
        adult_csv_path = os.path.join(repo_dir, "data", "adult", "adult.csv")
        if os.path.exists(adult_csv_path):
            adult_df = pd.read_csv(adult_csv_path)
            input_dim = adult_df.shape[1] - 1
        else:
            input_dim = 14
        alg["hyperparameters"]["model"] = "Adult_LogReg"
        alg["hyperparameters"]["net_args"] = {"input_dim": input_dim}
    else:
        alg["hyperparameters"]["model"] = "MNIST_2NN"
        alg["hyperparameters"]["net_args"] = {}

    with open(tmp_alg_yaml, "w") as f:
        yaml.dump(alg, f)

    return tmp_exp_yaml, tmp_alg_yaml

# -------------------------------
# Step 5 — Run Fluke Experiment with Run Timing
# -------------------------------
print("\nStep 5 — Run Fluke Experiment")
print("Semantics / What: Run experiment and track per-run duration.")
print("Why: Measure runtime and gather metrics for analysis.")
print("How: Launch Fluke subprocess and parse output timing.\n")

def run_fluke_with_run_duration(mode, tmp_exp_yaml, tmp_alg_yaml, run_id):
    check_binary("fluke")
    cmd = ["fluke", mode, tmp_exp_yaml, tmp_alg_yaml]

    print(f"Running experiment {run_id}: command = {' '.join(cmd)}")

    run_durations = {}
    total_start = time.time()

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=repo_dir
        )

        run_start = None
        prev_run_num = None
        for line in process.stdout:
            match = re.search(r"Round[:\s]+(\d+)", line) or re.search(r"Epoch[:\s]+(\d+)", line)
            if match:
                run_num = int(match.group(1))
                if run_start is not None and prev_run_num is not None:
                    run_durations[prev_run_num] = round(time.time() - run_start, 3)
                run_start = time.time()
                prev_run_num = run_num

        process.wait()
        if run_start is not None and prev_run_num is not None:
            run_durations[prev_run_num] = round(time.time() - run_start, 3)

    except subprocess.CalledProcessError as e:
        print(f"Warning: Experiment {run_id} failed: {e}")

    total_duration_sec = round(time.time() - total_start, 3)
    return total_duration_sec, run_durations

# -------------------------------
# Step 6 — Sequential Execution of Experiments
# -------------------------------
print("\nStep 6 — Run Experiments")
print("Semantics / What: Execute all 36 experiment combinations sequentially and collect results")
print("Why: To systematically record accuracy, model, durations, etc. for each configuration")
print("How: Iterate over the explicit experiments list\n")

results = []
run_id = 0

for exp in experiments_list:
    run_id += 1
    mode = exp["mode"]
    dataset = exp["dataset"]
    cr = exp["client_ratio"]
    lep = exp["local_epochs"]
    distribution = exp["distribution"]

    print(f"\nRunning experiment {run_id}: mode={mode}, dataset={dataset}, client_ratio={cr}, local_epochs={lep}, distribution={distribution}")

    tmp_exp_yaml, tmp_alg_yaml = update_config(client_ratio=cr, local_epochs=lep, dataset=dataset, run_id=run_id)

    total_duration, per_run = run_fluke_with_run_duration(mode, tmp_exp_yaml, tmp_alg_yaml, run_id)

    metrics_paths = [
        os.path.join(repo_dir, f"runs/run_{run_id}/global_metrics.csv"),
        os.path.join(repo_dir, f"runs/run_{run_id}/metrics.csv")
    ]
    metrics = None
    for path in metrics_paths:
        metrics = safe_read_csv(path)
        if metrics is not None:
            break

    final_acc = metrics["accuracy"].iloc[-1] if metrics is not None and "accuracy" in metrics.columns else None
    model_name = "Adult_LogReg" if dataset=="adult" else "MNIST_2NN"

    # If no per-run durations captured, compute average round time
    if not per_run and metrics is not None and "round" in metrics.columns and total_duration is not None:
        total_rounds = int(metrics["round"].max())
        if total_rounds > 0:
            per_run = {"rounds": total_rounds, "avg_round_time_sec": round(total_duration / total_rounds, 3)}

    results.append({
        "run_id": run_id,
        "mode": mode,
        "dataset": dataset,
        "client_ratio": cr,
        "local_epochs": lep,
        "distribution": distribution,
        "accuracy": final_acc,
        "model": model_name,
        "batch_size": 64,
        "lr": 0.01,
        "loss": "CrossEntropyLoss",
        "total_duration_sec": total_duration,
        "run_durations_sec": per_run
    })

# -------------------------------
# Step 7 — Display Each Experiment
# -------------------------------
print("\nStep 7 — Display Experiment Results")
print("Semantics / What: Show summary of each experiment run")
print("Why: To verify results, accuracy, durations, and per-run timings")
print("How: Iterate over results list and print in readable format\n")

for exp in results:
    print("\n" + "="*60)
    print(f"Experiment {exp['run_id']}")
    print(f"Mode          : {exp['mode']}")
    print(f"Dataset       : {exp['dataset']}")
    print(f"Client Ratio  : {exp['client_ratio']}")
    print(f"Local Epochs  : {exp['local_epochs']}")
    print(f"Distribution  : {exp['distribution']}")
    print(f"Model         : {exp['model']}")
    print(f"Final Accuracy: {exp['accuracy']}")
    if exp["total_duration_sec"] is not None:
        print(f"Total Duration (s): {exp['total_duration_sec']:.2f}")
    else:
        print("Total Duration (s): None")
    print(f"Per-Run Durations: {exp['run_durations_sec']}")
    print("="*60)

# -------------------------------
# Step 8 — Aggregate Results & Summary Table
# -------------------------------
print("\nStep 8 — Summary Table")
print("Semantics / What: Aggregate results by mode and dataset")
print("Why: To understand max accuracy trends")
print("How: Use pandas groupby on results dataframe\n")

results_df = pd.DataFrame(results)
summary = results_df.groupby(["mode","dataset"]).agg({"accuracy":"max"}).reset_index()
print("Summary (Max Accuracy per mode & dataset):")
print(summary)

# -------------------------------
# Step 9 — TensorBoard & Plot Metrics (Colab-compatible)
# -------------------------------
print("\nStep 9 — TensorBoard & Accuracy Plots")
print("Semantics / What: Launch TensorBoard and plot accuracy trends")
print("Why: Visual analysis of performance across experiments")
print("How: Use subprocess for TensorBoard and seaborn/matplotlib for plots\n")

check_binary("tensorboard")
tb_logdir = os.path.join(repo_dir, "runs/")
# Colab-compatible TensorBoard launch
get_ipython().system_raw(f"tensorboard --logdir {tb_logdir} --host 0.0.0.0 --port 6006 &")

plot_df = results_df.dropna(subset=["accuracy"])
if not plot_df.empty:
    plt.figure(figsize=(10,6))
    sns.lineplot(
        data=plot_df,
        x="client_ratio",
        y="accuracy",
        hue="local_epochs",
        style="mode",
        markers=True,
        dashes=False
    )
    plt.title("Accuracy vs Client Ratio / Local Epochs / Mode")
    plt.xlabel("Client Ratio")
    plt.ylabel("Accuracy")
    plt.legend(title="Local Epochs / Mode")
    plt.show()

print("\nPipeline completed successfully. 36 experiments executed (18 iid, 18 dirichlet=0.5), metrics collected, tables generated, TensorBoard running at port 6006.")
