Federated Learning Experiment Pipeline
│
├── 1. Initialization Phase
│   │
│   ├── Define utility functions
│   │   ├── safe_read_csv()
│   │   ├── compute_spd()
│   │   ├── compute_eod()
│   │   └── add_fairness_metrics()
│   │
│   └── Define directory structure
│       ├── root_dir (/storage/fl-lab)
│       ├── data_dir
│       ├── config_dir
│       └── runs_dir
│
├── 2. Dataset Preparation
│   │
│   ├── Load Adult dataset
│   │   ├── adult.csv
│   │   └── dataframe df_adult
│   │
│   ├── Detect key dataset columns
│   │   ├── sensitive attribute → sex
│   │   └── target variable → income
│   │
│   ├── Convert column types if necessary
│   │   ├── sex → {Female, Male}
│   │   └── income → {<=50K, >50K}
│   │
│   └── Compute baseline fairness metrics
│       ├── SPD
│       └── EOD
│
├── 3. Federated Experiment Loop
│   │
│   ├── Runs = 3 experiments
│   │
│   ├── Run 1
│   │   └── local_epochs = 5
│   │
│   ├── Run 2
│   │   └── local_epochs = 10
│   │
│   └── Run 3
│       └── local_epochs = 20
│
├── 4. Per-Run Configuration
│   │
│   ├── Generate temporary YAML configuration
│   │   │
│   │   ├── Experiment YAML (tmp_exp_X.yaml)
│   │   │   ├── protocol.n_clients = 4
│   │   │   ├── protocol.n_rounds = 20
│   │   │   ├── dataset = Adult
│   │   │   ├── data distribution = Dirichlet
│   │   │   ├── beta = 1.0
│   │   │   └── log_dir = runs/run_X
│   │   │
│   │   └── Algorithm YAML (tmp_alg_X.yaml)
│   │       ├── model = Adult_LogReg
│   │       ├── input_dim = 14
│   │       └── client.local_epochs = {5,10,20}
│
├── 5. Federated Learning Execution
│   │
│   └── Launch FLuKE federation
│       └── command
│           fluke federation tmp_exp_X.yaml tmp_alg_X.yaml
│
├── 6. Internal Federated Training (FLuKE)
│   │
│   ├── Dataset partitioning
│   │   └── Dirichlet distribution (β=1.0)
│   │
│   ├── Create 4 clients (simulated hospitals)
│   │
│   └── Federated Training Rounds (20 rounds)
│       │
│       ├── Round t
│       │
│       ├── Server initialization
│       │   └── global model W_t
│       │
│       ├── Model broadcast
│       │   └── Server → Clients
│       │
│       ├── Local client training
│       │   │
│       │   ├── Hospital A
│       │   ├── Hospital B
│       │   ├── Hospital C
│       │   └── Hospital D
│       │
│       │   └── Each client trains
│       │       └── local_epochs passes over local dataset
│       │
│       ├── Local model updates
│       │   ├── W_A
│       │   ├── W_B
│       │   ├── W_C
│       │   └── W_D
│       │
│       ├── Upload model updates
│       │   └── Clients → Server
│       │
│       └── Server aggregation
│           └── FedAvg
│               W_(t+1) = average(W_A, W_B, W_C, W_D)
│
├── 7. Metrics Collection
│   │
│   ├── Read training outputs
│   │   ├── postfit_metrics.csv
│   │   └── comm_costs.csv
│   │
│   ├── Compute performance metrics
│   │   ├── accuracy
│   │   ├── macro_f1
│   │   └── micro_f1
│   │
│   ├── Compute fairness metrics
│   │   ├── SPD
│   │   └── EOD
│   │
│   └── Compute system metrics
│       ├── communication cost
│       └── runtime
│
├── 8. Per-Run Results
│   │
│   ├── Store results in results list
│   │
│   └── Print run summary
│       ├── Run ID
│       ├── Local epochs
│       ├── Accuracy
│       ├── Macro F1
│       ├── SPD
│       ├── EOD
│       └── Runtime
│
└── 9. Final Experiment Summary
    │
    ├── Create dataframe df_runs
    │
    ├── Compute averages
    │   ├── avg_accuracy
    │   ├── avg_macro_f1
    │   ├── avg_micro_f1
    │   ├── avg_spd
    │   ├── avg_eod
    │   ├── total_comm_cost
    │   └── runtime statistics
    │
    └── Print final experiment report
