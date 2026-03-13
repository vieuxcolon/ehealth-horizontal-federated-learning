# How Federated Learning Works in Details in Our Project - We compiled this information also for our own learning

## Overview

This document explains **how Federated Learning (FL) works step by step** using the experiment implemented in this project. The goal is to understand the **semantics of the federated learning process**, including the interactions between:

* the dataset
* the participating clients (hospitals)
* the central server
* communication between nodes
* local training epochs
* federated learning rounds
* full experiment runs

The project uses the **Adult dataset** and simulates **four hospitals collaboratively training a model without sharing patient data**.

---

# 1. Experimental Setup

## Dataset

The experiments use the **Adult dataset**, which contains demographic and employment information.

Dataset properties:

| Property            | Value    |
| ------------------- | -------- |
| Samples             | 27,145   |
| Features            | 14       |
| Target variable     | `income` |
| Sensitive attribute | `sex`    |

The dataset is used to predict whether an individual's income exceeds **$50K per year**.

---

## Federated Learning Configuration

| Parameter                     | Value               |
| ----------------------------- | ------------------- |
| Number of clients (hospitals) | 4                   |
| Algorithm                     | FedAvg              |
| FL rounds                     | 20                  |
| Local epochs                  | 5, 10, 20           |
| Data distribution             | Dirichlet (β = 1.0) |

The experiments are executed as **three independent runs**, each with a different number of local epochs.

---

# 2. Dataset Preparation Phase

## Step 1 — Dataset Loading

The system loads the Adult dataset:

```
27145 samples
14 features
target = income
sensitive attribute = sex
```

At this stage the dataset exists as a **centralized dataset only for experiment preparation**.

---

## Step 2 — Dataset Partitioning Across Hospitals

The dataset is then **distributed across four clients**, representing hospitals.

The distribution method used is:

```
Dirichlet distribution (β = 1.0)
```

This means:

* each hospital receives a **different subset of patients**
* all hospitals share the **same feature space**
* the statistical distribution of samples may differ slightly between hospitals

Example conceptual distribution:

| Hospital   | Samples |
| ---------- | ------- |
| Hospital A | ~6800   |
| Hospital B | ~6500   |
| Hospital C | ~7000   |
| Hospital D | ~6800   |

Each hospital **only sees its own patients**.

No hospital has access to the complete dataset.

---

# 3. Experiment Structure

The pipeline runs **three full federated learning experiments**.

| Run   | Local Epochs | FL Rounds |
| ----- | ------------ | --------- |
| Run 1 | 5            | 20        |
| Run 2 | 10           | 20        |
| Run 3 | 20           | 20        |

Each run represents a **complete federated training session**.

---

# 4. Federated Learning Initialization

Before training begins, the **server initializes the global model**.

Example model used in the experiment:

```
Logistic Regression
input_dim = 14
```

Initial model weights are created:

```
W0
```

The server sends this initial model to all clients.

---

# 5. Federated Training Process

Each experiment consists of **20 federated learning rounds**.

During each round, the clients and server collaborate to improve the global model.

---

# 6. Federated Round Workflow

A **federated learning round** consists of several steps.

---

## Step 1 — Server Broadcast

The server sends the current global model:

```
W_t
```

to all participating hospitals.

```
Hospital A
Hospital B
Hospital C
Hospital D
```

This is the **first communication event of the round**.

---

## Step 2 — Local Training at Hospitals

Each hospital trains the received model **using only its local dataset**.

Examples:

Hospital A trains on:

```
local dataset A
```

Hospital B trains on:

```
local dataset B
```

Each hospital performs local training independently.

---

## Step 3 — Local Epochs

Local training consists of several **local epochs**.

A **local epoch** means:

```
one full pass over the local dataset
```

Example for Run 1:

```
Local epochs = 5
```

This means each hospital performs:

```
5 full training passes over its local dataset
```

Run 2 performs:

```
10 passes
```

Run 3 performs:

```
20 passes
```

---

## Step 4 — Local Model Updates

After local training, each hospital obtains updated model weights:

```
Hospital A → W_A
Hospital B → W_B
Hospital C → W_C
Hospital D → W_D
```

These weights represent the **knowledge learned from each hospital’s local data**.

---

## Step 5 — Communication to Server

Hospitals send their model updates back to the central server.

Important:

Hospitals **do not send raw patient data**.

They only send:

```
model parameters (weights)
```

This is the **second communication event of the round**.

---

## Step 6 — Server Aggregation (FedAvg)

The server combines the models using **Federated Averaging (FedAvg)**.

Conceptually:

```
W_(t+1) = average(W_A, W_B, W_C, W_D)
```

This creates the **new global model**.

---

## Step 7 — Next Round Begins

The updated global model is then **broadcast again to the hospitals**.

This process repeats for:

```
20 rounds
```

---

# 7. End of Training

After completing all rounds, the **final global model** is obtained.

The model is then evaluated using several metrics.

---

# 8. Performance Metrics

The following metrics measure prediction performance:

| Metric   | Meaning                        |
| -------- | ------------------------------ |
| Accuracy | Overall prediction correctness |
| Macro F1 | Average F1 across classes      |
| Micro F1 | Global F1 score                |

Observed results:

| Local Epochs | Accuracy | Macro F1 |
| ------------ | -------- | -------- |
| 5            | 0.8738   | 0.7785   |
| 10           | 0.8711   | 0.7666   |
| 20           | 0.8686   | 0.7584   |

Interpretation:

Increasing local epochs slightly decreases global accuracy due to **client model divergence**.

---

# 9. Fairness Metrics

The experiments also measure **fairness across genders** using:

| Metric | Meaning                       |
| ------ | ----------------------------- |
| SPD    | Statistical Parity Difference |
| EOD    | Equal Opportunity Difference  |

Observed values:

```
SPD = -0.1991
EOD = -0.1991
```

Interpretation:

* The dataset contains inherent bias between male and female predictions.
* Standard federated learning **does not automatically correct dataset bias**.

---

# 10. Communication Cost

Communication cost measures the **amount of information exchanged between clients and server**.

Observed value:

```
4920 per run
```

Communication cost remains constant because:

* number of clients is fixed
* number of rounds is fixed
* model size is constant

---

# 11. Runtime

Runtime measures the **total computation time for each experiment**.

Example results:

| Local Epochs | Runtime     |
| ------------ | ----------- |
| 5            | ~69 seconds |
| 10           | ~70 seconds |
| 20           | ~98 seconds |

Runtime increases because **more local training occurs per round**.

---

# 12. Privacy Properties

Federated learning preserves privacy because:

* raw datasets remain **stored locally at hospitals**
* only **model parameters** are shared
* no patient records are transmitted

Conceptually:

```
Hospitals keep their data
↓
Local model training
↓
Only model weights are shared
↓
Server aggregates updates
```

This enables collaborative training without centralizing sensitive data.

---

# 13. Complete Training Timeline

The full experiment workflow is:

```
Dataset loaded
      ↓
Dataset split across 4 hospitals
      ↓
Run 1 begins
      ↓
Server initializes global model
      ↓
Round 1
   server → clients (model)
   clients train locally
   clients → server (weights)
   server aggregates
      ↓
Rounds repeat until Round 20
      ↓
Metrics computed
      ↓
Run 2 begins (10 epochs)
      ↓
Run 3 begins (20 epochs)
      ↓
Final experiment analysis
      ↓
Metrics visualization
```

---

# 14. Key Insights

The experiments demonstrate several core principles of federated learning:

### Data Locality

Hospitals keep their patient data locally.

### Collaborative Model Training

Multiple institutions jointly train a model without sharing raw data.

### Communication–Computation Tradeoff

More local epochs increase computation but may reduce global model quality.

### Bias Propagation

Federated learning alone **does not remove dataset bias**.

Fairness-aware algorithms are needed to mitigate this.

---

# 15. Conceptual Summary

The experiment simulates a **real-world collaborative healthcare training scenario**:

```
Hospital A (patients)
Hospital B (patients)
Hospital C (patients)
Hospital D (patients)

        ↓

Local training

        ↓

Server aggregation (FedAvg)

        ↓

Global predictive model
```

This allows institutions to **build a shared machine learning model without sharing sensitive data**.

---

# Conclusion

Federated learning enables **distributed collaborative training** across multiple institutions while preserving data locality.

The experiments illustrate:

* how models are trained across clients and server
* how communication and aggregation occur
* how performance, fairness, and runtime behave in practice

This setup provides a realistic demonstration of **federated learning in a multi-hospital environment**.
