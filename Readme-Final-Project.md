---
# Using the Fluke Framework for Horizontal Federated Learning with 4 Hospitals

This project demonstrates the application of the **Fluke framework** to perform **horizontal federated learning (HFL)** across 4 hospitals using the Adult dataset. The focus is on model performance, fairness metrics, and resource usage under different local training configurations.

---

## 1. Project Overview

- **Framework:** FLuKE (Federated Learning Framework)  
- **Scenario:** Horizontal Federated Learning (HFL) across 4 clients (hospitals)  
- **Dataset:** Adult dataset (~27,145 samples, 14 features)  
- **Sensitive Attribute:** `sex`  
- **Target:** `income` (<=50K or >50K)  
- **Federated Algorithm:** FedAvg (federated averaging)  
- **Fairness Metrics:** Statistical Parity Difference (SPD) & Equal Opportunity Difference (EOD)  

**Objective:** Evaluate federated learning performance and fairness, while measuring runtime and communication costs.

---

## 2. Experiments Setup

- **Number of clients (hospitals):** 4  
- **Data distribution:** Dirichlet (β = 1.0) → balanced but non-iid splits  
- **Local epochs per client:** 5, 10, 20  
- **Federated rounds:** 20  
- **Model:** Logistic Regression (Adult_LogReg)  
- **Input features:** 14  

### Running Experiments

```

Running experiment 1: fluke federation /storage/fl-lab/config/tmp_exp_1.yaml /storage/fl-lab/config/tmp_alg_1.yaml
Parameters -> Local epochs: 5, Clients: 4, Rounds: 20, Distribution: Dirichlet (beta=1.0)

Running experiment 2: fluke federation /storage/fl-lab/config/tmp_exp_2.yaml /storage/fl-lab/config/tmp_alg_2.yaml
Parameters -> Local epochs: 10, Clients: 4, Rounds: 20, Distribution: Dirichlet (beta=1.0)

Running experiment 3: fluke federation /storage/fl-lab/config/tmp_exp_3.yaml /storage/fl-lab/config/tmp_alg_3.yaml
Parameters -> Local epochs: 20, Clients: 4, Rounds: 20, Distribution: Dirichlet (beta=1.0)

```

---

## 3. Dataset and Baseline Fairness

- **Adult dataset loaded:** 27,145 samples, 14 features  
- **Sensitive column:** `sex`  
- **Target column:** `income`  
- **Baseline SPD:** -0.1991  
- **Baseline EOD:** -0.1991  

> The dataset exhibits a bias against the privileged group (Male) for high-income prediction.

---

## 4. Results Analysis

### Performance Metrics Per Run

| Run | Local Epochs | Accuracy | Macro F1 | SPD    | EOD    | Runtime (s) |
|-----|--------------|----------|----------|--------|--------|-------------|
| 1   | 5            | 0.8738   | 0.7785   | -0.1991 | -0.1991 | 46.8        |
| 2   | 10           | 0.8711   | 0.7666   | -0.1991 | -0.1991 | 150.0       |
| 3   | 20           | 0.8686   | 0.7584   | -0.1991 | -0.1991 | 174.5       |

**Observations:**
- Increasing local epochs slightly decreases accuracy and macro F1 → minor overfitting at client level.  
- SPD and EOD remain constant, showing that federated averaging alone does not mitigate baseline bias.  
- Runtime increases with local epochs due to more local computation.

### Aggregated Metrics

| Metric               | Value |
|----------------------|-------|
| Avg Accuracy         | 0.8712 |
| Avg Macro F1         | 0.7678 |
| Avg Micro F1         | 0.8712 |
| Avg SPD              | -0.1991 |
| Avg EOD              | -0.1991 |
| Total Communication Cost | 14,760 |
| Avg Runtime (s)      | 123.78 |
| Total Runtime (s)    | 371.33 |

**Analysis:**
1. **Model Utility:** High predictive performance maintained across all experiments. Slight drop with higher local epochs due to local overfitting.  
2. **Fairness:** SPD/EOD constant → additional fairness-aware strategies needed for bias mitigation.  
3. **Cost & Runtime:** Communication cost fixed; runtime increases with local epochs as expected.

---

## 5. Conclusion

The experiments show that **horizontal federated learning using Fluke** can:

- Achieve **high model utility** while preserving **data privacy** across 4 hospitals.  
- Maintain **baseline fairness characteristics**, but federated averaging alone does not reduce bias.  
- Require a **trade-off** between local computation (epochs) and runtime.  

> Future work can explore fairness-aware federated algorithms to reduce SPD/EOD while retaining performance.

---

## 6. References

- [FLuKE Framework](https://git.liris.cnrs.fr/nbenarba/fl-lab)  
- Adult dataset: UCI Machine Learning Repository  

```
