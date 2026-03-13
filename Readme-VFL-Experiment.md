---

# Vertical Federated Learning (VFL) with Differential Privacy – Results Summary

This repository implements a **research-grade Vertical Federated Learning (VFL) pipeline** with differential privacy (DP) and fairness evaluation, applied to the **UCI Breast Cancer dataset**.

---

## **Problem Statement**

The goal of this study is to evaluate **vertical federated learning**, where multiple clients each own **different feature subsets of the same samples**. We aim to:

1. Train neural network models collaboratively across clients.
2. Incorporate **differential privacy** to analyze the privacy–utility tradeoff.
3. Evaluate **fairness** across a synthetic demographic attribute (`age`).

---

## **Pipeline Overview**

1. **Step 1: Imports & Reproducibility**

   * **What:** Load ML, DP, and plotting libraries
   * **Why:** Ensure reproducible experiments
   * **How:** Set seeds + import `numpy`, `pandas`, `torch`, `sklearn`, `matplotlib`

2. **Step 2: Load Dataset**

   * **What:** Load UCI Breast Cancer dataset with headers
   * **Why:** Structured dataset required for VFL pipeline
   * **How:** Assign column names manually
   * **Dataset shape:** `(569, 32)`

3. **Step 3: Synthetic Sensitive Attribute**

   * **What:** Add `age` (25–80)
   * **Why:** Evaluate fairness across age groups
   * **How:** Random integer assignment

4. **Step 4: Train/Test Split + Standardization**

   * **What:** Stratified split and feature normalization
   * **Why:** Prevent leakage and scale sensitivity
   * **How:** `train_test_split` + `StandardScaler`

5. **Step 5: Vertical Feature Partitioning**

   * **What:** Split feature columns across 4 clients
   * **Why:** Simulate VFL scenario (clients own complementary features)
   * **How:** Evenly split columns among clients

6. **Step 6: Model Factory Initialization**

   * **What:** Create fresh client and server models per experiment
   * **Why:** Avoid weight leakage across DP runs
   * **How:** Define a factory function returning models for each client and the server

7. **Step 7: Training & Evaluation Functions**

   * **What:** Modular forward, train, and evaluate
   * **Why:** Maintain clean experiment isolation
   * **How:** Client embeddings → concatenation → server → compute BCELoss → backprop

8. **Step 8: Differential Privacy Sweep**

   * **What:** Compare utility and fairness across DP levels (`alpha = 0.0, 0.01, 0.1`)
   * **Why:** Study privacy–utility–fairness tradeoff
   * **How:** Train fresh models per alpha and store histories

---

## **Training Results**

**DP alpha = 0.0**

| Epoch | Train Acc | Test Acc |
| ----- | --------- | -------- |
| 1     | 0.8176    | 0.7982   |
| 5     | 0.8901    | 0.8684   |
| 10    | 0.9165    | 0.8947   |
| 15    | 0.9275    | 0.9211   |
| 20    | 0.9451    | 0.9298   |

**DP alpha = 0.01**

| Epoch | Train Acc | Test Acc |
| ----- | --------- | -------- |
| 1     | 0.3297    | 0.2544   |
| 5     | 0.8945    | 0.8596   |
| 10    | 0.9275    | 0.9211   |
| 15    | 0.9363    | 0.9123   |
| 20    | 0.9407    | 0.9211   |

**DP alpha = 0.1**

| Epoch | Train Acc | Test Acc |
| ----- | --------- | -------- |
| 1     | 0.2110    | 0.2632   |
| 5     | 0.7319    | 0.7632   |
| 10    | 0.8527    | 0.8596   |
| 15    | 0.8747    | 0.8860   |
| 20    | 0.8813    | 0.8947   |

---

## **Step 9: Summary Table**

| DP_alpha | Final_Test_Accuracy | PosRate_Young | PosRate_Old | Demographic_Parity_Diff |
| -------- | ------------------- | ------------- | ----------- | ----------------------- |
| 0.00     | 0.9298              | 0.2857        | 0.3385      | -0.0527                 |
| 0.01     | 0.9211              | 0.2857        | 0.3231      | -0.0374                 |
| 0.10     | 0.8947              | 0.2245        | 0.2923      | -0.0678                 |

* **Observation:**

  * Higher DP levels (α = 0.1) slightly reduce final accuracy.
  * Demographic parity differences are small, indicating fair predictions across age groups.

---

## **Step 10: Plot Learning Curves**

* **What:** Visualize test accuracy over epochs
* **Why:** Compare DP impact on convergence and performance
* **How:** Use stored histories (no retraining required)

---

## **Analysis**

1. **Utility:**

   * VFL achieves high test accuracy (≈0.93) without DP.
   * Small drop with moderate DP (α = 0.01) and larger drop at high DP (α = 0.1).

2. **Fairness:**

   * Fairness metrics (`PosRate_Young`, `PosRate_Old`, `Demographic_Parity_Diff`) show minor differences between demographic groups, consistent across DP levels.

3. **Stability:**

   * Training dynamics are smooth; no NaN issues or dimension mismatches were observed (unlike HFL before fixes).

---

## **Conclusion**

The **VFL pipeline is robust and reproducible**:

* Correctly handles **feature partitioning across clients**.
* Supports **differential privacy** with gradient noise.
* Computes **fairness metrics reliably**.
* Provides **publication-ready results** and **visualizations** for learning dynamics.

---

Do you want me to do that next?
