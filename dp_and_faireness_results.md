---

# Experiment related to the final Project: Analysis of Differential Privacy and Fairness on UCI Breast Cancer.

This experiment modifies one of our code base to implement **Vertical (VFL) and Horizontal Federated Learning (HFL) pipelines** with **differential privacy** and **fairness evaluation** on the **UCI Breast Cancer dataset**.

# Remark: We used documents formatted in this Summary style to also serve as learning aid in our FL Programme. 
---

## **Metric Definitions**

### **Differential Privacy (DP) Metrics**

* **DP Alpha (α):** Privacy parameter controlling the amount of noise added to gradients during training.

  * **Higher α → stronger privacy → more noise → potential decrease in model accuracy.**
* **Test Accuracy:** Standard classification accuracy on the held-out test set; used to measure **utility** under different DP levels.

### **Fairness Metrics**

* **PosRate_Young / PosRate_Old:** Fraction of positive predictions (e.g., predicting cancer) for each demographic group (young vs old).

  * Measures whether the model **predicts differently for different age groups**.
* **Demographic Parity Difference (DPD):** Difference between positive rates of age groups (`PosRate_Young – PosRate_Old`).

  * A value near **0** indicates that predictions are **fair across age groups**.
  * Positive/negative values indicate bias toward one group.

> Remark: Differential privacy affects training noise and accuracy, while fairness metrics evaluate whether this noise disproportionately affects certain demographic groups.

---

## **1. Differential Privacy (DP) Analysis**

Differential privacy is applied by adding **noise to gradients** during training, with varying levels of privacy (`alpha = 0.0, 0.01, 0.1`). This section focuses on **model utility under DP**.

### **VFL Results (Feature Partitioned)**

| DP_alpha | Test Accuracy |
| -------- | ------------- |
| 0.00     | 0.930         |
| 0.01     | 0.921         |
| 0.10     | 0.895         |

* **Observation:**

  * Accuracy decreases as DP noise increases.
  * VFL models remain robust and converge smoothly.

### **HFL Results (Sample Partitioned)**

| DP_alpha | Test Accuracy |
| -------- | ------------- |
| 0.00     | 0.956         |
| 0.01     | 0.956         |
| 0.10     | 0.921         |

* **Observation:**

  * High DP noise (α = 0.1) introduces moderate accuracy degradation.
  * Training is stable and reproducible across DP levels.

---

## **2. Fairness Evaluation**

Fairness is measured using the **synthetic age attribute** with metrics defined above.

### **VFL Fairness Metrics**

| DP_alpha | PosRate_Young | PosRate_Old | Demographic Parity Diff |
| -------- | ------------- | ----------- | ----------------------- |
| 0.00     | 0.286         | 0.339       | -0.053                  |
| 0.01     | 0.286         | 0.323       | -0.037                  |
| 0.10     | 0.225         | 0.292       | -0.068                  |

* **Observation:** Minor differences across age groups; fairness remains acceptable under all DP levels.

### **HFL Fairness Metrics**

| DP_alpha | Acc_Young | Acc_Old | Demographic Parity Diff |
| -------- | --------- | ------- | ----------------------- |
| 0.00     | 0.980     | 0.938   | 0.041                   |
| 0.01     | 0.980     | 0.938   | 0.041                   |
| 0.10     | 0.898     | 0.938   | -0.041                  |

* **Observation:**

  * Demographic parity differences are small.
  * Fairness remains stable even with high DP noise.

---

## **Conclusions**

* **Differential Privacy:** Increasing α reduces accuracy slightly, but pipelines remain robust and reproducible.
* **Fairness:** Small demographic parity differences indicate models are **fair across age groups**, even under DP.
