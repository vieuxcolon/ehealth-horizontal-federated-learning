---

# Federated Learning Experiment — Metrics and Plots

This document explains the metrics collected during federated learning experiments and the corresponding plots used to visualize performance, fairness, communication cost, and runtime.

---

## 1. **Overview of Plots**

The experiments generate a **2×2 grid of plots** representing different categories of metrics across FL runs:

* **X-axis:** FL run index (`run_id`) representing each separate experiment.
* **Y-axis:** Metric-specific values for each category.
* **Purpose:** Allows comparison of metrics across different federated learning runs.

The four categories are:

1. **Performance Metrics**
2. **Quality / Fairness Metrics**
3. **Cost / Communication Metrics**
4. **Utility / Runtime Metrics**

---

## 2. **Performance Metrics (Top-left subplot)**

* **Metrics plotted:**

  * **Accuracy** – Overall correctness of model predictions.
  * **Macro F1** – Balances precision and recall across classes; important for imbalanced datasets.
  * **Micro F1** – Aggregates contributions from all instances; overall model performance.

* **Why it matters in FL:**

  * Evaluates the **global model performance** after aggregating local updates.
  * Shows the effect of different **local epochs** and training strategies.
  * Helps identify **overfitting or underfitting trends** due to federated training.

---

## 3. **Quality / Fairness Metrics (Top-right subplot)**

* **Metrics plotted:**

  * **SPD (Statistical Parity Difference)** – Difference in positive outcome rates between unprivileged and privileged groups.
  * **EOD (Equal Opportunity Difference)** – Difference in true positive rates between unprivileged and privileged groups.

* **Why it matters in FL:**

  * Ensures the **federated model does not introduce or amplify bias** across clients.
  * Evaluates fairness **per run**, which is crucial when combining data from multiple sources.
  * Helps monitor equity in predictions across sensitive features like `sex`.

---

## 4. **Cost / Communication Metrics (Bottom-left subplot)**

* **Metrics plotted:**

  * **Total communication cost per run** (sum of client-to-server and server-to-client exchanges).

* **Why it matters in FL:**

  * Communication is often the **primary bottleneck** in federated learning.
  * Useful to **analyze trade-offs** between local computation and network usage.
  * Helps plan deployment strategies across distributed clients.

---

## 5. **Utility / Runtime Metrics (Bottom-right subplot)**

* **Metrics plotted:**

  * **Runtime per run** – Total time taken for each federated learning experiment.

* **Why it matters in FL:**

  * Monitors computational efficiency.
  * Important for **scalability**, especially in distributed or resource-constrained environments.
  * Helps assess the impact of **local epochs or client count** on training duration.

---

## 6. **Summary of Relevance**

| Plot                       | Importance in Federated Learning                                    |
| -------------------------- | ------------------------------------------------------------------- |
| Performance Metrics        | Evaluates global model quality after aggregation.                   |
| Quality / Fairness Metrics | Ensures predictions remain fair across sensitive groups.            |
| Communication Cost         | Monitors network efficiency; crucial for FL scalability.            |
| Runtime                    | Measures computational efficiency and informs deployment decisions. |

---
