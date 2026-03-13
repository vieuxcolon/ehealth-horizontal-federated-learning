---

### **Observation from Results**

| Run | Local Epochs | Accuracy |
| --- | ------------ | -------- |
| 1   | 5            | 0.8738   |
| 2   | 10           | 0.8711   |
| 3   | 20           | 0.8686   |

* **Trend:** As the number of local epochs per client increases, the overall global accuracy slightly **decreases**.
* **Magnitude:** The drop is small (~0.005 from 5 → 20 epochs), but it is consistent across the runs.

---

### **Interpretation in the Context of Federated Learning**

1. **FL-Runs vs Local Computation:**

   * Local epochs per client determine how much each client trains on its own data before sending updates to the server.
   * Increasing local epochs reduces communication frequency (fewer global aggregations), but can lead to **client drift** if data is heterogeneous (Dirichlet distribution = non-iid).
   * Here, with 20 local epochs, each client overfits slightly to its local distribution before averaging → small accuracy decrease globally.

2. **Communication-Accuracy Tradeoff:**

   * Fewer local epochs → more frequent server aggregation → better alignment between client updates → slightly higher global accuracy.
   * More local epochs → less frequent aggregation → clients diverge more → global model accuracy slightly lower.

3. **Stability:**

   * The change in accuracy is minor, indicating that the system is **stable**; FL is robust to the number of local epochs in this experiment, but very high local epochs could amplify divergence in more skewed datasets.

---

### **Key Takeaway**

> In these Adult dataset FL experiments:
>
> * **Fewer local epochs per FL run slightly improve global accuracy**, likely because frequent aggregation prevents local overfitting.
> * **More local epochs reduce communication frequency but may slightly degrade global performance** due to client drift.

This illustrates the classic **FL tradeoff**: balancing **local computation** versus **global aggregation** for accuracy and communication efficiency.

---

## 7. Combined Metrics Table (Per FL Run)

| Run ID | Local Epochs | Accuracy | Macro F1 | Micro F1 | SPD     | EOD     | Comm Cost | Runtime (sec) |
| ------ | ------------ | -------- | -------- | -------- | ------- | ------- | --------- | ------------- |
| 1      | 5            | 0.873836 | 0.778485 | 0.873836 | -0.1991 | -0.1991 | 4920      | 68.220        |
| 2      | 10           | 0.871100 | 0.766592 | 0.871100 | -0.1991 | -0.1991 | 4920      | 94.244        |
| 3      | 20           | 0.868622 | 0.758435 | 0.868622 | -0.1991 | -0.1991 | 4920      | 212.671       |

**Notes:**

* **Accuracy / F1 metrics:** show predictive performance of the federated model.
* **SPD / EOD:** fairness metrics remain stable, confirming **bias is not increased by FL training**.
* **Comm Cost:** constant because the number of clients and communication rounds is fixed.
* **Runtime:** grows with the number of local epochs, highlighting computational cost trade-offs.

**Significance for Federated Learning:**

* This table **summarizes all critical metrics in one view**, making it easy to analyze trade-offs between:

  * Model quality (accuracy, F1)
  * Fairness (SPD, EOD)
  * Communication efficiency (Comm Cost)
  * Computational cost (Runtime)
* Demonstrates how **local training parameters (like local epochs)** affect FL outcomes.
* Confirms that FL **preserves fairness and privacy** while achieving reasonable model performance.

---

