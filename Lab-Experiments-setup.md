---

## Experiment Summary: This part summarizes the setup used for all the Labs

This table provides a **high-level overview of all 36 experiments**, grouped by mode and dataset. It shows the run ranges, models, client ratios, local epochs, and training paradigms.

| Runs  | Mode        | Dataset | Model        | Client Ratio  | Local Epochs | Training Paradigm           |
| ----- | ----------- | ------- | ------------ | ------------- | ------------ | --------------------------- |
| 1–9   | Centralized | MNIST   | MNIST_2NN    | 0.3, 0.6, 1.0 | 5, 10, 20    | Centralized Training        |
| 10–18 | Centralized | Adult   | Adult_LogReg | 0.3, 0.6, 1.0 | 5, 10, 20    | Centralized Training        |
| 19–27 | Federated   | MNIST   | MNIST_2NN    | 0.3, 0.6, 1.0 | 5, 10, 20    | Federated Learning (FedAvg) |
| 28–36 | Federated   | Adult   | Adult_LogReg | 0.3, 0.6, 1.0 | 5, 10, 20    | Federated Learning (FedAvg) |

---

### Hyperparameter Grid

| Category             | Parameter          | Values                  |
| -------------------- | ------------------ | ----------------------- |
| Training Mode        | Mode               | centralized, federation |
| Dataset              | Dataset            | MNIST, Adult            |
| Client Participation | Client Ratio       | 0.3, 0.6, 1.0           |
| Local Training       | Local Epochs       | 5, 10, 20               |
| FL Algorithm         | Aggregation Method | FedAvg                  |
| MNIST Model          | Architecture       | MNIST_2NN               |
| Adult Model          | Architecture       | Logistic Regression     |
| Adult Feature Size   | Input Dimension    | 14                      |

---

### Experiment Grid Size

| Parameter    | Options | Count                   |
| ------------ | ------- | ----------------------- |
| Mode         | 2       | centralized, federation |
| Dataset      | 2       | MNIST, Adult            |
| Client Ratio | 3       | 0.3, 0.6, 1.0           |
| Local Epochs | 3       | 5, 10, 20               |

**Total experiments:** 2 × 2 × 3 × 3 = **36 runs**

---
