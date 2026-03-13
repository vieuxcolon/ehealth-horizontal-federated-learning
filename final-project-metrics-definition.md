# Limitations: While we experimented with all metrics described in this file, we did not follow the rigorous task of plotting the different metrics against FL Rounds

# Evaluation Metrics in Federated Learning Experiments

This document explains the **evaluation metrics used in this project** and how they relate to **federated learning (FL)**. The metrics measure both:

1. **Model performance** (how accurate the global model is)
2. **Model fairness** (whether predictions differ across demographic groups)

The experiment uses the **Adult dataset** to predict whether a person's income exceeds **$50K**, while evaluating fairness with respect to the **sensitive attribute `sex`**.

---

# 1. Classification Metrics

The federated learning pipeline reports the following performance metrics:

* Accuracy
* Precision
* Recall
* F1 Score (Macro and Micro)

These metrics evaluate **the quality of the global model produced after federated aggregation**.

In federated learning, each client (hospital) trains locally and sends model updates to a server. The server aggregates these updates into a **global model**, and these metrics measure how well that model performs.

---

# 2. Confusion Matrix

All classification metrics are derived from the **confusion matrix**.

|                 | Predicted Positive  | Predicted Negative  |
| --------------- | ------------------- | ------------------- |
| Actual Positive | True Positive (TP)  | False Negative (FN) |
| Actual Negative | False Positive (FP) | True Negative (TN)  |

Definitions:

* **True Positive (TP)** – correctly predicted positive class
* **True Negative (TN)** – correctly predicted negative class
* **False Positive (FP)** – predicted positive but actually negative
* **False Negative (FN)** – predicted negative but actually positive

In this project:

Positive class:

```
income > 50K
```

---

# 3. Accuracy

## Formula

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

## Meaning

Accuracy measures the **overall correctness of the model**.

Example result from the experiment:

```
Accuracy ≈ 0.87
```

Interpretation:

```
About 87% of predictions are correct.
```

## Role in Federated Learning

In federated learning, accuracy measures the **quality of the final aggregated model** after all training rounds across distributed clients.

It answers the question:

> Did the collaborative federated training successfully produce a useful predictive model?

---

# 4. Precision

## Formula

```
Precision = TP / (TP + FP)
```

## Meaning

Precision answers:

> Of all individuals predicted to have high income, how many actually do?

High precision indicates **few false positives**.

Example interpretation:

If precision is high, the model rarely misclassifies low-income individuals as high-income.

---

# 5. Recall (Sensitivity)

## Formula

```
Recall = TP / (TP + FN)
```

## Meaning

Recall answers:

> Of all individuals who truly earn more than $50K, how many did the model correctly identify?

High recall means **few false negatives**.

Example interpretation:

If recall is high, the model successfully identifies most high-income individuals.

---

# 6. F1 Score

The **F1 score** combines precision and recall into a single metric.

## Formula

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

## Meaning

The F1 score is the **harmonic mean of precision and recall**.

It is particularly useful when:

* datasets are **imbalanced**
* accuracy alone may be misleading

The Adult dataset contains **class imbalance**, making F1 an important metric.

---

# 7. Macro F1 vs Micro F1

The experiment reports **both Macro and Micro F1 scores**.

## Micro F1

Micro F1 aggregates all predictions globally.

Conceptually:

```
Micro F1 = global precision/recall across all samples
```

Micro F1 emphasizes the **overall dataset performance**.

For binary classification it is often **close to accuracy**.

---

## Macro F1

Macro F1 computes F1 **independently for each class** and then averages them.

```
Macro F1 = (F1_class1 + F1_class2) / 2
```

Macro F1 treats **all classes equally**, regardless of their frequency.

This makes it useful when class distributions are **imbalanced**.

---

# 8. Fairness Metrics

In addition to performance metrics, the project evaluates **fairness** using two metrics:

* Statistical Parity Difference (SPD)
* Equal Opportunity Difference (EOD)

These metrics measure whether predictions differ between demographic groups.

Sensitive attribute:

```
sex
```

Groups:

```
Male (privileged group)
Female (unprivileged group)
```

---

# 9. Statistical Parity Difference (SPD)

## Formula

```
SPD = P(Ŷ = 1 | Female) − P(Ŷ = 1 | Male)
```

Where:

```
Ŷ = predicted label
```

## Meaning

SPD measures the **difference in positive prediction rates** between groups.

Ideal fairness:

```
SPD = 0
```

Example result from the dataset:

```
SPD ≈ -0.199
```

Interpretation:

```
Females receive positive predictions about 20% less often than males.
```

A negative SPD indicates **disadvantage for the unprivileged group**.

---

# 10. Equal Opportunity Difference (EOD)

## Formula

```
EOD = TPR_Female − TPR_Male
```

Where:

```
TPR = True Positive Rate
TPR = TP / (TP + FN)
```

## Meaning

EOD measures the difference in **recall between groups**.

Example result:

```
EOD ≈ -0.199
```

Interpretation:

```
Female high-income individuals are identified less often than male high-income individuals.
```

Ideal fairness:

```
EOD = 0
```

---

# 11. Why the Current Pipeline Computes SPD/EOD Using the Dataset

The current implementation computes fairness metrics using **true dataset labels**, not model predictions.

Conceptually, the pipeline calculates:

```
P(Y = 1 | group)
```

instead of

```
P(Ŷ = 1 | group)
```

This means the pipeline measures **dataset bias**, not **model bias**.

---

# 12. What the Current Fairness Values Represent

The fairness metrics in the pipeline measure:

```
Bias inherent in the dataset itself
```

Since the dataset remains constant across runs, the fairness values are also constant.

Example:

```
SPD = -0.1991
EOD = -0.1991
```

This indicates that **the Adult dataset contains gender bias**.

---

# 13. Correct Fairness Evaluation for Machine Learning Models

In fairness-aware machine learning, fairness is typically computed **using predictions from the trained model**.

Correct formulas:

Statistical parity:

```
SPD = P(Ŷ = 1 | Female) − P(Ŷ = 1 | Male)
```

Equal opportunity:

```
EOD = TPR_Female − TPR_Male
```

Where predictions are produced by the trained model.

This measures **whether the model introduces additional bias**.

---

# 14. Why Measuring Dataset Bias Is Still Useful

Measuring fairness using the dataset still provides valuable insight.

It answers the question:

```
Is the dataset itself biased?
```

If the dataset is biased:

```
Standard federated learning will preserve that bias.
```

Federated learning improves privacy and collaboration but **does not automatically correct biased data**.

---

# 15. Interpretation for This Federated Learning Experiment

The experiment demonstrates three important properties of federated learning.

## 1. Collaborative Learning

Multiple clients (simulated hospitals) jointly train a model without sharing raw data.

## 2. Privacy Preservation

Each hospital keeps its data locally and only shares **model parameters**.

## 3. Bias Propagation

Bias present in local datasets can propagate into the **global federated model**.

This highlights the need for:

```
Fairness-aware federated learning algorithms
```

when sensitive attributes are involved.

---

# 16. Key Takeaways

The metrics used in this project evaluate both **performance** and **fairness** of federated learning models.

Performance metrics measure:

```
How well the global model predicts outcomes.
```

Fairness metrics measure:

```
Whether predictions differ across demographic groups.
```

The experiment shows that while federated learning enables **privacy-preserving collaborative training**, it **does not inherently eliminate bias present in the underlying datasets**.
