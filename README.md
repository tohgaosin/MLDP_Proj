# 🧠 Stroke Prediction (MLDP\_Proj)

A supervised machine learning project to **predict stroke risk** from demographic, clinical, and lifestyle data.
It uses data cleaning, exploratory analysis, categorical encoding, class imbalance handling (**SMOTE**), and multiple classifiers — with **Quadratic Discriminant Analysis (QDA)** chosen for its strong recall on stroke cases.

Dataset: `healthcare-dataset-stroke-data.csv` (commonly known as the Kaggle “Stroke Prediction” dataset).

---

## 🚀 Project Overview

This project:

* Performs EDA to understand variable distributions, detect missing values and outliers.
* Cleans data (median imputation, outlier capping).
* Encodes categorical variables using one-hot encoding.
* Uses **SMOTE** to handle severe class imbalance in training data.
* Trains multiple models and evaluates them on a held-out test set.
* Tunes and selects **QDA** for its superior **recall** on the stroke (positive) class.
* Drops features with **zero tree-based feature importance**.
* Saves the trained model with `joblib` for deployment.

---

## 📂 Repository Structure

```
MLDP_Proj/
├─ healthcare-dataset-stroke-data.csv     # Input dataset
├─ stroke.ipynb                           # Analysis and model training code
├─ stroke-app.py                          # Streamlit application for interactive stroke risk prediction using the trained model
├─ stroke_qda.pkl                         # Final saved model
├─ requirements.txt                       # Environment dependencies
└─ README.md
```

---

## 🧰 Dependencies

```bash
pip install -U pandas numpy seaborn matplotlib scikit-learn imbalanced-learn joblib
```

**Main libraries:**
`pandas`, `numpy`, `seaborn`, `matplotlib`, `scikit-learn`, `imbalanced-learn` (SMOTE), `joblib`.

---

## 📊 Data & Preprocessing

**Original columns (12):**
`id`, `gender`, `age`, `hypertension`, `heart_disease`, `ever_married`, `work_type`, `Residence_type`, `avg_glucose_level`, `bmi`, `smoking_status`, `stroke`

**Target:** `stroke` (binary)

**Cleaning steps:**

* Missing `bmi` values → filled with median.
* Outliers in `bmi` and `avg_glucose_level` detected using IQR rule → capped to median.
* Dropped `id`.
* One-hot encoded categorical variables.
* **Dropped zero-importance features** (from tree-based probe):

  * `gender_Other`
  * `work_type_Never_worked`
  * `work_type_children`
* Created a parallel dataset with **BMI binning** (*Underweight*, *Healthy*, *Overweight*) for comparison.

---
## 🔎 Feature Selection via Feature Importance

A Decision Tree baseline was trained to probe feature importances before final model selection. Three features had **0.0 importance** and were dropped:

* `gender_Other`
* `work_type_Never_worked`
* `work_type_children`

## ⚖️ Class Imbalance

The dataset is highly imbalanced:

```
Before SMOTE (training set):
0    3404
1     173

After SMOTE:
0    3404
1    3404
```

 SMOTE is used to address class imbalance in datasets. It works by generating synthetic samples for the minority class, effectively balancing the dataset and improving the performance of machine learning models. 

---

## 📈 Models Tested

Trained on SMOTE-balanced data and evaluated on the original test set:

| Model                               | Accuracy  | Recall (Stroke) |
| ----------------------------------- | --------- | --------------- |
| Decision Tree                       | 0.905     | 0.09            |
| K-Nearest Neighbors                 | 0.797     | 0.58            |
| Gradient Boosting                   | 0.947     | 0.00            |
| Random Forest                       | 0.944     | 0.00            |
| Support Vector Classifier           | 0.639     | 0.93            |
| **Quadratic Discriminant Analysis** | **0.721** | **0.76**        |

---

## ✅ Why QDA was chosen

* **High recall** for stroke class (0.76) — a key metric in medical datasets where positive cases are rare but critical.
* Prioritising recall reduces false negatives, ensuring fewer stroke cases are missed.
* In screening contexts, false positives are acceptable since flagged cases can undergo further clinical testing.
* Tuned QDA parameters:

  ```python
  priors=[0.6, 0.4], reg_param=0.01, store_covariance=True, tol=1e-4
  ```

---



## 💾 Model Saving & Loading

```python
import joblib
joblib.dump(model, "stroke_qda.pkl")
clf = joblib.load("stroke_qda.pkl")
```

---

## 🔮 Inference Example

Order of features after encoding and drops:

```
[ age, hypertension, heart_disease, avg_glucose_level, bmi,
  gender_Female, gender_Male,
  ever_married_No, ever_married_Yes,
  work_type_Govt_job, work_type_Private, work_type_Self-employed,
  Residence_type_Rural, Residence_type_Urban,
  smoking_status_Unknown, smoking_status_formerly smoked,
  smoking_status_never smoked, smoking_status_smokes ]
```

**Prediction example:**

```python
import numpy as np
x = np.array([[
    61.0, 0, 0, 91.885, 28.1,
    1, 0,
    0, 1,
    0, 0, 1,
    1, 0,
    0, 0, 1, 0
]])
print(clf.predict(x))  # '0' or '1'
```

---

## 🌐 Streamlit App

https://mldpproj-gcdwdjkwxswbdosyejvjjq.streamlit.app

---

## 📚 Data Source

https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

---

## ⚠️ Disclaimer

This model is for educational/research purposes only and is **not** a medical diagnostic tool.

---

