# 🍷 Wine Quality Classification - ML Assignment 2

**Author:** Dhruv  
**Programme:** M.Tech (AIML/DSE), BITS Pilani  
**Subject:** Machine Learning  
**Assignment:** Lab Assignment 2  

---

## 📌 Problem Statement

Predict the quality of wine based on physicochemical properties measured during wine production. This is a **binary classification problem** where wines rated **≥ 7** are classified as **Good Quality (1)** and wines rated **< 7** as **Standard Quality (0)**. The dataset combines both red and white Portuguese "Vinho Verde" wine variants from the UCI Machine Learning Repository.

Accurate wine quality prediction can help winemakers optimize production processes and maintain consistent quality. This project implements and compares 6 different machine learning classification models to determine which best predicts wine quality.

---

## 📊 Dataset Description

| Property | Details |
|----------|---------|
| **Dataset Name** | Wine Quality Dataset (Combined Red & White) |
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality) |
| **Total Instances** | 6497 |
| **Total Features** | 12 |
| **Target Variable** | `target` — Binary: 1 = Good Quality (rating ≥ 7), 0 = Standard Quality (rating < 7) |
| **Task Type** | Binary Classification |
| **Train/Test Split** | 80/20 (4256 train, 1064 test) |

### Features

| # | Feature | Description |
|---|---------|-------------|
| 1 | `fixed_acidity` | Non-volatile acids contributing to wine taste (tartaric acid) — g/dm³ |
| 2 | `volatile_acidity` | Amount of acetic acid; high levels lead to vinegar taste — g/dm³ |
| 3 | `citric_acid` | Adds freshness and flavor to wines — g/dm³ |
| 4 | `residual_sugar` | Sugar remaining after fermentation; affects sweetness — g/dm³ |
| 5 | `chlorides` | Amount of salt in the wine — g/dm³ |
| 6 | `free_sulfur_dioxide` | Free form of SO₂; prevents microbial growth — mg/dm³ |
| 7 | `total_sulfur_dioxide` | Total SO₂ (free + bound); detectable at >50 ppm — mg/dm³ |
| 8 | `density` | Density of wine; related to alcohol and sugar content — g/cm³ |
| 9 | `pH` | Acidity level on 0–14 scale (most wines are 3–4) |
| 10 | `sulphates` | Wine additive contributing to SO₂ levels — g/dm³ |
| 11 | `alcohol` | Alcohol content of the wine — % by volume |
| 12 | `wine_type` | Type of wine (0 = Red, 1 = White) |

**Citation:** P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. "Modeling wine preferences by data mining from physicochemical properties." Decision Support Systems, 47(4):547-553, 2009.

---

## 🤖 Models Used

Six classification models were implemented and evaluated on the same dataset:

1. **Logistic Regression** — Linear model for binary classification with L2 regularization
2. **Decision Tree Classifier** — Tree-based model with interpretable rules (max_depth=10)
3. **K-Nearest Neighbor (KNN) Classifier** — Instance-based learning with distance weighting (k=7)
4. **Naive Bayes (Gaussian)** — Probabilistic classifier based on Bayes' theorem
5. **Random Forest (Ensemble)** — Bagging ensemble of 200 decision trees
6. **XGBoost (Ensemble)** — Gradient boosting with 200 estimators and learning rate 0.1

### 📋 Evaluation Metrics Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.8102 | 0.8174 | 0.7826 | 0.8102 | 0.7896 | 0.2793 |
| Decision Tree | 0.7857 | 0.7313 | 0.7711 | 0.7857 | 0.7774 | 0.2542 |
| KNN | 0.8402 | 0.8379 | 0.8251 | 0.8402 | 0.8285 | 0.4213 |
| Naive Bayes | 0.7331 | 0.7592 | 0.8036 | 0.7331 | 0.7560 | 0.3392 |
| Random Forest (Ensemble) | 0.8421 | 0.8702 | 0.8239 | 0.8421 | 0.8211 | 0.3975 |
| XGBoost (Ensemble) | 0.8383 | 0.8641 | 0.8203 | 0.8383 | 0.8225 | 0.3993 |

---

## 📝 Model Performance Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| **Logistic Regression** | Achieved 81.02% accuracy with an AUC of 0.8174. As a linear model, it provides a solid baseline but struggles with the non-linear relationships in wine chemistry data. The relatively low MCC (0.2793) indicates that while overall accuracy is decent, the model has difficulty correctly classifying the minority class (Good Quality wines). The linear decision boundary is too simplistic for the complex feature interactions present in wine quality. |
| **Decision Tree** | Showed the **lowest accuracy (78.57%)** and **lowest AUC (0.7313)** among all models. Despite having max_depth=10 constraint, the single tree overfits to training patterns that don't generalize well. The low MCC (0.2542) confirms poor balanced classification. Single decision trees are inherently unstable and sensitive to small data variations, making them unreliable for this task. |
| **KNN** | Delivered strong performance with **84.02% accuracy** and the second-highest AUC (0.8379). The **highest MCC (0.4213)** among all models indicates the best balanced classification between both classes. Distance-weighted KNN (k=7) effectively captures local patterns in the feature space. The physicochemical properties form natural clusters that KNN exploits well, though it's computationally expensive at prediction time. |
| **Naive Bayes** | Achieved the **lowest accuracy (73.31%)** but shows an interesting trade-off: it has the **highest recall on the minority class** (Good Quality: 65.3%), meaning it catches more truly good wines. The high precision (0.8036) combined with lower accuracy reveals the independence assumption is violated — wine features like alcohol, density, and residual sugar are inherently correlated. Best suited when prioritizing recall for good wines. |
| **Random Forest (Ensemble)** | Achieved the **best accuracy (84.21%)** and **highest AUC (0.8702)** among all models, demonstrating excellent overall discriminative ability. The ensemble of 200 trees successfully overcomes individual decision tree instability. However, MCC (0.3975) is lower than KNN, suggesting slight imbalance in per-class performance. Random Forest also provides feature importance — alcohol content, volatile acidity, and density are likely the most predictive features. |
| **XGBoost (Ensemble)** | Close second in accuracy (83.83%) with the second-highest AUC (0.8641). XGBoost's sequential boosting approach builds strong predictive power with learning rate 0.1 preventing overfitting. The **best MCC among ensemble models (0.3993)** suggests slightly better balanced performance than Random Forest. XGBoost handles the class imbalance reasonably well and benefits from built-in regularization. Could improve further with hyperparameter tuning (e.g., scale_pos_weight for imbalanced classes). |

---

## 🚀 Streamlit Web Application

The interactive web application includes:

- ✅ **Dataset upload option (CSV)** — Upload test data for evaluation
- ✅ **Model selection dropdown** — Choose from 6 trained ML models
- ✅ **Display of evaluation metrics** — Accuracy, AUC, Precision, Recall, F1, MCC
- ✅ **Confusion matrix and classification report** — Detailed performance visualization

### App Sections:
1. **Overview** — Dataset description and feature information
2. **Model Comparison** — Side-by-side comparison of all 6 models with visualizations
3. **Individual Model Analysis** — Detailed metrics, confusion matrix, ROC curve per model
4. **Upload & Predict** — Upload CSV data and get predictions with evaluation

---

## 📂 Repository Structure

```
project-folder/
│── app.py                    # Streamlit web application
│── requirements.txt          # Python dependencies
│── README.md                 # Project documentation
│── model/
│   ├── train_models.py       # Model training script
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   ├── scaler.joblib
│   ├── results.json
│   ├── dataset_info.json
│   ├── feature_names.json
│   └── test_data.csv
```

---

## 🛠️ How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train models (if not already trained):**
   ```bash
   python model/train_models.py
   ```

4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

---

## 🔗 Links

- **GitHub Repository:** [https://github.com/07dhruvsingh/ML_Assignment_2](https://github.com/07dhruvsingh/ML_Assignment_2)
- **Live Streamlit App:** [https://ml-assignment-2.streamlit.app/](https://ml-assignment-2.streamlit.app/)

---

## 📜 License

This project is developed as part of the ML Lab Assignment 2 for M.Tech (AIML/DSE) at BITS Pilani.
