"""
Train and evaluate 6 ML classification models on the Wine Quality dataset.
Models: Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, XGBoost
Metrics: Accuracy, AUC, Precision, Recall, F1, MCC
Dataset: Combined Red & White Wine Quality from UCI (6497 instances, 12 features)
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import joblib
import warnings
warnings.filterwarnings('ignore')


def load_wine_quality_data():
    """
    Load the Wine Quality dataset from UCI Machine Learning Repository.
    Combines Red and White wine datasets + adds wine_type feature.
    Binary classification: Good quality (>= 7) vs Standard quality (< 7).
    Total: 6497 instances, 12 features.
    """
    # Load both wine quality datasets
    red_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    white_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

    red_wine = pd.read_csv(red_url, sep=';')
    white_wine = pd.read_csv(white_url, sep=';')

    # Add wine type feature (0 = red, 1 = white)
    red_wine['wine_type'] = 0
    white_wine['wine_type'] = 1

    # Combine datasets
    df = pd.concat([red_wine, white_wine], ignore_index=True)

    # Clean column names (remove spaces)
    df.columns = df.columns.str.replace(' ', '_')

    # Convert to binary classification: Good quality (>= 7) vs Standard (< 7)
    df['target'] = (df['quality'] >= 7).astype(int)
    df = df.drop('quality', axis=1)

    return df


def preprocess_data(df):
    """Clean and preprocess the dataset."""
    # Drop rows with missing values
    df = df.dropna()

    # Remove duplicates
    df = df.drop_duplicates()

    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']

    # Encode categorical features if any
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Convert all to numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.median())

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    return X_scaled, y, scaler, X.columns.tolist()


def get_models():
    """Return a dictionary of classification models."""
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=42, C=1.0, solver='lbfgs'
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=42, max_depth=10, min_samples_split=5
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=7, weights='distance', metric='minkowski'
        ),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, random_state=42, max_depth=15,
            min_samples_split=5, n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200, random_state=42, max_depth=6,
            learning_rate=0.1, use_label_encoder=False,
            eval_metric='logloss'
        )
    }
    return models


def evaluate_model(model, X_test, y_test):
    """Evaluate a model and return metrics dictionary."""
    y_pred = model.predict(X_test)

    # Get prediction probabilities for AUC
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)
        if y_proba.shape[1] == 2:
            auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
    else:
        auc = 0.0

    metrics = {
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'AUC': round(auc, 4),
        'Precision': round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 4),
        'Recall': round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 4),
        'F1': round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 4),
        'MCC': round(matthews_corrcoef(y_test, y_pred), 4)
    }

    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True)

    return metrics, cm, report


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("  Wine Quality Classification - Model Training Pipeline")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/5] Loading Wine Quality dataset...")
    df = load_wine_quality_data()
    print(f"  Dataset shape: {df.shape}")
    print(f"  Features: {df.shape[1] - 1}")
    print(f"  Instances: {df.shape[0]}")
    print(f"  Target distribution:\n{df['target'].value_counts().to_string()}")

    # Step 2: Preprocess
    print("\n[2/5] Preprocessing data...")
    X, y, scaler, feature_names = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")

    # Step 3: Train and evaluate
    print("\n[3/5] Training and evaluating models...")
    models = get_models()
    results = {}

    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.fit(X_train, y_train)
        metrics, cm, report = evaluate_model(model, X_test, y_test)
        results[name] = {
            'metrics': metrics,
            'confusion_matrix': cm,
            'classification_report': report
        }
        print(f"    Accuracy: {metrics['Accuracy']:.4f} | AUC: {metrics['AUC']:.4f} | "
              f"F1: {metrics['F1']:.4f} | MCC: {metrics['MCC']:.4f}")

    # Step 4: Save models and artifacts
    print("\n[4/5] Saving models and artifacts...")
    model_dir = os.path.dirname(os.path.abspath(__file__))

    for name, model in models.items():
        filename = name.lower().replace(' ', '_').replace('-', '_') + '.joblib'
        filepath = os.path.join(model_dir, filename)
        joblib.dump(model, filepath)
        print(f"  Saved: {filename}")

    # Save scaler
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
    print("  Saved: scaler.joblib")

    # Save results
    with open(os.path.join(model_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("  Saved: results.json")

    # Save feature names
    with open(os.path.join(model_dir, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f)
    print("  Saved: feature_names.json")

    # Save dataset info
    dataset_info = {
        'name': 'Wine Quality Dataset (Combined Red & White)',
        'source': 'UCI Machine Learning Repository',
        'url': 'https://archive.ics.uci.edu/ml/datasets/Wine+Quality',
        'total_instances': int(df.shape[0]),
        'total_features': int(df.shape[1] - 1),
        'feature_names': feature_names,
        'target_name': 'target',
        'target_description': 'Binary: 1 = Good quality (rating >= 7), 0 = Standard quality (rating < 7)',
        'classes': ['Standard Quality (0)', 'Good Quality (1)'],
        'train_size': int(X_train.shape[0]),
        'test_size': int(X_test.shape[0])
    }
    with open(os.path.join(model_dir, 'dataset_info.json'), 'w') as f:
        json.dump(dataset_info, f, indent=2)
    print("  Saved: dataset_info.json")

    # Save test data for the Streamlit app
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df['target'] = y_test.values
    test_df.to_csv(os.path.join(model_dir, 'test_data.csv'), index=False)
    print("  Saved: test_data.csv")

    # Step 5: Print comparison table
    print("\n[5/5] Model Comparison Table")
    print("=" * 90)
    header = f"{'Model':<25} {'Accuracy':>10} {'AUC':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'MCC':>10}"
    print(header)
    print("-" * 90)
    for name, data in results.items():
        m = data['metrics']
        row = (f"{name:<25} {m['Accuracy']:>10.4f} {m['AUC']:>10.4f} "
               f"{m['Precision']:>10.4f} {m['Recall']:>10.4f} {m['F1']:>10.4f} {m['MCC']:>10.4f}")
        print(row)
    print("=" * 90)

    print("\n✅ All models trained and saved successfully!")
    return results


if __name__ == '__main__':
    main()
