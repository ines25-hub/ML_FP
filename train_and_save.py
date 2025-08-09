# train_and_save.py
"""
Train 3 models for multi-label classification (MLP required + RF + KNN),
compare them, save the best model and the scaler/metadata for Streamlit app.

Usage:
    python train_and_save.py --data_path steelplate_faults.csv
Or, set DATA_URL to download directly (if you have a hosted csv).
"""

import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import os
import json

# ---------------------------
# Config / label names
# ---------------------------
# Change these if your dataset columns differ.
LABEL_COLS = ['Pastry','Z_Scratch','K_Scratch','Stains','Dirtiness','Bumps','Other_Faults']

# ---------------------------
# Helper functions
# ---------------------------
def load_data(path_or_df):
    """
    Accepts a path to CSV or a pandas DataFrame.
    Expects label columns to be binary (0/1) with names in LABEL_COLS.
    """
    if isinstance(path_or_df, str):
        df = pd.read_csv(path_or_df)
    else:
        df = path_or_df.copy()
    # Basic check:
    missing_labels = [c for c in LABEL_COLS if c not in df.columns]
    if missing_labels:
        raise ValueError(f"Missing expected label columns in data: {missing_labels}")
    # Features: all numeric columns excluding LABEL_COLS
    X = df.drop(columns=LABEL_COLS)
    y = df[LABEL_COLS]
    return X, y, df

def preprocess(X_train, X_test):
    # Impute numeric and scale
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    X_train_num = imputer.fit_transform(X_train[num_cols])
    X_test_num = imputer.transform(X_test[num_cols])

    X_train_scaled = scaler.fit_transform(X_train_num)
    X_test_scaled = scaler.transform(X_test_num)

    # If there are non-numeric columns, one-hot them (rare for this dataset)
    # For simplicity, assume this dataset is numeric. If not, extend here.

    return X_train_scaled, X_test_scaled, {'imputer': imputer, 'scaler': scaler, 'num_cols': num_cols}

def evaluate_model(model, X_test, y_test):
    """
    Returns a dict with per-label precision/recall/f1 and averages.
    """
    y_pred = model.predict(X_test)
    # per-label metrics (no averaging)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)
    # micro/macro averages
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(y_test, y_pred, average='micro', zero_division=0)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
    acc = accuracy_score(y_test, y_pred)  # note: strict exact-match accuracy for multi-label rows

    result = {
        'per_label': {
            LABEL_COLS[i]: {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            } for i in range(len(LABEL_COLS))
        },
        'averages': {
            'micro': {'precision': float(p_micro), 'recall': float(r_micro), 'f1': float(f1_micro)},
            'macro': {'precision': float(p_macro), 'recall': float(r_macro), 'f1': float(f1_macro)},
        },
        'accuracy_exact_match': float(acc),
        'y_pred': y_pred.tolist()
    }
    return result

def print_comparison_table(results_dict):
    """
    results_dict: {model_name: eval_result_dict}
    Prints a summary table (model vs micro-f1, macro-f1, exact accuracy)
    """
    rows = []
    for name, r in results_dict.items():
        rows.append({
            'model': name,
            'micro_f1': r['averages']['micro']['f1'],
            'macro_f1': r['averages']['macro']['f1'],
            'exact_accuracy': r['accuracy_exact_match']
        })
    df = pd.DataFrame(rows).sort_values(by='micro_f1', ascending=False)
    print(df.to_string(index=False))
    return df

def plot_model_comparison(df_summary, outpath='model_comparison.png'):
    ax = df_summary.plot(x='model', y=['micro_f1','macro_f1','exact_accuracy'], kind='bar', figsize=(9,5))
    ax.set_ylabel('Score')
    ax.set_ylim(0,1)
    plt.tight_layout()
    plt.savefig(outpath)
    print(f"Saved comparison plot: {outpath}")

def save_artifacts(best_model, prep_objects, metadata, save_dir='saved_model'):
    os.makedirs(save_dir, exist_ok=True)
    # joblib save model
    joblib.dump(best_model, os.path.join(save_dir, 'best_model.joblib'))
    # save scaler and imputer separately
    joblib.dump(prep_objects['imputer'], os.path.join(save_dir, 'imputer.joblib'))
    joblib.dump(prep_objects['scaler'], os.path.join(save_dir, 'scaler.joblib'))
    # save metadata (label cols, numeric cols)
    with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)
    print(f"Saved artifacts to {save_dir}")

# ---------------------------
# Main training logic
# ---------------------------
def main(args):
    # Load data
    X, y, df_all = load_data(args.data_path)
    # Train/test split (random)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocess numeric columns
    X_train_scaled, X_test_scaled, prep = preprocess(X_train, X_test)

    # Build models (wrap in MultiOutputClassifier for multi-label)
    models = {}

    # MLP (required)
    mlp_base = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    mlp = MultiOutputClassifier(mlp_base)
    models['MLP'] = mlp

    # Random Forest
    rf_base = RandomForestClassifier(n_estimators=200, random_state=42)
    rf = MultiOutputClassifier(rf_base)
    models['RandomForest'] = rf

    # KNN
    knn_base = KNeighborsClassifier(n_neighbors=5)
    knn = MultiOutputClassifier(knn_base)
    models['KNN'] = knn

    trained_models = {}
    eval_results = {}

    for name, model in models.items():
        print(f"\nTraining {name} ...")
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
        print(f"Evaluating {name} ...")
        res = evaluate_model(model, X_test_scaled, y_test)
        eval_results[name] = res

    # Compare
    summary_df = print_comparison_table(eval_results)
    plot_model_comparison(summary_df, outpath='model_comparison.png')

    # Choose best model by micro-F1 (you can change criterion)
    best_model_name = summary_df.iloc[0]['model']
    best_model = trained_models[best_model_name]
    print(f"\nBest model: {best_model_name}")

    # Save artifacts for Streamlit: best model, imputer, scaler, metadata
    metadata = {
        'label_cols': LABEL_COLS,
        'num_cols': prep['num_cols'],
        'feature_order': prep['num_cols']  # currently numeric-only and in this order
    }
    save_artifacts(best_model, prep, metadata, save_dir=args.save_dir)

    # Save evaluation results to json for reporting
    with open(os.path.join(args.save_dir, 'evaluation_summary.json'), 'w') as f:
        json.dump(eval_results, f, indent=2)
    print("Saved evaluation_summary.json")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to CSV dataset (must include label columns).')
    parser.add_argument('--save_dir', type=str, default='saved_model', help='Directory to save model + artifacts.')
    args = parser.parse_args()
    main(args)
