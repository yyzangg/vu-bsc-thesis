import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error

data = pd.read_csv('/pred_baselines/resampled_pred_input_job.csv')

# Task 1: Classify 'state_encoded'

X_class = data[['ReqCPUS', 'submit_hour_of_day', 'submit_day_of_week', 'submit_day_of_month']]
y_class = data['state_encoded']

# Define hyperparameter grid for Logistic Regression
param_grid_log_reg = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'max_iter': [500, 1000, 2000]
}

def evaluate_model(X, y, model, param_grid, cv=5, n_repeats=10):
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'auc_roc': []
    }
    
    for _ in range(n_repeats):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='roc_auc_ovr', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test) if hasattr(best_model, "predict_proba") else None
        
        metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['precision'].append(precision_score(y_test, y_pred, average='weighted', zero_division=1))
        metrics['recall'].append(recall_score(y_test, y_pred, average='weighted'))
        metrics['f1_score'].append(f1_score(y_test, y_pred, average='weighted'))
        if y_pred_proba is not None:
            if len(y_class.unique()) > 2:
                # Multiclass case
                metrics['auc_roc'].append(roc_auc_score(y_test, y_pred_proba, multi_class='ovr'))
            else:
                # Binary case
                metrics['auc_roc'].append(roc_auc_score(y_test, y_pred_proba[:, 1]))
        else:
            metrics['auc_roc'].append(0.0)
    
    results = {}
    for key, values in metrics.items():
        results[key] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }
    
    return results

log_reg = LogisticRegression()
log_reg_results = evaluate_model(X_class, y_class, log_reg, param_grid_log_reg)

print("--- Logistic Regression ---")
for metric, result in log_reg_results.items():
    print(f"{metric.capitalize()}: {result['mean']:.4f} ± {result['std']:.4f}")

# Task 2: Predict 'running_time' for instances where 'state_encoded' is not '1'

data_filtered = data[data['state_encoded'] != 1]
X_reg = data_filtered[['ReqCPUS', 'submit_hour_of_day', 'submit_day_of_week', 'submit_day_of_month', 'state_encoded']]
y_reg = data_filtered['running_time']

# Define hyperparameter grid for Random Forest Regressor
param_grid_rf_reg = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

def evaluate_regression_model(X, y, model, param_grid, cv=5, n_repeats=10):
    metrics = {
        'mse': [],
        'rmse': []
    }
    
    for _ in range(n_repeats):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        target_scaler = MinMaxScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).ravel()
        
        grid_search = GridSearchCV(model, param_grid, cv=cv, n_jobs=-1)
        grid_search.fit(X_train, y_train_scaled)
        best_model = grid_search.best_estimator_
        
        y_pred_scaled = best_model.predict(X_test)
        
        # Denormalize the predicted target variable
        y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        metrics['mse'].append(mean_squared_error(y_test, y_pred))
        metrics['rmse'].append(mean_squared_error(y_test, y_pred, squared=False))
    
    results = {}
    for key, values in metrics.items():
        results[key] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }
    
    return results

rf_reg = RandomForestRegressor(random_state=0)
rf_reg_results = evaluate_regression_model(X_reg, y_reg, rf_reg, param_grid_rf_reg)

print("--- Random Forest Regression for Running Time ---")
for metric, result in rf_reg_results.items():
    print(f"{metric.upper()}: {result['mean']:.4f} ± {result['std']:.2f}")