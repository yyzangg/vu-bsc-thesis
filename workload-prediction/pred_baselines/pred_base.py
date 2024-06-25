import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import uniform, randint

# Load data
df = pd.read_csv('/home/yzg244/pred_baselines/pred_input_job.csv')

data = df

# Feature and target selection
X = data.drop('state_encoded', axis=1)
y = data['state_encoded']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define hyperparameter distributions
param_dist_log_reg = {
    'C': uniform(0.01, 100),
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

param_dist_rf = {
    'n_estimators': randint(100, 300),
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 4)
}

param_dist_xgb = {
    'n_estimators': randint(100, 300),
    'max_depth': randint(3, 9),
    'learning_rate': uniform(0.01, 0.2),
    'subsample': uniform(0.6, 0.4)
}

# Perform hyperparameter tuning using RandomizedSearchCV
log_reg = LogisticRegression()
random_search_log_reg = RandomizedSearchCV(log_reg, param_dist_log_reg, n_iter=50, cv=5, scoring='accuracy', random_state=0)
random_search_log_reg.fit(X_train, y_train)
best_log_reg = random_search_log_reg.best_estimator_

rf = RandomForestClassifier()
random_search_rf = RandomizedSearchCV(rf, param_dist_rf, n_iter=50, cv=5, scoring='accuracy', random_state=0)
random_search_rf.fit(X_train, y_train)
best_rf = random_search_rf.best_estimator_

xgb = XGBClassifier()
random_search_xgb = RandomizedSearchCV(xgb, param_dist_xgb, n_iter=50, cv=5, scoring='accuracy', random_state=0)
random_search_xgb.fit(X_train, y_train)
best_xgb = random_search_xgb.best_estimator_

# Update models with best estimators
models = {
    'Logistic Regression': best_log_reg,
    'Random Forest': best_rf,
    'XGBoost': best_xgb
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    
    # Calculate AUC-ROC score
    if y_pred_proba is not None:
        if len(y.unique()) > 2:
            # Multiclass case
            auc_roc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        else:
            # Binary case
            auc_roc = roc_auc_score(y_test, y_pred_proba[:, 1])
    else:
        auc_roc = "Not available"
    
    print(f"--- {name} ---")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='weighted'))
    print("Recall:", recall_score(y_test, y_pred, average='weighted'))
    print("F1-score:", f1_score(y_test, y_pred, average='weighted'))
    print("AUC-ROC:", auc_roc)
    print("\n")
