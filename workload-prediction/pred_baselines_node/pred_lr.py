import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import uniform
import joblib

# Load data
data = pd.read_csv('/home/yzg244/pred_baselines/pred_input.csv')
data = data.iloc[:100000]

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
    'C': uniform(0.01, 10),
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'max_iter': [500, 1000, 2000]
}

# Perform hyperparameter tuning using RandomizedSearchCV
log_reg = LogisticRegression()
random_search_log_reg = RandomizedSearchCV(log_reg, param_dist_log_reg, n_iter=50, cv=5, scoring='accuracy', random_state=0, n_jobs=-1)
random_search_log_reg.fit(X_train, y_train)
best_log_reg = random_search_log_reg.best_estimator_

# Save the best model
joblib.dump(best_log_reg, 'best_logistic_regression_model.pkl')

# Evaluate the model
y_pred = best_log_reg.predict(X_test)
y_pred_proba = best_log_reg.predict_proba(X_test) if hasattr(best_log_reg, "predict_proba") else None

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

print("--- Logistic Regression ---")
print(classification_report(y_test, y_pred, digits=2))
print("Accuracy:", round(accuracy_score(y_test, y_pred), 2))
print("Precision:", round(precision_score(y_test, y_pred, average='weighted'), 2))
print("Recall:", round(recall_score(y_test, y_pred, average='weighted'), 2))
print("F1-score:", round(f1_score(y_test, y_pred, average='weighted'), 2))
print("AUC-ROC:", round(auc_roc, 2))
