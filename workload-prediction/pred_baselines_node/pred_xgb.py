# xgboost_model.py
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import randint, uniform

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
param_dist_xgb = {
    'n_estimators': randint(100, 300),
    'max_depth': randint(3, 9),
    'learning_rate': uniform(0.01, 0.2),
    'subsample': uniform(0.6, 0.4)
}

# Perform hyperparameter tuning using RandomizedSearchCV
xgb = XGBClassifier()
random_search_xgb = RandomizedSearchCV(xgb, param_dist_xgb, n_iter=50, cv=5, scoring='accuracy', random_state=0)
random_search_xgb.fit(X_train, y_train)
best_xgb = random_search_xgb.best_estimator_

# Save the best model
import joblib
joblib.dump(best_xgb, 'best_xgboost_model.pkl')

# Evaluate the model
y_pred = best_xgb.predict(X_test)
y_pred_proba = best_xgb.predict_proba(X_test) if hasattr(best_xgb, "predict_proba") else None

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

print("--- XGBoost ---")
print(classification_report(y_test, y_pred, digits=2))
print("Accuracy:", round(accuracy_score(y_test, y_pred), 2))
print("Precision:", round(precision_score(y_test, y_pred, average='weighted'), 2))
print("Recall:", round(recall_score(y_test, y_pred, average='weighted'), 2))
print("F1-score:", round(f1_score(y_test, y_pred, average='weighted'), 2))
print("AUC-ROC:", round(auc_roc, 2))

