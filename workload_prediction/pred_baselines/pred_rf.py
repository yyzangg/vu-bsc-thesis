import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import randint
import joblib

# Load data
data = pd.read_csv('/home/yzg244/pred_baselines/pred_input_job.csv')

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
param_dist_rf = {
    'n_estimators': randint(100, 300),
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 4)
}

# Perform hyperparameter tuning using RandomizedSearchCV
rf = RandomForestClassifier()
random_search_rf = RandomizedSearchCV(rf, param_dist_rf, n_iter=50, cv=5, scoring='accuracy', random_state=0)
random_search_rf.fit(X_train, y_train)
best_rf = random_search_rf.best_estimator_

# Save the best model
joblib.dump(best_rf, 'best_random_forest_model.pkl')

# Evaluate the model
y_pred = best_rf.predict(X_test)
y_pred_proba = best_rf.predict_proba(X_test) if hasattr(best_rf, "predict_proba") else None

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

print("--- Random Forest ---")
print(classification_report(y_test, y_pred, digits=2))
print("Accuracy:", round(accuracy_score(y_test, y_pred), 2))
print("Precision:", round(precision_score(y_test, y_pred, average='weighted'), 2))
print("Recall:", round(recall_score(y_test, y_pred, average='weighted'), 2))
print("F1-score:", round(f1_score(y_test, y_pred, average='weighted'), 2))
print("AUC-ROC:", round(auc_roc, 2))
