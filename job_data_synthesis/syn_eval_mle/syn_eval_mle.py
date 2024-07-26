import logging
import numpy as np
import pandas as pd
from scipy.stats import randint
from imblearn.over_sampling import RandomOverSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Setup logging
logging.basicConfig(filename='mle_model_evaluation.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Define global variables
RANDOM_STATE = 0
N_ITER = 50
CV_FOLDS = 5

def load_and_subset_data(filenames):
    try:
        datasets = [pd.read_csv(filename) for filename in filenames]
        min_rows = min(len(dataset) for dataset in datasets)
        return [dataset.sample(n=min_rows, random_state=RANDOM_STATE) for dataset in datasets]
    except Exception as e:
        logging.error(f"Error loading datasets: {e}")
        return []

def binarize_target(dataset, target_column='state_encoded'):
    dataset[target_column] = dataset[target_column].apply(lambda x: 1 if x == 1 else 0)
    return dataset

def setup_randomized_search(model, param_dist, cv_folds=CV_FOLDS, scoring_metric='roc_auc'):
    return RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=N_ITER,
        cv=StratifiedKFold(cv_folds),
        scoring=scoring_metric,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

def train_and_evaluate_once(X_train, y_train, X_test, y_test, oversampler):
    X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)

    # Decision Tree
    decision_tree = setup_randomized_search(DecisionTreeClassifier(random_state=RANDOM_STATE), {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': randint(2, 11),
        'min_samples_leaf': randint(1, 11),
        'max_features': [None, 'sqrt', 'log2']
    })
    decision_tree.fit(X_resampled, y_resampled)
    y_pred_tree = decision_tree.best_estimator_.predict_proba(X_test)[:, 1]

    # Random Forest
    random_forest = setup_randomized_search(RandomForestClassifier(random_state=RANDOM_STATE), {
        'n_estimators': randint(50, 201),
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': randint(2, 11),
        'min_samples_leaf': randint(1, 11),
        'max_features': [None, 'sqrt', 'log2']
    })
    random_forest.fit(X_resampled, y_resampled)
    y_pred_rf = random_forest.best_estimator_.predict_proba(X_test)[:, 1]

    return compute_metrics(y_test, y_pred_tree), compute_metrics(y_test, y_pred_rf)

def compute_metrics(y_test, y_pred):
    y_pred_rounded = (y_pred >= 0.5).astype(int)
    return {
        'auc_roc': roc_auc_score(y_test, y_pred),
        'accuracy': accuracy_score(y_test, y_pred_rounded),
        'precision': precision_score(y_test, y_pred_rounded, average='weighted', zero_division=1),
        'recall': recall_score(y_test, y_pred_rounded, average='weighted'),
        'f1': f1_score(y_test, y_pred_rounded, average='weighted')
    }

def print_aggregated_metrics(metrics_list, model_name):
    metrics_df = pd.DataFrame(metrics_list)
    mean_metrics = metrics_df.mean()
    std_metrics = metrics_df.std()
    
    print(f"{model_name} Performance Metrics (Mean ± Std):")
    for metric in mean_metrics.index:
        print(f"{metric.capitalize()}: {mean_metrics[metric]:.2f} ± {std_metrics[metric]:.4f}")

def train_and_evaluate_models(dataset, dataset_name, feature_subset, oversampler, X_test, y_test, n_iterations=10):
    X_train = dataset[feature_subset]
    y_train = dataset['state_encoded']

    if y_train.nunique() < 2:
        logging.info(f"Dataset {dataset_name} skipped: target variable has less than two classes.")
        return

    metrics_tree_all, metrics_rf_all = [], []

    for iteration in range(n_iterations):
        metrics_tree, metrics_rf = train_and_evaluate_once(X_train, y_train, X_test, y_test, oversampler)
        metrics_tree_all.append(metrics_tree)
        metrics_rf_all.append(metrics_rf)
    
    logging.info(f"Results for {dataset_name} averaged over {n_iterations} iterations:")
    print(f"Results for {dataset_name} (averaged over {n_iterations} iterations)")
    print_aggregated_metrics(metrics_tree_all, "Decision Tree")
    print_aggregated_metrics(metrics_rf_all, "Random Forest")
    print("-" * 50)

def main():
    filenames = ['syn_input_job.csv', 'out_ctgan.csv', 'out_tabgan.csv', 
                 'out_bart_base.csv', 'out_bart_large.csv', 
                 'out_db_base.csv', 'out_bert_base.csv', 'out_bert_large.csv', 'out_rb_base.csv', 'out_rb_large.csv', 
                 'out_gpt.csv', 'out_great.csv', 'out_rltf.csv']
    datasets = load_and_subset_data(filenames)
    datasets = [binarize_target(dataset) for dataset in datasets]

    feature_subset = ['ReqCPUS', 'submit_hour_of_day', 'submit_day_of_week', 'submit_day_of_month']
    oversampler = RandomOverSampler(random_state=RANDOM_STATE)

    X_input = datasets[0][feature_subset].head(1250)
    y_input = datasets[0]['state_encoded'].head(1250)
    X_train_input, X_test_input, y_train_input, y_test_input = train_test_split(X_input, y_input, test_size=0.2, random_state=RANDOM_STATE)

    for i, dataset in enumerate(datasets):
        dataset_name = filenames[i]
        train_and_evaluate_models(dataset, dataset_name, feature_subset, oversampler, X_test_input, y_test_input)

if __name__ == "__main__":
    main()
