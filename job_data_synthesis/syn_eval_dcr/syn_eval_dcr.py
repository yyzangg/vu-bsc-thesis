import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import trim_mean
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Define a dictionary mapping model names to titles
model_titles = {
    'Testset': 'Test Subset',
    'TabGAN': 'TabGAN',
    'CTGAN': 'CTGAN',
    'GPT-2': 'GPT-2',
    'BERT-B': 'BERT-Base',
    'BERT-L': 'BERT-Large',
    'DB-B': 'DistilBERT-Base',
    'RB-B': 'RoBERTa-Base',
    'RB-L': 'RoBERTa-Large',
    'BART-B': 'BART-Base',
    'BART-L': 'BART-Large',
    'GReaT': 'GReaT',
    'RELTF': 'REaLTabFormer',
}

colors = ['#f6ffd4', '#e8f7b2', '#d9ed92', 
         '#b5e48c', '#99d98c', '#76c893', 
         '#52b69a', '#34a0a4', '#168aad', 
         '#1a759f', '#1e6091', '#184e77', 
         '#093b63']

def scale_data(original_data, synthetic_data):
    """Scale original and synthetic data using StandardScaler."""
    scaler = StandardScaler()
    original_scaled = scaler.fit_transform(original_data)
    synthetic_scaled = scaler.transform(synthetic_data)
    return original_scaled, synthetic_scaled

def calculate_dcr(original_data, synthetic_data, feature_weights=None):
    """Calculate Distance to Closest Record (DCR) for synthetic data using Euclidean Distance."""
    dcr_distances = []
    for synthetic_row in synthetic_data:
        if feature_weights is not None:
            weighted_distances = []
            for original_row in original_data:
                weighted_distance = 0
                for j in range(len(original_row)):
                    if j in weighted_feature_indices:
                        weighted_distance += (original_row[j] - synthetic_row[j]) ** 2 * 1.5
                    else:
                        weighted_distance += (original_row[j] - synthetic_row[j]) ** 2 * 1.0
                weighted_distances.append(weighted_distance)
            dcr_distances.append(np.min(weighted_distances))
        else:
            # Default case: no feature weights applied
            distances = pairwise_distances([synthetic_row], original_data, metric='euclidean')
            min_distance = np.min(distances)
            dcr_distances.append(min_distance)
    return dcr_distances

def plot_histogram(dcr_distances, model_name, model_titles):
    """Plot the DCR histogram."""
    plt.figure(figsize=(15, 13))

    # Calculate the dynamic bin range based on the data
    min_dcr = np.min(dcr_distances)
    max_dcr = np.max(dcr_distances)
    num_bins = 13
    bin_range = np.linspace(min_dcr, max_dcr, num_bins)

    sns.histplot(dcr_distances, bins=bin_range, kde=False, stat='count', edgecolor='none')

    bars = plt.gca().patches
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    if model_name in model_titles:
        title = model_titles[model_name]
    else:
        title = model_name

    plt.title(title, fontsize=60)

    if model_name in ['BART-B', 'BART-L', 'BERT-B', 'BERT-L', 'DB-B']:
        plt.xscale('log')

    if model_name in ['BART-B', 'BART-L', 'BERT-B', 'BERT-L', 'DB-B']:
        plt.xlabel('DCR (Log)', fontsize=48)
    else:
        plt.xlabel('DCR', fontsize=48)

    plt.yscale('log')
    plt.ylabel('Number of Samples', fontsize=48)
    plt.xticks(fontsize=42)
    plt.yticks(fontsize=42)
    plt.savefig(f'dcr_count_{model_name.lower().replace(" ", "_")}.pdf')
    plt.close()

def trim_dcr_values(dcr_distances, proportion_to_cut):
    """Trim the specified proportion of extreme values from the DCR list."""
    trimmed_distances = np.sort(dcr_distances)
    lower_index = int(len(trimmed_distances) * proportion_to_cut)
    upper_index = int(len(trimmed_distances) * (1 - proportion_to_cut))
    return trimmed_distances[lower_index:upper_index]

def plot_boxplot_combined(dcr_results, proportion_to_cut, good_dcr_threshold):
    """Plot combined box plots for trimmed DCR values of all models."""
    plt.figure(figsize=(16, 8))
    model_names = list(dcr_results.keys())
    palette = {model: color for model, color in zip(model_names, colors)}

    combined_data = pd.DataFrame(
        [(model, dcr) for model, distances in dcr_results.items()
         for dcr in trim_dcr_values(distances, proportion_to_cut)],
        columns=['Model', 'DCR']
    )

    sns.boxplot(x='Model', y='DCR', data=combined_data, palette=palette)

    # Add vertical line denoting the good DCR threshold
    plt.axhline(y=good_dcr_threshold, color='darkgreen', linestyle='--', label=f'DCR Threshold: {good_dcr_threshold:.2f}')

    plt.yscale('log')
    plt.xlabel('Model', fontsize=22)
    plt.ylabel('DCR (Log)', fontsize=22)
    plt.xticks(rotation=0, fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig('dcr_boxplot.pdf')
    plt.close()

def plot_dcr(dcr_distances, model_name):
    plot_histogram(dcr_distances, model_name, model_titles)

def process_model(model_name, original_data, synthetic_data, feature_weights=None):
    """Process a single model: scale data, calculate DCR, and plot the results."""
    original_scaled, synthetic_scaled = scale_data(original_data, synthetic_data)
    dcr_distances = calculate_dcr(original_scaled, synthetic_scaled, feature_weights=feature_weights)
    plot_dcr(dcr_distances, model_name)
    return dcr_distances

def process_test_subset_dcr(original_data, test_data):
    """Process the DCR calculation for the test subset against the training subset."""
    train_data, temp_data = train_test_split(original_data, test_size=0.3, random_state=0)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=0)

    test_subset = test_data.sample(n=1000, random_state=0)
    train_scaled, test_scaled = scale_data(train_data, test_subset)
    dcr_distances = calculate_dcr(train_scaled, test_scaled)

    plot_dcr(dcr_distances, 'Testset')

    return dcr_distances

if __name__ == "__main__":
    df = pd.read_csv('/syn_input_job.csv')

    # Dictionary containing synthetic data for each model
    synthetic_datasets = {
        "TabGAN": pd.read_csv('/out_tabgan.csv'),
        "CTGAN": pd.read_csv('/out_ctgan.csv'),
        "BART-B": pd.read_csv('/out_bart_base.csv'),
        "BART-L": pd.read_csv('/out_bart_large.csv'),
        "BERT-B": pd.read_csv('/out_bert_base.csv'),
        "BERT-L": pd.read_csv('/out_bert_large.csv'),
        "DB-B": pd.read_csv('/out_db_base.csv'),
        "RB-B": pd.read_csv('/out_rb_base.csv'),
        "RB-L": pd.read_csv('/out_rb_large.csv'),
        "GPT-2": pd.read_csv('/out_gpt.csv'),
        "GReaT": pd.read_csv('/out_great.csv'),
        "RELTF": pd.read_csv('/out_rltf.csv'),
    }

    dcr_results = {}

    # Define feature weights
    feature_weights = np.ones(df.shape[1])  # Initialize with all ones
    weighted_feature_names = ['state_encoded', 'submit_hour_of_day', 'submit_day_of_month', 'submit_day_of_week']
    weighted_feature_indices = [df.columns.get_loc(feature) for feature in weighted_feature_names]

    test_dcr_distances = process_test_subset_dcr(df, None)
    dcr_results['Testset'] = test_dcr_distances

    for model_name, synthetic_data in synthetic_datasets.items():
        dcr_distances = process_model(model_name, df, synthetic_data, feature_weights=feature_weights)
        dcr_results[model_name] = dcr_distances

    # Calculate the DCR threshold
    good_dcr_threshold = 0.0

    for i in range(df.shape[1]):
        if df.columns[i] in weighted_feature_names:
            good_dcr_threshold += 0.01 # 1% of 1
        else:
            good_dcr_threshold += 0.05 # 5% of 1

    for model_name, dcr_distances in dcr_results.items():
        mean_value = np.mean(dcr_distances)
        trimmed_mean_value = trim_mean(dcr_distances, proportiontocut=0.10)  # 10% trimmed mean
        median_value = np.median(dcr_distances)
        print(f"{model_name} DCR mean: {mean_value:.4f}")
        print(f"{model_name} DCR trimmed mean: {trimmed_mean_value:.4f}")
        print(f"{model_name} DCR median: {median_value:.4f}")

    # Plot combined box plot for all models with 10% trimmed values
    plot_boxplot_combined(dcr_results, proportion_to_cut=0.10, good_dcr_threshold=good_dcr_threshold)
