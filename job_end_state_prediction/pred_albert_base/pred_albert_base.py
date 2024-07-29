import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, classification_report
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AlbertTokenizer, AlbertModel, AdamW, get_linear_schedule_with_warmup

df = pd.read_csv('/pred_baselines/resampled_pred_input_job.csv')

# Step 1: Preprocess
X = df[['ReqCPUS', 'submit_hour_of_day', 'submit_day_of_week', 'submit_day_of_month']]
y = df['state_encoded']

scaler = StandardScaler()
scaled_data = scaler.fit_transform(X.values)
data_strings = [' '.join(map(str, row)) for row in scaled_data]

# Step 2: Data Pipeline
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

# Tokenize the string representations of numerical data
encoded_inputs = tokenizer(data_strings, padding=True, truncation=True, return_tensors='pt', max_length=10)

# Step 3: Enhanced LLM
class ALBERTWithNumericalFeatures(nn.Module):
    def __init__(self, albert_model, numerical_input_dim, numerical_hidden_dim, output_dim):
        super(ALBERTWithNumericalFeatures, self).__init__()
        self.albert = albert_model
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(numerical_input_dim, numerical_hidden_dim)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(numerical_hidden_dim)
        self.fc2 = nn.Linear(numerical_hidden_dim, 768)
        self.fc3 = nn.Linear(768 * 2, output_dim)
    
    def forward(self, numerical_input, input_ids, attention_mask):
        # Process text with ALBERT
        albert_outputs = self.albert(input_ids=input_ids, attention_mask=attention_mask)
        albert_pooled_output = albert_outputs.pooler_output  # Use ALBERT's pooled output

        # Process numerical data
        numerical_output = self.fc1(numerical_input)
        numerical_output = self.batch_norm(numerical_output)
        numerical_output = self.relu(numerical_output)
        numerical_output = self.fc2(numerical_output)

        # Concatenate ALBERT and numerical features
        combined_output = torch.cat((albert_pooled_output, numerical_output), dim=1)
        combined_output = self.dropout(combined_output)

        # Final classification layer
        logits = self.fc3(combined_output)
        return logits

def train_model(model, train_dataloader, criterion, optimizer, scheduler, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{epochs}'):
            numerical_input, input_ids, attention_mask, labels = batch
            numerical_input, input_ids, attention_mask, labels = numerical_input.to(device), input_ids.to(device), attention_mask.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(numerical_input, input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
        
        auc_roc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
        print(f'Epoch {epoch + 1}, Loss: {total_loss/len(train_dataloader)}, AUC-ROC: {auc_roc:.4f}')

# Step 4: Evaluation
def evaluate_model(model, test_dataloader):
    model.eval()
    predictions, true_labels = [], []
    probabilities = []
    with torch.no_grad():
        for batch in test_dataloader:
            numerical_input, input_ids, attention_mask, labels = batch
            numerical_input, input_ids, attention_mask, labels = numerical_input.to(device), input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(numerical_input, input_ids, attention_mask)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            probabilities.extend(torch.softmax(outputs, dim=1).cpu().numpy())
    return predictions, true_labels, probabilities

def compute_metrics(true_labels, predictions, probabilities):
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')
    auc_roc = roc_auc_score(true_labels, probabilities, multi_class='ovr')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc
    }

def run_multiple_iterations(X, y, n_iterations=5, test_size=0.2):
    metrics_list = []

    for i in range(n_iterations):
        X_train, X_test, y_train, y_test, text_train, text_test = train_test_split(X, y, data_strings, test_size=test_size, random_state=i)
        
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Convert to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.long)
        y_test = torch.tensor(y_test.values, dtype=torch.long)

        # Create TensorDataset and DataLoader
        train_encodings = tokenizer(text_train, truncation=True, padding=True, max_length=10, return_tensors='pt')
        test_encodings = tokenizer(text_test, truncation=True, padding=True, max_length=10, return_tensors='pt')
        
        train_data = TensorDataset(X_train, train_encodings['input_ids'], train_encodings['attention_mask'], y_train)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        test_data = TensorDataset(X_test, test_encodings['input_ids'], test_encodings['attention_mask'], y_test)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

        albert_model = AlbertModel.from_pretrained('albert-base-v2')

        # Define hyperparameters
        model = ALBERTWithNumericalFeatures(albert_model, numerical_input_dim, numerical_hidden_dim, output_dim).to(device)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        train_model(model, train_dataloader, criterion, optimizer, scheduler, epochs)
        
        predictions, true_labels, probabilities = evaluate_model(model, test_dataloader)
        metrics = compute_metrics(true_labels, predictions, probabilities)
        metrics_list.append(metrics)
        print(f"Iteration {i+1} - Metrics: {metrics}")

    metrics_df = pd.DataFrame(metrics_list)
    mean_metrics = metrics_df.mean()
    std_metrics = metrics_df.std()
    
    print("\nAggregated Metrics (Mean ± Std):")
    for metric in mean_metrics.index:
        print(f"{metric.capitalize()}: {mean_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")

# Define global variables
numerical_input_dim = X.shape[1]
numerical_hidden_dim = 64
output_dim = 7
learning_rate = 1e-4
epochs = 10
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

run_multiple_iterations(X, y, n_iterations=3)
