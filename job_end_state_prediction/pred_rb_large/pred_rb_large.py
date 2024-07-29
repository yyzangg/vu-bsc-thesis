import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, classification_report, accuracy_score
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import RobertaTokenizer, RobertaModel, AdamW, get_linear_schedule_with_warmup

df = pd.read_csv('/pred_baselines/resampled_pred_input_job.csv')

# Step 1: Preprocess
X = df[['ReqCPUS', 'submit_hour_of_day', 'submit_day_of_week', 'submit_day_of_month']]
y = df['state_encoded']

scaler = StandardScaler()
scaled_data = scaler.fit_transform(X.values)
data_strings = [' '.join(map(str, row)) for row in scaled_data]

# Step 3: Enhanced LLM
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

class RoBERTaWithNumericalFeatures(nn.Module):
    def __init__(self, roberta_model, numerical_input_dim, numerical_hidden_dim, output_dim):
        super(RoBERTaWithNumericalFeatures, self).__init__()
        self.roberta = roberta_model
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(numerical_input_dim, numerical_hidden_dim)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(numerical_hidden_dim)
        self.fc2 = nn.Linear(numerical_hidden_dim, 768)
        self.fc3 = nn.Linear(1024 + 768, 512)
        self.fc4 = nn.Linear(512, output_dim)

    def forward(self, numerical_input, input_ids, attention_mask):
        # Process text with RoBERTa
        roberta_outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        roberta_pooled_output = roberta_outputs.pooler_output

        # Process numerical data
        numerical_output = self.fc1(numerical_input)
        numerical_output = self.batch_norm(numerical_output)
        numerical_output = self.relu(numerical_output)
        numerical_output = self.fc2(numerical_output)

        # Concatenate RoBERTa and numerical features
        combined_output = torch.cat((roberta_pooled_output, numerical_output), dim=1)
        combined_output = self.dropout(combined_output)

        # Additional hidden layer
        combined_output = self.fc3(combined_output)
        combined_output = self.relu(combined_output)
        combined_output = self.dropout(combined_output)

        # Final classification layer
        logits = self.fc4(combined_output)
        return logits

def train_model(model, train_dataloader, criterion, optimizer, scheduler, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
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

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1).detach().cpu().numpy())
        
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
            probs = torch.softmax(outputs, dim=1)
            probabilities.extend(probs.cpu().numpy())
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return predictions, true_labels, probabilities

def compute_metrics(true_labels, predictions, probabilities):
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')
    auc_roc = roc_auc_score(pd.get_dummies(true_labels), probabilities, multi_class='ovr')
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc
    }

def run_multiple_iterations(X, y, n_iterations=2, test_size=0.2):
    metrics_list = []

    for i in range(n_iterations):
        print(f"Running iteration {i + 1}/{n_iterations}...")

        # Step 2: Data Pipeline
        X_train, X_test, y_train, y_test, text_train, text_test = train_test_split(X, y, data_strings, test_size=test_size, random_state=i)
        
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Convert to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.long)
        y_test = torch.tensor(y_test.values, dtype=torch.long)

        # Define numerical input dimension after scaling
        numerical_input_dim = X_train.shape[1]

        # Tokenize text data
        train_encodings = tokenizer(text_train, truncation=True, padding=True, max_length=10, return_tensors='pt')
        test_encodings = tokenizer(text_test, truncation=True, padding=True, max_length=10, return_tensors='pt')

        # Create TensorDataset and DataLoader
        train_data = TensorDataset(X_train, train_encodings['input_ids'], train_encodings['attention_mask'], y_train)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        test_data = TensorDataset(X_test, test_encodings['input_ids'], test_encodings['attention_mask'], y_test)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

        roberta_model = RobertaModel.from_pretrained('roberta-large')

        # Define hyperparameters
        model = RoBERTaWithNumericalFeatures(roberta_model, numerical_input_dim, numerical_hidden_dim, output_dim).to(device)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_steps, num_training_steps=total_steps)

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
numerical_hidden_dim = 256
output_dim = 7
learning_rate = 2e-5
epochs = 20
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

run_multiple_iterations(X, y, n_iterations=2)