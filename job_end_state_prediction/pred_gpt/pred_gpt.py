import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Model, GPT2Tokenizer, AdamW, GPT2Config
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

df = pd.read_csv('/pred_baselines/resampled_pred_input_job.csv')

# Step 1: Pre-Process
X = df[['ReqCPUS', 'submit_hour_of_day', 'submit_day_of_week', 'submit_day_of_month']]
y = df['state_encoded']

# Convert to lists for PyTorch Dataset
X = X.values.tolist()
y = y.values.tolist()
y = [int(label) for label in y]

# Step 2: Data Pipeline
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

class JobDataset(Dataset):
    def __init__(self, X, y, tokenizer, max_length=128):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        text = ' '.join(map(str, self.X[idx]))
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].squeeze(0)  # Remove batch dimension
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return input_ids, label

class GPTClassifier(nn.Module):
    def __init__(self, gpt_model, num_labels):
        super(GPTClassifier, self).__init__()
        self.gpt_model = gpt_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(gpt_model.config.hidden_size, num_labels)
        
    def forward(self, input_ids):
        outputs = self.gpt_model(input_ids=input_ids)[0]  # Get hidden states
        pooled_output = outputs[:, -1, :]  # Take the output of the last token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# Step 3: Fine-Tuning
def train_and_evaluate(X_train, y_train, X_val, y_val, epochs=5, batch_size=16, learning_rate=3e-5):
    train_dataset = JobDataset(X_train, y_train, tokenizer)
    val_dataset = JobDataset(X_val, y_val, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    config = GPT2Config.from_pretrained('gpt2', output_hidden_states=False, num_labels=7)
    model = GPT2Model.from_pretrained('gpt2', config=config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier_model = GPTClassifier(model, num_labels=7).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(classifier_model.parameters(), lr=learning_rate)
    
    classifier_model.train()

    for epoch in range(epochs):
        total_loss = 0
        for input_ids, labels in train_loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = classifier_model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

    classifier_model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for input_ids, labels in val_loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            outputs = classifier_model(input_ids)
            probs = nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return all_preds, all_labels, all_probs

# Step 4: Evaluation
def compute_metrics(true_labels, predictions, probabilities):
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted', zero_division=1)
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
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=i)
        predictions, true_labels, probabilities = train_and_evaluate(X_train, y_train, X_val, y_val)
        metrics = compute_metrics(true_labels, predictions, probabilities)
        metrics_list.append(metrics)
        print(f"Iteration {i+1} - Metrics: {metrics}")

    metrics_df = pd.DataFrame(metrics_list)
    mean_metrics = metrics_df.mean()
    std_metrics = metrics_df.std()
    
    print("\nAggregated Metrics (Mean ± Std):")
    for metric in mean_metrics.index:
        print(f"{metric.capitalize()}: {mean_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")

run_multiple_iterations(X, y, n_iterations=5)
