import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from transformers import RobertaTokenizer, RobertaModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

# Load data
df = pd.read_csv('/path/to/new/dataset.csv')
df = df.sample(10000)

# Extract features and labels
X = df[['CPUTimeRAW', 'NCPUS', 'NNode', 'AllocCPUS', 'AllocNode', 
        'ReqCPUS', 'submit_hour_of_day', 'submit_day_of_week', 
        'submit_day_of_month', 'waiting_time', 'running_time']]
y = df['state_encoded']

# Create dummy text input
dummy_text = ['[CLS]'] * len(df)

# Train-test split
X_train, X_test, y_train, y_test, text_train, text_test = train_test_split(X, y, dummy_text, test_size=0.2, random_state=0)

# Scale numerical data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.long)
y_test = torch.tensor(y_test.values, dtype=torch.long)

# Tokenize text data for RoBERTa
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
train_encodings = tokenizer(text_train, truncation=True, padding=True, max_length=10, return_tensors='pt')
test_encodings = tokenizer(text_test, truncation=True, padding=True, max_length=10, return_tensors='pt')

# Batch size for training
batch_size = 32

# Create TensorDataset and DataLoader
train_data = TensorDataset(X_train, train_encodings['input_ids'], train_encodings['attention_mask'], y_train)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

test_data = TensorDataset(X_test, test_encodings['input_ids'], test_encodings['attention_mask'], y_test)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

class RoBERTaWithNumericalFeatures(nn.Module):
    def __init__(self, roberta_model, numerical_input_dim, numerical_hidden_dim, output_dim):
        super(RoBERTaWithNumericalFeatures, self).__init__()
        self.roberta = roberta_model
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(numerical_input_dim, numerical_hidden_dim)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(numerical_hidden_dim)
        self.fc2 = nn.Linear(numerical_hidden_dim, 768)
        self.fc3 = nn.Linear(768 * 2, 512)
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

# Load RoBERTa model
roberta_model = RobertaModel.from_pretrained('roberta-base')

# Define hyperparameters
numerical_input_dim = X_train.shape[1]
numerical_hidden_dim = 128
output_dim = 2  # Binary classification
learning_rate = 3e-5
epochs = 50

# Create the combined model
model = RoBERTaWithNumericalFeatures(roberta_model, numerical_input_dim, numerical_hidden_dim, output_dim)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_steps, num_training_steps=total_steps)

# Move model to device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to train the model
def train_model(model, train_dataloader, criterion, optimizer, scheduler, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
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
        print(f'Epoch {epoch + 1}, Loss: {total_loss/len(train_dataloader)}')

# Train the model
train_model(model, train_dataloader, criterion, optimizer, scheduler, epochs)

# Function to evaluate the model
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
    
    probabilities = np.array(probabilities)

    return predictions, true_labels, probabilities

# Evaluate the model
predictions, true_labels, probabilities = evaluate_model(model, test_dataloader)

accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
auc_roc = roc_auc_score(true_labels, probabilities[:, 1])  # AUC-ROC for binary classification

class_names = ['FAILED', 'COMPLETED']

print("--- RoBERTa-Base Model ---")
print(classification_report(true_labels, predictions, target_names=class_names, digits=2, zero_division=0))
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")
print(f"AUC-ROC: {auc_roc:.2f}")
