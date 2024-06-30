import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Model, GPT2Tokenizer, AdamW, GPT2Config
from torch import nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('/home/yzg244/pred_baselines/pred_input_job.csv')

# Separate features and target
X = df.drop(columns=['state_encoded'])
y = df['state_encoded']

# Convert to lists for PyTorch Dataset
X = X.values.tolist()
y = y.values.tolist()

# Ensure y is in the correct format for classification (integers)
y = [int(label) for label in y]

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

# Define Dataset and DataLoader
class JobDataset(Dataset):
    def __init__(self, X, y, tokenizer, max_length=128):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        text = ' '.join(map(str, self.X[idx]))  # Combine features into a single text
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

# Initialize the tokenizer and GPT-2 model
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
config = GPT2Config.from_pretrained(model_name, output_hidden_states=False, num_labels=7)
model = GPT2Model.from_pretrained(model_name, config=config)

# Define GPTClassifier for Classification
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

# Create Dataset and DataLoader
train_dataset = JobDataset(X_train, y_train, tokenizer)
val_dataset = JobDataset(X_val, y_val, tokenizer)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Initialize the classifier model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier_model = GPTClassifier(model, num_labels=7).to(device)

# Training Setup
epochs = 5
learning_rate = 3e-5
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(classifier_model.parameters(), lr=learning_rate)

# Training Loop
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

    # Validation
    classifier_model.eval()
    with torch.no_grad():
        all_preds = []
        all_labels = []
        for input_ids, labels in val_loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            outputs = classifier_model(input_ids)
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        accuracy = accuracy_score(all_labels, all_preds)
        print(f'Validation Accuracy: {accuracy:.4f}')
    classifier_model.train()

# Evaluation
def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            outputs = model(input_ids)
            probs = nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return all_preds, all_labels, all_probs

# Evaluate the model
predictions, true_labels, probabilities = evaluate_model(classifier_model, val_loader)

accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions, average='weighted')
recall = recall_score(true_labels, predictions, average='weighted')
f1 = f1_score(true_labels, predictions, average='weighted')
auc_roc = roc_auc_score(true_labels, probabilities, multi_class='ovr')

class_names = ['CANCELLED', 'COMPLETED', 'FAILED', 'NODE_FAIL', 'OUT_OF_MEMORY', 'REQUEUED', 'TIMEOUT']

print("--- GPT-2 Model ---")
print(classification_report(true_labels, predictions, target_names=class_names, digits=2))
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")
print(f"AUC-ROC: {auc_roc:.2f}")
