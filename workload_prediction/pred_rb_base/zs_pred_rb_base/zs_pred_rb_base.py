import pandas as pd
from transformers import pipeline
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Load data
df = pd.read_csv('/home/yzg244/pred_baselines/pred_input_job.csv')

# Define possible classes
class_names = ['CANCELLED', 'COMPLETED', 'FAILED', 'NODE_FAIL', 'OUT_OF_MEMORY', 'REQUEUED', 'TIMEOUT']

df['text'] = df.apply(lambda row: f"CPUTimeRAW: {row['CPUTimeRAW']}, NCPUS: {row['NCPUS']}, NNode: {row['NNode']}, AllocCPUS: {row['AllocCPUS']}, AllocNode: {row['AllocNode']}, ReqCPUS: {row['ReqCPUS']}, submit_hour_of_day: {row['submit_hour_of_day']}, submit_day_of_week: {row['submit_day_of_week']}, submit_day_of_month: {row['submit_day_of_month']}, waiting_time: {row['waiting_time']}, running_time: {row['running_time']}", axis=1)

# Initialize zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Batch processing
batch_size = 8
num_batches = len(df) // batch_size + 1

def classify_batch(batch_texts):
    results = classifier(batch_texts, candidate_labels=class_names)
    return [result['labels'][0] for result in results]

predicted_classes = []
for i in range(num_batches):
    batch_texts = df['text'][i*batch_size:(i+1)*batch_size].tolist()
    if batch_texts:  # Ensure the batch is not empty
        predicted_classes.extend(classify_batch(batch_texts))

# Truncate predictions list to match dataframe length
predicted_classes = predicted_classes[:len(df)]

# Check unique predicted classes
print("Unique predicted classes:", set(predicted_classes))

df['predicted_class'] = predicted_classes

# Evaluate the results
y_true = df['state_encoded']
y_pred = df['predicted_class']

# Convert predicted class labels to numeric values for evaluation
class_to_index = {class_name: index for index, class_name in enumerate(class_names)}
y_pred_numeric = y_pred.map(class_to_index)

# Check for any unmapped classes
print("Unmapped classes:", y_pred[~y_pred.isin(class_to_index.keys())].unique())

# Calculate evaluation metrics
accuracy = accuracy_score(y_true, y_pred_numeric)
precision = precision_score(y_true, y_pred_numeric, average='weighted', zero_division=0)
recall = recall_score(y_true, y_pred_numeric, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred_numeric, average='weighted', zero_division=0)

print("--- Zero-Shot Classification Model ---")
print(classification_report(y_true, y_pred_numeric, target_names=class_names, digits=2, zero_division=0))
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")
