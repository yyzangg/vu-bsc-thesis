import os
import random
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from transformers import BartTokenizer, BartForConditionalGeneration, TrainingArguments, Trainer

df = pd.read_csv('/syn_eval_mle/syn_input_job.csv')

# Step 1: Preprocess
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.values)
data_strings = [' '.join(map(str, row)) for row in scaled_data]

# Step 2: Data Pipeline
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

class TabularDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length=512, truncation=True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation = truncation
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        chunk = self.data[index]
        input_row = random.choice(chunk)
        target_row = random.choice(chunk)  # For simplicity, target can also be input in unsupervised setting
        
        input_ids = self.tokenizer.encode(input_row, truncation=self.truncation, max_length=self.max_length, padding='max_length', return_tensors='pt')
        target_ids = self.tokenizer.encode(target_row, truncation=self.truncation, max_length=self.max_length, padding='max_length', return_tensors='pt')
        
        return {
            'input_ids': input_ids.flatten(),
            'labels': target_ids.flatten()
        }

chunk_size = 500
overlap_size = 20
chunked_data = [data_strings[i:i + chunk_size] for i in range(0, len(data_strings), chunk_size - overlap_size)]

train_data, val_data = train_test_split(chunked_data, test_size=0.2, random_state=0)

# Step 3: Fine-Tuning
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base').to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    per_device_train_batch_size=4,
    num_train_epochs=4,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=500,
    save_total_limit=1,
)

def train_model(model, tokenizer, train_data, training_args):
    train_dataset = TabularDataset(train_data, tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )
    trainer.train()

train_model(model, tokenizer, train_data, training_args)

# Step 4: Text Generation
def generate_new_rows(model, tokenizer, scaler, num_new_rows):
    generated_rows = []
    for _ in range(num_new_rows):
        input_row = random.choice(data_strings)
        input_ids = tokenizer.encode(input_row, max_length=512, truncation=True, padding='max_length', return_tensors='pt').to(device)
        
        outputs = model.generate(input_ids, max_length=512, num_return_sequences=1, num_beams=5, early_stopping=True)
        generated_row = tokenizer.decode(outputs[0], skip_special_tokens=True)

        generated_rows.append(generated_row)
    
    return generated_rows

num_generated_rows = 1000
generated_rows = generate_new_rows(model, tokenizer, scaler, num_generated_rows)

# Step 5: Post-Process
generated_data = [row.split() for row in generated_rows]
num_columns = len(df.columns)

correct_length_data = []
for row in generated_data:
    validated_row = []
    for item in row:
        try:
            float_value = float(item)
            validated_row.append(float_value)
        except ValueError:
            validated_row.append(0.0)
    
    if len(validated_row) < num_columns:
        validated_row.extend([0.0] * (num_columns - len(validated_row)))
    elif len(validated_row) > num_columns:
        validated_row = validated_row[:num_columns]
    
    correct_length_data.append(validated_row)

denormalized_data = scaler.inverse_transform(correct_length_data)
generated_df = pd.DataFrame(denormalized_data, columns=df.columns)
generated_df = generated_df.applymap(lambda x: max(0, float(x))).round().astype(int)

generated_df.to_csv('out_bart_base.csv', index=False)
print(generated_df.head())
