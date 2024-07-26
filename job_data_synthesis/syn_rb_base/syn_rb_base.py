import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments

df = pd.read_csv('/syn_eval_mle/syn_input_job.csv')

# Step 1: Preprocess
def serialize_row(row):
    serialized = (
        f"The raw CPU time is {row['CPUTimeRAW']} seconds. "
        f"The number of nodes is {row['NNode']}. "
        f"The number of allocated CPUs is {row['AllocCPUS']}. "
        f"The number of allocated nodes is {row['AllocNode']}. "
        f"The number of requested CPUs is {row['ReqCPUS']}. "
        f"The submit hour of the day is the {row['submit_hour_of_day']}th hour. "
        f"The submit day of the week is the {row['submit_day_of_week']}th day. "
        f"The submit day of the month is the {row['submit_day_of_month']}th day. "
        f"The waiting time is {row['waiting_time']} seconds. "
        f"The running time is {row['running_time']} seconds. "
        f"The end state of the job is type {row['state_encoded']}."
    )
    return serialized

serialized_data = df.apply(serialize_row, axis=1)
data_strings = serialized_data.tolist()

# Step 2: Data Pipeline
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
tokenized_datasets = tokenizer(data_strings, padding=True, truncation=True, max_length=256, return_tensors='pt')

tokenized_datasets['input_ids'] = tokenized_datasets['input_ids'].type(torch.long)
tokenized_datasets['attention_mask'] = tokenized_datasets['attention_mask'].type(torch.long)

class CustomDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, features):
        batch = {'input_ids': [], 'attention_mask': [], 'labels': []}
        for feature in features:
            batch['input_ids'].append(feature[0])
            batch['attention_mask'].append(feature[1])
            batch['labels'].append(feature[2])
        batch = {k: torch.stack(v) for k, v in batch.items()}
        return batch

data_collator = CustomDataCollator(tokenizer=tokenizer, mlm=False)

train_ids, val_ids, train_masks, val_masks = train_test_split(
    tokenized_datasets['input_ids'], 
    tokenized_datasets['attention_mask'], 
    test_size=0.2, 
    random_state=0
)

train_dataset = TensorDataset(train_ids, train_masks, train_ids)
val_dataset = TensorDataset(val_ids, val_masks, val_ids)

# Step 4: Fine-Tuning
model = RobertaForCausalLM.from_pretrained('roberta-base')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_steps=5000,
    save_total_limit=2,
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator
)

trainer.train()

# Step 4: Text Generation
def generate_text(prompt, model, tokenizer, max_length=256, num_return_sequences=1):
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    outputs = model.generate(
        inputs['input_ids'],
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9
    )
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

def parse_generated_text(text):
    mapping = {
        "CPUTimeRAW": "The raw CPU time is ",
        "NNode": "The number of nodes is ",
        "AllocCPUS": "The number of allocated CPUs is ",
        "AllocNode": "The number of allocated nodes is ",
        "ReqCPUS": "The number of requested CPUs is ",
        "submit_hour_of_day": "The submit hour of the day is the ",
        "submit_day_of_week": "The submit day of the week is the ",
        "submit_day_of_month": "The submit day of the month is the ",
        "waiting_time": "The waiting time is ",
        "running_time": "The running time is ",
        "state_encoded": "The end state of the job is type "
    }
    
    row = []
    for col, prefix in mapping.items():
        start = text.find(prefix) + len(prefix)
        if col in ["submit_hour_of_day", "submit_day_of_week", "submit_day_of_month"]:
            end = text.find("th", start)
        else:
            end = text.find(" ", start)
        
        value = text[start:end].strip().replace(",", "")
        try:
            row.append(float(value))
        except ValueError:
            row.append(0.0)
    return row

def generate_new_rows(input_table, model, tokenizer, scaler, num_new_rows):
    generated_rows = []
    current_num_rows = 0
    while current_num_rows < num_new_rows:
        chunk_index = random.randint(0, len(input_table) - 1)
        prompt_chunk = input_table.iloc[chunk_index]

        prompt = serialize_row(prompt_chunk)
        generated_text = generate_text(prompt, model, tokenizer, num_return_sequences=1)[0]

        try:
            new_row = parse_generated_text(generated_text)
            generated_rows.append(new_row)
            current_num_rows += 1
        except Exception as e:
            print(f"Error parsing generated text: {e}")
            continue

    generated_rows = np.array(generated_rows)
    denormalized_rows = scaler.inverse_transform(generated_rows)
    return pd.DataFrame(denormalized_rows, columns=input_table.columns)

scaler = StandardScaler()
normalized_input_table = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

num_generated_rows = 1000
generated_table = generate_new_rows(normalized_input_table, model, tokenizer, scaler, num_generated_rows)

# Step 5: Post-process
generated_table = generated_table.applymap(lambda x: max(0, x))
generated_table = generated_table.round().astype(int)

generated_table.to_csv('out_rb_base.csv', index=False)
print(generated_table.head())
