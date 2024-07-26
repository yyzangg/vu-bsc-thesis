import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, TFDistilBertForMaskedLM, DataCollatorForLanguageModeling, create_optimizer

df = pd.read_csv('/syn_eval_mle/syn_input_job.csv')

# Step 1: Preprocess
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.values)
data_strings = [' '.join(map(str, row)) for row in scaled_data]

# Step 2: Data Pipeline
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

class TabularDataset(tf.data.Dataset):
    def _generator(data):
        for chunk in data:
            for line in chunk:
                tokens = tokenizer(line, return_tensors='tf', padding='max_length', max_length=512, truncation=True)
                input_ids = tokens['input_ids']
                attention_mask = tokens['attention_mask']
                yield input_ids[0], attention_mask[0], input_ids[0]

    def __new__(cls, data):
        return tf.data.Dataset.from_generator(
            lambda: cls._generator(data),
            output_signature=(
                tf.TensorSpec(shape=(512,), dtype=tf.int32),
                tf.TensorSpec(shape=(512,), dtype=tf.int32),
                tf.TensorSpec(shape=(512,), dtype=tf.int32)
            )
        )

chunk_size = 500
overlap_size = 20
chunked_data = [data_strings[i:i + chunk_size] for i in range(0, len(data_strings), chunk_size - overlap_size)]

train_data, val_data = train_test_split(chunked_data, test_size=0.2, random_state=0)

train_dataset = TabularDataset(train_data).batch(8)
val_dataset = TabularDataset(val_data).batch(4)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)

# Step 3: Fine-Tuning
model = TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
model.resize_token_embeddings(len(tokenizer))

num_epochs = 6
optimizer, lr_schedule = create_optimizer(
    init_lr=3e-5,
    num_train_steps=len(train_data) * num_epochs,
    num_warmup_steps=1500
)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

@tf.function
def train_step(input_ids, attention_mask, labels):
    with tf.GradientTape() as tape:
        outputs = model(input_ids, attention_mask=attention_mask, training=True)
        logits = outputs.logits
        loss = loss_fn(labels, logits)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

steps_per_epoch = len(train_data) // 8
best_loss = float('inf')

checkpoint_dir = 'model_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    epoch_loss = []
    for step, (input_ids, attention_mask, labels) in enumerate(train_dataset):
        loss = train_step(input_ids, attention_mask, labels)
        loss_value = tf.reduce_mean(loss).numpy()
        epoch_loss.append(loss_value)
        if step % 10 == 0:
            print(f"Step {step}/{steps_per_epoch}, Loss: {loss_value}")

    avg_epoch_loss = np.mean(epoch_loss)
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        model.save_pretrained(checkpoint_dir)
        print(f"Model saved at epoch {epoch+1} with loss {best_loss}")

# Step 4: Text Generation
def generate_text_with_masking(prompt, model, tokenizer, max_new_tokens=50, temperature=1.2):
    inputs = tokenizer(prompt, return_tensors='tf')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    for _ in range(max_new_tokens):
        mask_token_index = tf.where(input_ids == tokenizer.mask_token_id)

        if mask_token_index.shape[0] == 0:
            break

        mask_token_index = mask_token_index[0].numpy()

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        mask_token_logits = logits[0, mask_token_index[1], :] / temperature
        mask_token_prob = tf.nn.softmax(mask_token_logits)
        mask_token_id = tf.random.categorical(tf.math.log([mask_token_prob]), num_samples=1).numpy()[0][0]
        
        input_ids = tf.concat([input_ids[:, :mask_token_index[1]], [[mask_token_id]], input_ids[:, mask_token_index[1] + 1:]], axis=-1)
        attention_mask = tf.concat([attention_mask, [[1]]], axis=-1)

    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text

def validate_and_convert_row(row, num_columns):
    validated_row = []
    is_all_integers = all(isinstance(item, int) for item in row)

    if is_all_integers:
        for item in row:
            validated_row.append(round(item))
    else:
        for item in row:
            try:
                validated_row.append(round(float(item), 2))
            except ValueError:
                validated_row.append(0.00)

    if len(validated_row) < num_columns:
        validated_row.extend([0.00] * (num_columns - len(validated_row)))
    elif len(validated_row) > num_columns:
        validated_row = validated_row[:num_columns]

    return validated_row

def add_noise(row, noise_level=0.05):
    return row + np.random.normal(scale=noise_level, size=row.shape)

def generate_new_rows(input_table, model, tokenizer, scaler, num_new_rows, temperature=1.2, noise_level=0.05):
    generated_rows = pd.DataFrame(columns=input_table.columns)
    current_num_rows = 0
    integer_columns = input_table.select_dtypes(include=['int']).columns

    unique_prompts = input_table.sample(n=min(100, len(input_table)), random_state=0).astype(str).values.tolist()

    while current_num_rows < num_new_rows:
        prompt_chunk = random.choice(unique_prompts)
        prompt = ' '.join(prompt_chunk)

        generated_text = generate_text_with_masking(prompt, model, tokenizer, max_new_tokens=50, temperature=temperature)
        new_row = generated_text.split(' ')
        validated_row = validate_and_convert_row(new_row, len(input_table.columns))

        denormalized_row = scaler.inverse_transform([validated_row])[0]

        denormalized_row = add_noise(denormalized_row, noise_level=noise_level)

        for col in integer_columns:
            col_idx = input_table.columns.get_loc(col)
            denormalized_row[col_idx] = int(round(denormalized_row[col_idx]))

        new_row_df = pd.DataFrame([denormalized_row], columns=input_table.columns)
        generated_rows = pd.concat([generated_rows, new_row_df], ignore_index=True)
        current_num_rows += 1

    return generated_rows

flattened_data = [item for sublist in chunked_data for item in sublist]
input_table = pd.DataFrame([x.split(' ') for x in flattened_data], columns=df.columns)
input_table = pd.DataFrame(scaler.inverse_transform(input_table.values), columns=df.columns)

num_generated_rows = 1000
generated_table = generate_new_rows(input_table, model, tokenizer, scaler, num_generated_rows, temperature=1.2, noise_level=0.05)

# Step 5: Post-process
generated_table = generated_table.applymap(lambda x: max(0, x))
generated_table = generated_table.round().astype(int)

generated_table.to_csv('out_db_base.csv', index=False)
print(generated_table.head())
