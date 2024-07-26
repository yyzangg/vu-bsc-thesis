import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, DataCollatorForLanguageModeling, create_optimizer

df = pd.read_csv('/syn_eval_mle/syn_input_job.csv')

# Step 1: Preprocess
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.values)
data_strings = [' '.join(map(str, row)) for row in scaled_data]

# Step 2: Data Pipeline
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

class TabularDataset(tf.data.Dataset):
    def _generator(data):
        for chunk in data:
            for line in chunk:
                tokens = tokenizer(line, return_tensors='tf', padding='max_length', max_length=512, truncation=True)
                input_ids = tokens['input_ids']
                attention_mask = tokens['attention_mask']
                target_row_index = random.randint(0, len(chunk) - 1)
                target_row = chunk[target_row_index]
                target_tokens = tokenizer(target_row, return_tensors='tf', padding='max_length', max_length=512, truncation=True)
                target_ids = target_tokens['input_ids']
                yield input_ids[0], attention_mask[0], target_ids[0]

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

train_dataset = TabularDataset(train_data).batch(4)
val_dataset = TabularDataset(val_data).batch(4)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Step 3: Fine-Tuning
model = TFGPT2LMHeadModel.from_pretrained('gpt2')

num_epochs = 4
optimizer, lr_schedule = create_optimizer(
    init_lr=3e-5,
    num_train_steps=len(train_data) * num_epochs,
    num_warmup_steps=1000
)

def compute_loss(labels, logits):
    labels = tf.reshape(labels, (-1,))
    logits = tf.reshape(logits, (-1, logits.shape[-1]))
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    loss = tf.reduce_mean(loss)
    return loss

model.compile(optimizer=optimizer, loss=compute_loss)

@tf.function
def train_step(input_ids, attention_mask, labels):
    with tf.GradientTape() as tape:
        outputs = model(input_ids, attention_mask=attention_mask, training=True)
        logits = outputs.logits
        loss = compute_loss(labels, logits)
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
def generate_text(prompt, model, tokenizer, max_length=None, num_return_sequences=5):
    inputs = tokenizer(prompt, return_tensors='tf')
    input_length = tf.shape(inputs['input_ids'])[1]
    if max_length is None or max_length < input_length:
        max_length = input_length + 50
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9
    )
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

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

def generate_new_rows(input_table, model, tokenizer, scaler, num_new_rows):
    generated_rows = pd.DataFrame(columns=input_table.columns)
    current_num_rows = 0
    integer_columns = input_table.select_dtypes(include=['int']).columns

    while current_num_rows < num_new_rows:
        chunk_index = random.randint(0, len(input_table) - 1)
        prompt_chunk = input_table.iloc[chunk_index]

        prompt = ' '.join(prompt_chunk.astype(str))
        generated_text = generate_text(prompt, model, tokenizer, num_return_sequences=1)[0]

        new_row = generated_text.split(' ')
        validated_row = validate_and_convert_row(new_row, len(input_table.columns))

        denormalized_row = scaler.inverse_transform([validated_row])[0]

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
generated_table = generate_new_rows(input_table, model, tokenizer, scaler, num_generated_rows)

# Step 5: Post-process
generated_table = generated_table.applymap(lambda x: max(0, x))
generated_table = generated_table.round().astype(int)

generated_table.to_csv('out_gpt.csv', index=False)
print(generated_table.head())
