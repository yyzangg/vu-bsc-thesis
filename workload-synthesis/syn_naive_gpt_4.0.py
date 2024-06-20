import os
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, DataCollatorForLanguageModeling, create_optimizer
import tensorflow as tf

# Step 1: Load the Dataset
print("Loading the dataset...")
df = pd.read_csv('/home/yzg244/syn_mle/syn_input_job.csv')

# Step 2: Preprocess
# Convert each row to a string
data_strings = df.apply(lambda row: ' '.join(row.astype(str)), axis=1).tolist()

# Step 3: Data Chunking and Aggregation
chunk_size = 1000  # Number of rows per chunk
overlap_size = 50  # Number of overlapping rows between chunks

chunked_data = [data_strings[i:i+chunk_size] for i in range(0, len(data_strings), chunk_size - overlap_size)]

# Compute global summary statistics
global_mean = np.mean(df.values, axis=0)
global_std = np.std(df.values, axis=0)

# Step 4: Fine-Tune GPT-2
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set padding token
tokenizer.pad_token = tokenizer.eos_token  # Using end-of-sequence token as padding token

class TabularDataset(tf.data.Dataset):
    def _generator(data):
        for chunk in data:
            for line in chunk:
                input_ids = tokenizer(line, return_tensors='tf', padding='max_length', max_length=512, truncation=True)['input_ids']
                target_row_index = random.randint(0, len(chunk) - 1)
                target_row = chunk[target_row_index]
                target_ids = tokenizer(target_row, return_tensors='tf', padding='max_length', max_length=512, truncation=True)['input_ids']
                yield input_ids[0], target_ids[0]

    def __new__(cls, data):
        return tf.data.Dataset.from_generator(
            lambda: cls._generator(data),
            output_signature=(tf.TensorSpec(shape=(512,), dtype=tf.int32), tf.TensorSpec(shape=(512,), dtype=tf.int32))
        )

print("Creating dataset and data collator...")

# Split data into training and validation sets
train_data, val_data = train_test_split(chunked_data, test_size=0.1, random_state=42)

# Use appropriate batch sizes
train_batch_size = 3
val_batch_size = 3

train_dataset = TabularDataset(train_data).batch(train_batch_size)
val_dataset = TabularDataset(val_data).batch(val_batch_size)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
model = TFGPT2LMHeadModel.from_pretrained('gpt2')

# Compile the model
num_epochs = 3  # FixMe
optimizer, lr_schedule = create_optimizer(
    init_lr=5e-5,
    num_train_steps=len(train_data) * num_epochs,
    num_warmup_steps=0
)

# Define custom loss function
def compute_loss(labels, logits):
    labels = tf.reshape(labels, (-1,))
    logits = tf.reshape(logits, (-1, logits.shape[-1]))
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    loss = tf.reduce_mean(loss)
    return loss

# Compile the model with the custom loss function
model.compile(optimizer=optimizer, loss=None)

# Define custom training step
@tf.function
def train_step(input_ids, labels):
    with tf.GradientTape() as tape:
        outputs = model(input_ids, training=True)
        logits = outputs.logits
        loss = compute_loss(labels, logits)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Create a custom training loop
steps_per_epoch = len(train_data) // train_batch_size
best_loss = float('inf')

# Ensure the directory for saving models exists
checkpoint_dir = 'model_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

print("Starting training...")

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    for step, (input_ids, labels) in enumerate(train_dataset):
        loss = train_step(input_ids, labels)
        loss_value = tf.reduce_mean(loss).numpy()  # Convert loss to a scalar
        if step % 10 == 0:
            print(f"Step {step}/{steps_per_epoch}, Loss: {loss_value}")

    # Save the model if the current epoch loss is better than the best loss
    if loss_value < best_loss:
        best_loss = loss_value
        model.save_pretrained(checkpoint_dir)
        print(f"Model saved at epoch {epoch+1} with loss {best_loss}")

# Step 7: Generate New Data
def generate_text(prompt, model, tokenizer, max_length=None, num_return_sequences=5):
    inputs = tokenizer(prompt, return_tensors='tf')
    input_length = tf.shape(inputs['input_ids'])[1]
    if max_length is None or max_length < input_length:
        max_length = input_length + 50  # Ensuring max_length is larger than input length
    outputs = model.generate(
        inputs['input_ids'],
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        do_sample=True,  # Enable sampling to allow num_return_sequences > 1
        top_k=50,  # Control the sampling method
        top_p=0.95,  # Control the sampling method
        temperature=0.9  # Increase temperature for more diverse outputs
    )
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

def validate_and_convert_row(row, num_columns):
    """
    Validate and convert each item in the row to an integer if input values are integers.
    """
    validated_row = []
    is_all_integers = all(isinstance(item, int) for item in row)

    if is_all_integers:
        for item in row:
            validated_row.append(round(item))  # Round to the nearest integer
    else:
        for item in row:
            try:
                validated_row.append(round(float(item), 2))  # Round to two decimal places
            except ValueError:
                validated_row.append(0.00)  # Replace with a default value

    # Ensure the number of columns matches
    if len(validated_row) < num_columns:
        validated_row.extend([0.00] * (num_columns - len(validated_row)))  # Pad with zeros
    elif len(validated_row) > num_columns:
        validated_row = validated_row[:num_columns]  # Trim to the required number of columns

    return validated_row

def generate_new_rows(input_table, model, tokenizer, global_mean, global_std, num_new_rows):
    generated_rows = pd.DataFrame(columns=input_table.columns)
    current_num_rows = 0

    # Identify columns that are integers in the input
    integer_columns = input_table.select_dtypes(include=['int']).columns

    while current_num_rows < num_new_rows:
        # Select a random chunk from the existing table as prompt
        chunk_index = random.randint(0, len(input_table) - 1)
        prompt_chunk = input_table.iloc[chunk_index]  # Select chunk by index

        # Generate a new row based on the prompt chunk
        prompt = ' '.join(prompt_chunk.astype(str))
        generated_text = generate_text(prompt, model, tokenizer, num_return_sequences=1)[0]

        # Split generated text back into columns
        new_row = generated_text.split(' ')

        # Validate and convert the generated row
        validated_row = validate_and_convert_row(new_row, len(input_table.columns))

        # Denormalize the generated row using global statistics
        denormalized_row = (np.array(validated_row) * global_std) + global_mean

        # Convert integer columns back to integers
        for col in integer_columns:
            col_idx = input_table.columns.get_loc(col)
            denormalized_row[col_idx] = int(round(denormalized_row[col_idx]))

        # Convert the row to a DataFrame
        new_row_df = pd.DataFrame([denormalized_row], columns=input_table.columns)

        # Concatenate the new row to the generated_rows DataFrame
        generated_rows = pd.concat([generated_rows, new_row_df], ignore_index=True)

        current_num_rows += 1

    return generated_rows

# Flatten the chunked data into a single DataFrame
flattened_data = [item for sublist in chunked_data for item in sublist]
input_table = pd.DataFrame([x.split(' ') for x in flattened_data], columns=df.columns)

num_generated_rows = 400

# Generate new rows
generated_table = generate_new_rows(input_table, model, tokenizer, global_mean, global_std, num_generated_rows)

# Ensure the generated values are integers
generated_table = generated_table.round().astype(int)

# Ensure there are no negtive values
generated_table = generated_table.applymap(lambda x: max(x, -x))

generated_table.to_csv('out_gpt.csv', index=False)
print(generated_table.head())
