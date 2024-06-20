import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, DataCollatorForLanguageModeling, create_optimizer
import tensorflow as tf

# Step 1: Load the Dataset
print("Loading the dataset...")
df = pd.read_csv('/home/yzg244/syn_gpt/syn_input_encoded.csv')

# Step 2: Preprocess
# Convert each block of rows to a single string for better context
def format_data_block(df, block_size):
    data_blocks = []
    for i in range(0, len(df), block_size):
        block = df.iloc[i:i+block_size]
        data_blocks.append('\n'.join(block.apply(lambda row: ' '.join(row.astype(str)), axis=1)))
    return data_blocks

block_size = 5 # FixMe
data_blocks = format_data_block(df.head(1000), block_size) # FixMe

# Step 3: Fine-Tune GPT-2
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set padding token
tokenizer.pad_token = tokenizer.eos_token  # Using end-of-sequence token as padding token

class TabularDataset(tf.data.Dataset):
    def _generator(data_blocks):
        for block in data_blocks:
            inputs = tokenizer(block, return_tensors='tf', padding='max_length', max_length=512, truncation=True)
            input_ids = inputs['input_ids'][0]
            attention_mask = inputs['attention_mask'][0]
            yield input_ids, attention_mask

    def __new__(cls, data_blocks):
        return tf.data.Dataset.from_generator(
            lambda: cls._generator(data_blocks),
            output_signature=(
                tf.TensorSpec(shape=(512,), dtype=tf.int32),
                tf.TensorSpec(shape=(512,), dtype=tf.int32)
            )
        )

print("Creating dataset and data collator...")

# Split data into training and validation sets
train_data, val_data = train_test_split(data_blocks, test_size=0.1, random_state=0)

# Use appropriate batch sizes
train_batch_size = 3 # FixMe
val_batch_size = 3  # FixMe

train_dataset = TabularDataset(train_data).batch(train_batch_size)
val_dataset = TabularDataset(val_data).batch(val_batch_size)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
model = TFGPT2LMHeadModel.from_pretrained('gpt2')

# Compile the model
num_epochs = 10 # FixMe
optimizer, lr_schedule = create_optimizer(
    init_lr=5e-5,
    num_train_steps=len(train_data) * num_epochs,  # num_epochs
    num_warmup_steps=0
)

model.compile(optimizer=optimizer, loss=model.compute_loss)  # Specify loss function

# Define custom training step
@tf.function
def train_step(input_ids, attention_mask):
    with tf.GradientTape() as tape:
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
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
    for step, (input_ids, attention_mask) in enumerate(train_dataset):
        loss = train_step(input_ids, attention_mask)
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
        temperature=0.7  # Adjust temperature for more diverse outputs
    )
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

def validate_and_convert_row(row, num_columns):
    validated_row = []
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

def generate_new_rows(input_table, model, tokenizer, num_new_rows, block_size):
    generated_rows = pd.DataFrame(columns=input_table.columns)
    current_num_rows = 0

    while current_num_rows < num_new_rows:
        # Use a block of rows as context
        start_index = random.randint(0, len(input_table) - block_size)
        prompt_block = '\n'.join(input_table.iloc[start_index:start_index+block_size].apply(lambda row: ' '.join(row.astype(str)), axis=1))

        # Generate new rows based on the prompt block
        generated_texts = generate_text(prompt_block, model, tokenizer, num_return_sequences=block_size)
        
        for generated_text in generated_texts:
            new_row = generated_text.split('\n')[-1].split(' ')  # Get the last row from generated block
            validated_row = validate_and_convert_row(new_row, len(input_table.columns))
            generated_rows = pd.concat([generated_rows, pd.DataFrame([validated_row], columns=input_table.columns)], ignore_index=True)
            current_num_rows += 1
            if current_num_rows >= num_new_rows:
                break

    return generated_rows

num_generated_rows = 200  # Number of rows to generate
generated_table = generate_new_rows(df, model, tokenizer, num_generated_rows, block_size)

generated_table.to_csv('syn_output_gpt.csv', index=False)
print(generated_table.head())
