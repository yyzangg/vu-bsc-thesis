import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, DataCollatorForLanguageModeling, create_optimizer
import tensorflow as tf

# Step 1: Load the Dataset
print("Loading the dataset...")
df = pd.read_csv('/home/yzg244/syn_input_encoded.csv')

# Step 2: Preprocess
# Convert each row to a string
data_strings = df.apply(lambda row: ' '.join(row.astype(str)), axis=1).tolist()

# Step 3: Sample the Data
sample_size = 100  # FixMe
if len(data_strings) > sample_size:
    data_strings = data_strings[:sample_size]

# Step 4: Fine-Tune GPT-2
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set padding token
tokenizer.pad_token = tokenizer.eos_token  # Using end-of-sequence token as padding token

class TabularDataset(tf.data.Dataset):
    def _generator(data):
        for line in data:
            input_ids = tokenizer(line, return_tensors='tf', padding='max_length', max_length=512, truncation=True)['input_ids']
            target_row_index = random.randint(0, len(data) - 1)
            target_row = data[target_row_index]
            target_ids = tokenizer(target_row, return_tensors='tf', padding='max_length', max_length=512, truncation=True)['input_ids']
            yield input_ids[0], target_ids[0]

    def __new__(cls, data):
        return tf.data.Dataset.from_generator(
            lambda: cls._generator(data),
            output_signature=(tf.TensorSpec(shape=(512,), dtype=tf.int32), tf.TensorSpec(shape=(512,), dtype=tf.int32))
        )

print("Creating dataset and data collator...")

# Split data into training and validation sets
train_data, val_data = train_test_split(data_strings, test_size=0.1, random_state=42)

# Use appropriate batch sizes
train_batch_size = 2
val_batch_size = 2  # Smaller batch size for validation

train_dataset = TabularDataset(train_data).batch(train_batch_size)
val_dataset = TabularDataset(val_data).batch(val_batch_size)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
model = TFGPT2LMHeadModel.from_pretrained('gpt2')

# Compile the model
optimizer, lr_schedule = create_optimizer(
    init_lr=5e-5,
    num_train_steps=len(train_data) * 3,  # num_epochs
    num_warmup_steps=0
)

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')  # Specify loss function

# Define custom training step
@tf.function
def train_step(input_ids, labels):
    with tf.GradientTape() as tape:
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Create a custom training loop
num_epochs = 3
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
        temperature=0.7  # Adjust temperature for more diverse outputs
    )
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

def validate_and_convert_row(row, num_columns):
    """
    Validate and convert each item in the row to a float with two decimal places. If conversion fails, replace with a default value.
    """
    validated_row = []
    for item in row:
        try:
            # Attempt to convert to float and round to two decimal places
            validated_row.append(round(float(item), 2))
        except ValueError:
            # Replace with a default value or handle the error as needed
            validated_row.append(0.00)
    
    # Ensure the number of columns matches
    if len(validated_row) < num_columns:
        validated_row.extend([0.00] * (num_columns - len(validated_row)))  # Pad with zeros
    elif len(validated_row) > num_columns:
        validated_row = validated_row[:num_columns]  # Trim to the required number of columns

    return validated_row

def generate_new_rows(input_table, model, tokenizer, num_new_rows):
    generated_rows = pd.DataFrame(columns=input_table.columns)
    current_num_rows = 0

    while current_num_rows < num_new_rows:
        # Select a random row from the existing table as prompt
        prompt_index = random.randint(0, len(input_table) - 1)
        prompt = ' '.join(input_table.iloc[prompt_index].astype(str))

        # Generate a new row based on the prompt
        generated_text = generate_text(prompt, model, tokenizer, num_return_sequences=1)[0]

        # Split generated text back into columns
        new_row = generated_text.split(' ')

        # Validate and convert the generated row
        validated_row = validate_and_convert_row(new_row, len(input_table.columns))

        generated_rows = pd.concat([generated_rows, pd.DataFrame([validated_row], columns=input_table.columns)], ignore_index=True)
        current_num_rows += 1

    return generated_rows

num_generated_rows = 200  # FixMe # Number of rows to generate
generated_table = generate_new_rows(df, model, tokenizer, num_generated_rows)

generated_table.to_csv('synthetic_data_gpt.csv', index=False)
print(generated_table.head())

# Evaluate the model on the validation set
val_loss = model.evaluate(val_dataset)
print(f"Validation loss: {val_loss}")
