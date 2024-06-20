import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, DataCollatorForLanguageModeling, create_optimizer
import tensorflow as tf

# Print available GPUs
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Step 1: Load the Dataset
print("Loading the dataset...")

df = pd.read_csv('/home/yzg244/job_data.csv')
print(df.head())

# Step 2: Preprocess
# Convert each row to a string
data_strings = df.apply(lambda row: ' '.join(row.astype(str)), axis=1).tolist()

# Combine all rows into a single string for training
training_data = '\n'.join(data_strings)

# Step 3: Sample the Data
sample_size = 100 # FixMe
if len(data_strings) > sample_size:
    data_strings = data_strings[:sample_size]

# Step 4: Fine-Tune GPT-2
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set padding token
tokenizer.pad_token = tokenizer.eos_token  # Using end-of-sequence token as padding token

def encode(text):
    return tokenizer(text, return_tensors='tf', padding='max_length', max_length=512, truncation=True)['input_ids']

class TabularDataset(tf.data.Dataset):
    def _generator(data):
        for line in data:
            tokenized_line = tokenizer(line, return_tensors='tf', padding='max_length', max_length=512, truncation=True)
            yield tokenized_line['input_ids'][0], tokenized_line['input_ids'][0]  # Ensure input and target are the same

    def __new__(cls, data):
        return tf.data.Dataset.from_generator(
            lambda: cls._generator(data),
            output_signature=(tf.TensorSpec(shape=(512,), dtype=tf.int32), tf.TensorSpec(shape=(512,), dtype=tf.int32))
        )

print("Creating dataset and data collator...")

# Split data into training and validation sets
train_data, val_data = train_test_split(data_strings, test_size=0.1, random_state=0)

train_dataset = TabularDataset(train_data).batch(2)
val_dataset = TabularDataset(val_data).batch(2)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
model = TFGPT2LMHeadModel.from_pretrained('gpt2')

# Step 5: Compile the model
optimizer, lr_schedule = create_optimizer(
    init_lr=5e-5,
    num_train_steps=len(train_data) * 3,  # num_epochs
    num_warmup_steps=0
)

# Define custom training step
@tf.function
def train_step(input_ids, labels):
    with tf.GradientTape() as tape:
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Step 6: Train the model
print("Starting training...")
num_epochs = 3

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    for step, (input_ids, labels) in enumerate(train_dataset):
        loss = train_step(input_ids, labels)
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss.numpy()}")

# Step 7: Generate New Data
def generate_text(prompt, model, tokenizer, max_length=None):
    inputs = tokenizer(prompt, return_tensors='tf')
    input_length = tf.shape(inputs['input_ids'])[1]
    if max_length is None or max_length < input_length:
        max_length = input_length + 50  # Ensuring max_length is larger than input length
    outputs = model.generate(inputs['input_ids'], max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

prompt = data_strings[0]
generated_text = generate_text(prompt, model, tokenizer)
print(generated_text)

# Step 8: Post-Process the Generated Data
generated_rows = generated_text.split('\n')

# Ensure the number of columns matches
def pad_or_trim(row, num_columns):
    columns = row.split()
    if len(columns) < num_columns:
        columns.extend([''] * (num_columns - len(columns)))  # Pad with empty strings
    elif len(columns) > num_columns:
        columns = columns[:num_columns]  # Trim to the required number of columns
    return columns

num_columns = len(df.columns)
generated_data = [pad_or_trim(row, num_columns) for row in generated_rows]

generated_df = pd.DataFrame(generated_data, columns=df.columns)
generated_df.to_csv('generated_timeseries_data.csv', index=False)
print(generated_df.head())
