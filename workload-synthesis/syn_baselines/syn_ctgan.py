import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from ctgan import CTGAN
import matplotlib.pyplot as plt

# Load the Dataset
print("Loading the dataset...")
df = pd.read_csv('/home/yzg244/syn_mle/syn_input_job.csv')

# Apply Min-Max scaling to ensure all data is between 0 and 1
scaler = MinMaxScaler()
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=df.columns)

# Split the data into train, validation, and test sets
train_data, temp_data = train_test_split(scaled_df, test_size=0.3, random_state=0)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=0)

# Instantiate the CTGAN model
ctgan = CTGAN(
    epochs=100,
    batch_size=500,
    generator_dim=(256, 256, 256),
    discriminator_dim=(256, 256, 256),
    generator_lr=2e-4,
    discriminator_lr=2e-4,
    pac=10
)

# Fit the CTGAN model on the training dataset
ctgan.fit(train_data)

# Evaluate the model on the validation set by generating synthetic data and comparing with validation data
num_generated_rows = len(val_data)
synthetic_val_data = ctgan.sample(num_generated_rows)

# Generate synthetic data
num_generated_rows = 400
synthetic_data = ctgan.sample(num_generated_rows)

# Inverse transform the synthetic data back to the original scale
synthetic_data = scaler.inverse_transform(synthetic_data)
synthetic_data = pd.DataFrame(synthetic_data, columns=df.columns)

# Ensure the generated values are integers
synthetic_data = synthetic_data.round().astype(int)

# Ensure there are no negtive values
synthetic_data = synthetic_data.applymap(lambda x: max(x, -x))

# Save the synthetic data to a CSV file
synthetic_data.to_csv('out_ctgan.csv', index=False)
print(synthetic_data.head())
