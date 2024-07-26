import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ctgan import CTGAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('/syn_eval_mle/syn_input_job.csv')

scaler = MinMaxScaler()
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=df.columns)

train_data, temp_data = train_test_split(scaled_df, test_size=0.3, random_state=0)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=0)

ctgan = CTGAN(
    epochs=100,
    batch_size=500,
    generator_dim=(256, 256, 256),
    discriminator_dim=(256, 256, 256),
    generator_lr=2e-4,
    discriminator_lr=2e-4,
    pac=10
)

ctgan.fit(train_data)

num_generated_rows = len(val_data)
synthetic_val_data = ctgan.sample(num_generated_rows)

num_generated_rows = 1000
synthetic_data = ctgan.sample(num_generated_rows)

synthetic_data = scaler.inverse_transform(synthetic_data)
synthetic_data = pd.DataFrame(synthetic_data, columns=df.columns)

synthetic_data = synthetic_data.round().astype(int)
synthetic_data = synthetic_data.applymap(lambda x: max(x, -x))

synthetic_data.to_csv('out_ctgan.csv', index=False)
print(synthetic_data.head())
