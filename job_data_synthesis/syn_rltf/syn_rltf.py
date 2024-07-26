import pandas as pd
from realtabformer import REaLTabFormer

df = pd.read_csv('/syn_eval_mle/syn_input_job.csv')

# Non-relational or parent table.
rtf_model = REaLTabFormer(
    model_type="tabular",
    gradient_accumulation_steps=4,
    logging_steps=1000)

# Fit the model on the dataset.
rtf_model.fit(df)

rtf_model.save("rtf_model/")

# Generate synthetic data
synthetic_data = rtf_model.sample(n_samples=1000)

synthetic_data.to_csv('out_rltf.csv', index=False)
print(synthetic_data.head())
