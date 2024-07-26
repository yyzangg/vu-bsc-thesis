import pandas as pd
from be_great import GReaT

data = pd.read_csv('/syn_eval_mle/syn_input_job.csv')

model = GReaT(llm='distilgpt2', batch_size=32,  epochs=20, fp16=True)
model.fit(data)
synthetic_data = model.sample(n_samples=1000)

synthetic_data.to_csv('out_great.csv', index=False)
print(synthetic_data.head())
