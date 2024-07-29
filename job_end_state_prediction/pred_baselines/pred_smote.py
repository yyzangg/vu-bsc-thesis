import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

df = pd.read_csv('/pred_baselines/pred_input_job.csv')

X = df[['ReqCPUS', 'submit_hour_of_day', 'submit_day_of_week', 'submit_day_of_month', 'running_time']]
y = df['state_encoded']

# Define the resampling strategy
resampler = SMOTETomek(sampling_strategy='auto')

X_resampled, y_resampled = resampler.fit_resample(X, y)

resampled_df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=['state_encoded'])], axis=1)

resampled_df.to_csv('/pred_baselines/resampled_pred_input_job.csv', index=False)
