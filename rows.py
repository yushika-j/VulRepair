import pandas as pd

# This script shows the number of rows in the original data files vs the cleaned (deduped) data files.

df2_train = pd.read_csv('data/fine_tune_data/train.csv')
print("fine_tune_data/train: Train rows:", len(df2_train))

df2_test = pd.read_csv('data/fine_tune_data/test.csv')
print("fine_tune_data/test: Test rows:", len(df2_test))

df2_val = pd.read_csv('data/fine_tune_data/val.csv')
print("fine_tune_data/val: Val rows:", len(df2_val))

print("------------------")

df_train = pd.read_csv('data/cleaned_train.csv')
print("cleaned_train: Train rows:", len(df_train))

df_test = pd.read_csv('data/cleaned_test.csv')
print("cleaned_test: Test rows:", len(df_test))

df_val = pd.read_csv('data/fine_tune_data/val.csv')
print("cleaned_val: Val rows:", len(df_val))
