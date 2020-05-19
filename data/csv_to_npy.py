import pandas as pd
import numpy as np

filename = 'eurusd_hour.csv'
# filename = 'eurusd_minute.csv'


df = pd.read_csv(filename)
print(df.head(5))
print(df.shape)

df = (df['BidClose'] + df['AskClose']) / 2
print(df)

train_size = int(0.6 * df.shape[0])
val_size = int(0.2 * df.shape[0])
test_size = int(0.2 * df.shape[0])

train_df = df[0:train_size].to_numpy()
val_df = df[train_size:train_size + val_size].to_numpy()
test_df = df[train_size + val_size:].to_numpy()

print("train shape: ", train_df.shape)
print("val shape: ", val_df.shape)
print("test shape: ", test_df.shape)

np.save('train_data.npy', train_df)
np.save('test_data.npy', val_df)
np.save('val_data.npy', test_df)
