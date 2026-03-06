import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from torch.utils.data import TensorDataset, DataLoader

from model import TrafficVolume


def create_sequences(data, seq_length, y_col_idx):

    xs = []
    ys = []

    for i in range(len(data) - seq_length):

        x = data[i:(i + seq_length)]
        y = data[i + seq_length, y_col_idx]

        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)


test_scaled_df = pd.read_csv("data/test_scaled.csv")

test_scaled = test_scaled_df.to_numpy()

X_test, y_test = create_sequences(test_scaled, 12, -1)


dataset_test = TensorDataset(
    torch.tensor(X_test.astype(np.float32)),
    torch.tensor(y_test.astype(np.float32))
)

dataloader_test = DataLoader(dataset_test, batch_size=64)


model = TrafficVolume()

model.load_state_dict(torch.load("traffic_model.pth"))

model.eval()


predictions = []
labels = []


with torch.no_grad():

    for seqs, y in dataloader_test:

        outputs = model(seqs).squeeze()

        predictions.append(outputs)
        labels.append(y)


predictions = torch.cat(predictions)
labels = torch.cat(labels)


test_mse = F.mse_loss(predictions, labels)

print("Test MSE:", test_mse.item())
