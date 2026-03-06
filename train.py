import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

from model import TrafficVolume


train_scaled_df = pd.read_csv("data/train_scaled.csv")
test_scaled_df = pd.read_csv("data/test_scaled.csv")

train_scaled = train_scaled_df.to_numpy()
test_scaled = test_scaled_df.to_numpy()


def create_sequences(data, seq_length, y_col_idx):

    xs = []
    ys = []

    for i in range(len(data) - seq_length):

        x = data[i:(i + seq_length)]
        y = data[i + seq_length, y_col_idx]

        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)


X_train, y_train = create_sequences(train_scaled, 12, -1)
X_test, y_test = create_sequences(test_scaled, 12, -1)


dataset_train = TensorDataset(
    torch.tensor(X_train.astype(np.float32)),
    torch.tensor(y_train.astype(np.float32))
)

dataset_test = TensorDataset(
    torch.tensor(X_test.astype(np.float32)),
    torch.tensor(y_test.astype(np.float32))
)


dataloader_train = DataLoader(dataset_train, batch_size=64, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=64, shuffle=False)


traffic_model = TrafficVolume()

criterion = nn.MSELoss()

optimizer = optim.Adam(traffic_model.parameters(), lr=0.0001)


final_training_loss = 0


for epoch in range(2):

    for batch_x, batch_y in dataloader_train:

        optimizer.zero_grad()

        outputs = traffic_model(batch_x)

        loss = criterion(outputs.squeeze(), batch_y)

        loss.backward()

        optimizer.step()

    print(f"Epoch {epoch+1} Loss: {loss}")

    final_training_loss = loss


torch.save(traffic_model.state_dict(), "traffic_model.pth")
