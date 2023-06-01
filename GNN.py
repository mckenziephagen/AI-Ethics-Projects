import os
import numpy as np
import pandas as pd
import torch

from torchmetrics import R2Score,PearsonCorrCoef
from utilities import *
from model import *


if __name__ == "__main__":

    dataset  = DevDataset('pyg')
    dataset = dataset.shuffle()

    # Train/test split (80-20)
    train_share = int(len(dataset) * 0.8)

    train_dataset = dataset[:train_share]
    test_dataset = dataset[train_share:]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GraphNetwork(32, dataset.num_node_features, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    loss_fn = torch.nn.MSELoss()
    losses = []
    eval_metric = PearsonCorrCoef().to(device)

    for epoch in range(0, 30):

        model.train()
        loss = 0

        for batch in train_loader:
            batch = batch.to(device)

            # print(batch.x)
            optimizer.zero_grad()
            out = model(batch)

            loss = loss_fn(out, batch.y)

            loss.backward()
            optimizer.step()

        model.eval()

        with torch.no_grad():
            for batch in test_loader:

                pred_test = model(batch.to(device))
                pred_test = pred_test.flatten()

                correlation = eval_metric(pred_test, batch.y).detach().cpu().numpy()
                val_loss = torch.sqrt(loss_fn(pred_test, batch.y))

                print('Validation Loss: {}; Pearson correlation: {}; r^2: {}'.format(
                    val_loss.item(), np.round(correlation, 2), r2(pred_test, batch.y)
                ))

        losses.append(loss.item())

        print(f'Epoch: {epoch + 1:02d}, '
            f'Loss: {torch.sqrt(loss):.4f}, ')
            # f'Train: {100 * train_result:.2f}%, '
            # f'Test: {100 * test_result:.2f}%')