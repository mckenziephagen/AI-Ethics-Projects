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

    model = GraphNetwork(
        64,
        dataset.num_node_features,
        1
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    loss_fn = torch.nn.MSELoss()
    losses = []
    eval_metric = PearsonCorrCoef().to(device)
    total_step = 10

    for epoch in range(0, 100):

        model.train()
        loss = 0

        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()
            out = model(batch)

            loss = loss_fn(out, batch.y)

            loss.backward()
            optimizer.step()

        model.eval()
        losses.append(loss.item())

        if (epoch+1) % total_step == 0:

            print(
                f'Epoch: {epoch + 1:02d}, '
                f'Training Loss: {torch.sqrt(loss):.4f}, '
            )


            with torch.no_grad():
                for batch in test_loader:

                    pred_test = model(batch.to(device))
                    pred_test = pred_test.flatten()

                    correlation = eval_metric(pred_test, batch.y).detach().cpu().numpy()
                    val_loss = torch.sqrt(loss_fn(pred_test, batch.y.flatten()))

                print('Testing Loss: {}; Pearson correlation: {}; r^2: {}'.format(
                    val_loss.item(),
                    np.round(correlation, 2),
                    r2(pred_test, batch.y)
                ))
