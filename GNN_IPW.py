import os
import numpy as np
import pandas as pd
import torch

from torchmetrics import PearsonCorrCoef
from torch_geometric.data import  DataLoader

from utilities import *
from model import *


if __name__ == "__main__":

    dataset  = DevDataset('pyg')
    dataset = dataset.shuffle()

    # Train/test split (80-20)
    train_share = int(len(dataset) * 0.8)

    train_dataset = dataset[:train_share]
    test_dataset = dataset[train_share:]
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

    ## fille in the group counts here e.g. [ number_of_lable_1, number_of_lable_0]

    weighting_matrix = 1/torch.tensor([366.0, 413.0]).to(device)

    model = GraphNetwork(
        64,
        dataset.num_node_features,
        1
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-2
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    loss_fn = torch.nn.MSELoss(reduction='none')
    losses = []
    eval_metric = PearsonCorrCoef().to(device)
    total_step = 10

    for epoch in range(0, 100):

        model.train()
        loss = 0

        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()
            out = model(batch).flatten()

            loss = torch.sqrt(loss_fn(out, batch.y))

            sample_reweight = torch.tile(weighting_matrix, (batch.y.size()[0], 1))

            sample_reweight = torch.gather(
                sample_reweight,
                1,
                batch.gender.type(torch.int64).view(-1,1)
            )

            loss = torch.mean(loss*sample_reweight)
            loss.backward()

            optimizer.step()

        model.eval()
        losses.append(loss.item())

        if (epoch+1) % total_step == 0:

            print(
                f'Epoch: {epoch + 1:02d}, '
                f'Training Loss: {(loss):.4f}, '
            )

            with torch.no_grad():
                for batch in test_loader:

                    pred_test = model(batch.to(device))
                    pred_test = pred_test.flatten()

                    correlation = eval_metric(pred_test, batch.y).detach().cpu().numpy()
                    test_loss = torch.sqrt(loss_fn(pred_test, batch.y))

                    sample_reweight = torch.tile(weighting_matrix, (batch.y.size()[0], 1))

                    sample_reweight = torch.gather(
                        sample_reweight,
                        1,
                        batch.gender.type(torch.int64).view(-1,1)
                    )

                    test_loss = torch.mean(test_loss*sample_reweight)

                    print('Testing Loss: {}; Pearson correlation: {}; r^2: {}'.format(
                        test_loss.item(),
                        np.round(correlation, 2),
                        r2(pred_test, batch.y)
                    ))

                    pred_test = pred_test.detach().cpu().numpy()
                    gender_test = batch.gender.detach().cpu().numpy()
                    y_true_test = batch.y.detach().cpu().numpy()

                    # For males (gender == 1)
                    plot_scatter_and_compute_metrics(y_true_test, pred_test, gender_test==1, "Male")

                    # For females (gender == 0)
                    plot_scatter_and_compute_metrics(y_true_test, pred_test, gender_test==0, "Female")

                    # Print regression measures
                    print(calculate_regression_measures(y_true_test, pred_test, gender_test))
