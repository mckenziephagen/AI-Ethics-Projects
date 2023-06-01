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

    predictor = Predictor(64).to(device)

    optimizer_net = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    optimizer_predictor = torch.optim.Adam(
        predictor.parameters(),
        lr=1e-3,
        weight_decay = 1e-5
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    loss_fn = torch.nn.MSELoss()
    losses = []
    eval_metric = PearsonCorrCoef().to(device)
    total_step = 10
    eps = 1e-7

    for epoch in range(0, 100):

        model.train()
        loss = 0

        for batch in train_loader:
            batch = batch.to(device)

            feat_out, pred_y = model.forward_lnl(batch)
            feat_out = feat_out.to(device)

            _, pred_bias_prob = predictor(feat_out)

            loss_pred = torch.sqrt(loss_fn(pred_y.flatten(), batch.y))
            loss_bias = torch.mean(torch.sum(pred_bias_prob*torch.log(pred_bias_prob + eps)))

            loss = loss_pred + loss_bias
            optimizer_net.zero_grad()
            optimizer_predictor.zero_grad()


            loss = loss_pred + loss_bias
            optimizer_net.zero_grad()
            optimizer_predictor.zero_grad()

            loss.backward()

            optimizer_net.step()

            optimizer_net.zero_grad()
            optimizer_predictor.zero_grad()
            
            feat_label = grad_reverse(feat_out)
            pred_label, _ = predictor(feat_label)

            # import ipdb; ipdb.set_trace()
            
            bceloss = nn.BCELoss()
            loss_beta = bceloss(pred_label, batch.gender.flatten()).requires_grad_()
            loss_beta.backward()

            optimizer_net.step()
            optimizer_predictor.step()

            optimizer_net.zero_grad()
            optimizer_predictor.zero_grad()


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
                    test_loss = torch.sqrt(loss_fn(pred_test, batch.y))

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
