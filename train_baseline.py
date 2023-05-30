import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from torchmetrics import R2Score,PearsonCorrCoef

import math
import matplotlib.pyplot as plt

from model import *
from utilities import *

if __name__ == "__main__":

    pt_id =  pd.read_csv('data/subjectIDs.txt', sep=" ", header=None)
    pt_id.rename(columns={ pt_id.columns[0]: "Subject" }, inplace = True)

    data = pd.read_csv("data/ICA25/netmats1.txt", sep=" ", header=None)
    data["Subject"] = pt_id["Subject"]

    label = pd.read_csv("data/cognition_scores.csv")
    label.rename(columns={ label.columns[0]: "Subject" }, inplace = True)

    demographics = pd.read_csv("data/demographic.csv")
    data = data.merge(demographics[["Gender","Subject"]], how='left', on='Subject')
    data = data.merge(label[["Subject", "CogTotalComp_Unadj"]], how='left', on='Subject')

    data.dropna(subset=['CogTotalComp_Unadj'], inplace=True)

    # Convert Gender to 0 and 1
    X = data.copy()
    X["Gender"] = np.where(X["Gender"] == "M", 1.0, 0.0)
    X.pop("Subject")

    # Split the data into train and test set

    # X = X.fillna(lambda x: x.mean())

    x_train, x_test, y_train, y_test = train_test_split(
        X.iloc[:, :-1], 
        X["CogTotalComp_Unadj"].values, 
        test_size=0.2, 
        random_state=20
    )

    # Further split the training data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    # Save the Gender feature for later use before converting the rest to PyTorch tensors
    # gender_train = torch.from_numpy(x_train.pop('Gender').values).float()
    # gender_val = torch.from_numpy(x_val.pop('Gender').values).float()
    # gender_test = torch.from_numpy(x_test.pop('Gender').values).float()


    gender_train = torch.from_numpy(x_train['Gender'].values).float()
    gender_val = torch.from_numpy(x_val['Gender'].values).float()
    gender_test = torch.from_numpy(x_test['Gender'].values).float()
    # Convert pandas dataframe to PyTorch tensor

    normalizer = preprocessing.Normalizer()

    x_train = normalizer.fit_transform(x_train.values)
    x_val  = normalizer.transform(x_val.values)
    x_test  = normalizer.transform(x_test.values)

    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()

    x_val = torch.from_numpy(x_val).float()
    y_val = torch.from_numpy(y_val).float()

    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).float()

    # init the model

    model = ConnectomeNet(x_train.size()[1], 200)

    lr = 1e-3
    weight_decay = 1e-6
    num_epochs = 60
    batch_size = 32
    patience = 10  # Number of epochs to wait for improvement before stopping

    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    total_step = len(dataloader)

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    criterion = nn.MSELoss()
    eval_metric = PearsonCorrCoef()

    # For early stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(dataloader):

            pred_y = model(x).flatten()
            loss = torch.sqrt(criterion(pred_y, y))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % total_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Training Loss: {:.4f}'.format(
                    epoch+1, num_epochs,
                    i+1, total_step,
                    loss.item()
                ))

                with torch.no_grad():
                    correlation = eval_metric(model(x_val).flatten(), y_val)
                    pred_val = model(x_val).flatten()
                    val_loss = torch.sqrt(criterion(pred_val, y_val))

                    print('Validation Loss: {}; Pearson correlation: {}; r^2: {}'.format(
                        val_loss.item(), np.round(correlation, 2), r2(pred_val, y_val)
                    ))

                    # Check for early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve == patience:
                            print('Early stopping!')
                            break  # Break the inner loop, the next line will break the outer loop

        # Break the outer loop
        if epochs_no_improve == patience:
            break

    torch.save(model.state_dict(), "models/model.pth")
    
    predictions_test = model(x_test).flatten()
    loss_test = torch.sqrt(criterion(predictions_test, y_test))
    correlation_test = eval_metric(predictions_test, y_test).detach().numpy()

    # print(f"R2: {r2(predictions_test, y_test)}")
    print(f'Test Loss: {loss_test.item():.2f}; Pearson correlation: {np.round(correlation_test, 2)}')

    # Plotting and computing metrics
    plt.figure(figsize=(10, 8))  # Increase the size of your plot

    # For males (gender == 1)
    plot_scatter_and_compute_metrics(y_test, predictions_test, gender_test==1, "Male")

    # For females (gender == 0)
    plot_scatter_and_compute_metrics(y_test, predictions_test, gender_test==0, "Female")

    # Print regression measures
    print(calculate_regression_measures(y_test, predictions_test.detach().numpy(), gender_test))