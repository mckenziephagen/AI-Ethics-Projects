import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from torchmetrics import R2Score,PearsonCorrCoef

from model import *
from utilities import *

if __name__ == "__main__":

    df = pd.read_pickle("data/connectome.pkl")

    label = pd.read_csv("data/cognition_scores.csv")
    label.rename(columns={ label.columns[0]: "Subject" }, inplace = True)

    demographics = pd.read_csv("data/demographic.csv")

    df = df.merge(demographics[["Gender","Subject"]], how='left', on='Subject')

    # Convert Gender to 0 and 1
    X = df.copy()
    X["Gender"] = np.where(X["Gender"] == "M", 1, 0)

    # Split the data into train and test set
    x_train, x_test, y_train, y_test = train_test_split(X, label.iloc[:, 1], test_size=0.2, random_state=42)

    # Further split the training data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    # Save the Gender feature for later use before converting the rest to PyTorch tensors
    gender_train = torch.from_numpy(x_train.pop('Gender').values).float().cuda()
    gender_val = torch.from_numpy(x_val.pop('Gender').values).float().cuda()
    gender_test = torch.from_numpy(x_test.pop('Gender').values).float().cuda()

    # Convert pandas dataframe to PyTorch tensor
    x_train = torch.from_numpy(x_train.values).float()
    y_train = torch.from_numpy(y_train.values).float()

    x_val = torch.from_numpy(x_val.values).float()
    y_val = torch.from_numpy(y_val.values).float()

    x_test = torch.from_numpy(x_test.values).float()
    y_test = torch.from_numpy(y_test.values).float()

    # init the model

    model = ConnectomeNet(x_train.size()[1], 5000)

    lr = 1e-3
    weight_decay = 1e-5
    num_epochs = 100
    batch_size = 256

    train_dataset = BOLDDataset(x_train,y_train, gender_train)
    val_dataset = BOLDDataset(x_val,y_val, gender_val)
    test_dataset = BOLDDataset(x_test,y_test, gender_test)

    dataloader = DataLoader(train_dataset, batch_size = batch_size)

    total_step = len(dataloader)

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    # Reduction needs to be set to none.

    criterion = nn.MSELoss(reduction='none')

    eval_metric = PearsonCorrCoef()
    loss_computer = LossComputer(criterion=criterion, is_robust=True, normalize_loss=True)

    for epoch in range(num_epochs):

        for i, (x, y, gender) in enumerate(dataloader):

            pred_y = model(x).flatten()

            loss = torch.sqrt(loss_computer.loss(pred_y, y, gender))

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
                    val_loss = torch.sqrt(torch.mean(criterion(pred_val, y_val)))

                    print('Validation Loss: {}; Pearson correlation: {}; r^2: {}'.format(
                        val_loss.item(),
                        np.round(correlation, 2),
                        r2(pred_val, y_val)
                    ))

    torch.save(model.state_dict(), "model_DRO.pth")
    pred_test = model(x_test)
    pred_test = pred_test.flatten()
    test_loss = torch.mean(criterion(pred_test, y_test))

    gender_test = gender_test.detach().cpu().numpy()

    correlation = eval_metric(pred_test, y_test).detach().cpu().numpy()

    print('Test Loss: {}; Pearson correlation: {}; r2: {}'.format(
        test_loss.item()/len(pred_test),
        np.round(correlation, 2),
        r2(pred_test, y_test)
        ))
    print("MSE: male;", torch.mean(criterion(y_test[gender_test==1],pred_test[gender_test==1] )))
    print("MSE: female;", torch.mean(criterion(y_test[gender_test==0],pred_test[gender_test==0] )))

    print(
          eval_metric(pred_test[gender_test==1], y_test[gender_test==1]),
          eval_metric(pred_test[gender_test==0], y_test[gender_test==0])
        )
    print(calculate_regression_measures(y_test, pred_test.detach().cpu().numpy(), gender_test))