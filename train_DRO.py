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
from sklearn import preprocessing

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

    gender_train = torch.from_numpy(x_train['Gender'].values).float().cuda()
    gender_val = torch.from_numpy(x_val['Gender'].values).float().cuda()
    gender_test = torch.from_numpy(x_test['Gender'].values).float().cuda()
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
    loss_computer = LossComputer(criterion=criterion, is_robust=True, normalize_loss=True, group_counts=torch.tensor([478, 549]).float())

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