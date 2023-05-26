import pandas as pd
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import Dataset

class BOLDDataset(Dataset):
    def __init__(self, features, targets, genders):
        self.features = features
        self.targets = targets
        self.genders = genders

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.targets[idx]
        gender = self.genders[idx]

        return x, y, gender

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * 0.1

def grad_reverse(x):
    return GradReverse.apply(x)

def r2(output, target):
    target_mean = torch.mean(target)

    ss_res = torch.sum((target - output) ** 2)
    ss_tot = torch.sum((target - target_mean) ** 2)

    r2 = 1 - ss_res/ss_tot
    return r2

def calculate_regression_measures(y, y_hat, gender_info):

    data = pd.DataFrame(
        columns=[
            'subgroup',
            'independence',
            'separation',
            'sufficiency'
         ]
        )

    y_u = ((y - y.mean()) / y.std()).reshape(-1, 1)
    s_u = ((y_hat - y_hat.mean()) / y_hat.std()).reshape(-1, 1)

    # a = np.where(gender_info == 1, 1, 0)

    a = gender_info
    p_s = LogisticRegression()
    p_ys = LogisticRegression()
    p_y = LogisticRegression()

    p_s.fit(s_u, a)
    p_y.fit(y_u, a)
    p_ys.fit(np.c_[y_u, s_u], a)

    pred_p_s = p_s.predict_proba(s_u.reshape(-1, 1))[:, 1]
    pred_p_y = p_y.predict_proba(y_u.reshape(-1, 1))[:, 1]
    pred_p_ys = p_ys.predict_proba(np.c_[y_u, s_u])[:, 1]

    n = len(a)

    r_ind = ((n - a.sum()) / a.sum()) * (pred_p_s / (1 - pred_p_s)).mean()
    r_sep = ((pred_p_ys / (1 - pred_p_ys) * (1 - pred_p_y) / pred_p_y)).mean()
    r_suf = ((pred_p_ys / (1 - pred_p_ys)) * ((1 - pred_p_s) / pred_p_s)).mean()

    to_append = pd.DataFrame({'subgroup': "Male",
                              'independence': [r_ind],
                              'separation': [r_sep],
                              'sufficiency': [r_suf]})

    data = pd.concat([data, to_append])

    return data