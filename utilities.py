import pandas as pd
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import Dataset
from torchmetrics import R2Score,PearsonCorrCoef


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

class LossComputer:
    def __init__(
        self,
        criterion,
        is_robust,
        group_counts,
        dataset=None,
        alpha=0.2,
        gamma=0.1,
        adj=None,
        min_var_weight=0,
        step_size=0.01,
        normalize_loss=False,
        btl=False,
    ):
        self.criterion = criterion
        self.is_robust = is_robust
        self.gamma = gamma
        self.alpha = alpha
        self.min_var_weight = min_var_weight
        self.step_size = step_size
        self.normalize_loss = normalize_loss
        self.btl = btl

        self.n_groups = 2 #dataset.n_groups
        self.group_counts = group_counts.cuda() #dataset.group_counts().cuda()
        self.group_frac = self.group_counts/self.group_counts.sum()
        self.group_str = "Male" #dataset.group_str

        if adj is not None:
            self.adj = torch.from_numpy(adj).float().cuda()
        else:
            self.adj = torch.zeros(self.n_groups).float().cuda()

        if is_robust:
            assert alpha, 'alpha must be specified'

        # quantities maintained throughout training
        self.adv_probs = torch.ones(self.n_groups).cuda()/self.n_groups
        self.exp_avg_loss = torch.zeros(self.n_groups).cuda()
        self.exp_avg_initialized = torch.zeros(self.n_groups).byte().cuda()

        self.reset_stats()

    def loss(self, yhat, y, group_idx=None, is_training=False):
        # compute per-sample and per-group losses
        per_sample_losses = self.criterion(yhat, y).cuda()
        group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx)

        #group_acc, group_count = self.compute_group_avg((torch.argmax(yhat,1)==y).float(), group_idx)
        group_acc = 0
        # update historical losses
        self.update_exp_avg_loss(group_loss, group_count)

        # compute overall loss
        if self.is_robust and not self.btl:
            actual_loss, weights = self.compute_robust_loss(group_loss, group_count)
        elif self.is_robust and self.btl:
             actual_loss, weights = self.compute_robust_loss_btl(group_loss, group_count)
        else:
            actual_loss = per_sample_losses.mean()
            weights = None

        # update stats
        self.update_stats(actual_loss, group_loss, group_acc, group_count, weights)

        return actual_loss

    def compute_robust_loss(self, group_loss, group_count):

        adjusted_loss = group_loss

        if torch.all(self.adj>0):
            adjusted_loss += self.adj/torch.sqrt(self.group_counts)
        if self.normalize_loss:
            adjusted_loss = adjusted_loss/(adjusted_loss.sum())

        self.adv_probs = self.adv_probs * torch.exp(self.step_size*adjusted_loss.data)
        self.adv_probs = self.adv_probs/(self.adv_probs.sum())

        robust_loss = group_loss @ self.adv_probs
        return robust_loss, self.adv_probs

    def compute_robust_loss_btl(self, group_loss, group_count):
        adjusted_loss = self.exp_avg_loss + self.adj/torch.sqrt(self.group_counts)
        return self.compute_robust_loss_greedy(group_loss, adjusted_loss)

    def compute_robust_loss_greedy(self, group_loss, ref_loss):
        sorted_idx = ref_loss.sort(descending=True)[1]
        sorted_loss = group_loss[sorted_idx]
        sorted_frac = self.group_frac[sorted_idx]

        mask = torch.cumsum(sorted_frac, dim=0)<=self.alpha
        weights = mask.float() * sorted_frac /self.alpha
        last_idx = mask.sum()
        weights[last_idx] = 1 - weights.sum()
        weights = sorted_frac*self.min_var_weight + weights*(1-self.min_var_weight)

        robust_loss = sorted_loss @ weights

        # sort the weights back
        _, unsort_idx = sorted_idx.sort()
        unsorted_weights = weights[unsort_idx]
        return robust_loss, unsorted_weights

    def compute_group_avg(self, losses, group_idx):
        # compute observed counts and mean loss for each group

        group_map = (group_idx == torch.arange(self.n_groups).unsqueeze(1).long().cuda()).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count==0).float() # avoid nans

        group_loss = (group_map @ losses.view(-1))/group_denom

        return group_loss, group_count

    def update_exp_avg_loss(self, group_loss, group_count):

        prev_weights = (1 - self.gamma*(group_count>0).float()) * (self.exp_avg_initialized>0).float()
        curr_weights = 1 - prev_weights
        self.exp_avg_loss = self.exp_avg_loss * prev_weights + group_loss*curr_weights
        self.exp_avg_initialized = (self.exp_avg_initialized>0) + (group_count>0)

    def reset_stats(self):
        self.processed_data_counts = torch.zeros(self.n_groups).cuda()
        self.update_data_counts = torch.zeros(self.n_groups).cuda()
        self.update_batch_counts = torch.zeros(self.n_groups).cuda()
        self.avg_group_loss = torch.zeros(self.n_groups).cuda()
        self.avg_group_acc = torch.zeros(self.n_groups).cuda()
        self.avg_per_sample_loss = 0.
        self.avg_actual_loss = 0.
        self.avg_acc = 0.
        self.batch_count = 0.

    def update_stats(self, actual_loss, group_loss, group_acc, group_count, weights=None):
        # avg group loss
        denom = self.processed_data_counts + group_count
        denom += (denom==0).float()
        prev_weight = self.processed_data_counts/denom
        curr_weight = group_count/denom
        self.avg_group_loss = prev_weight*self.avg_group_loss + curr_weight*group_loss

        # avg group acc
        self.avg_group_acc = prev_weight*self.avg_group_acc + curr_weight*group_acc

        # batch-wise average actual loss
        denom = self.batch_count + 1
        self.avg_actual_loss = (self.batch_count/denom)*self.avg_actual_loss + (1/denom)*actual_loss

        # counts
        self.processed_data_counts += group_count
        if self.is_robust:
            self.update_data_counts += group_count*((weights>0).float())
            self.update_batch_counts += ((group_count*weights)>0).float()
        else:
            self.update_data_counts += group_count
            self.update_batch_counts += (group_count>0).float()
        self.batch_count+=1

        # avg per-sample quantities
        group_frac = self.processed_data_counts/(self.processed_data_counts.sum())
        self.avg_per_sample_loss = group_frac @ self.avg_group_loss
        self.avg_acc = group_frac @ self.avg_group_acc

    def get_model_stats(self, model, args, stats_dict):
        model_norm_sq = 0.
        for param in model.parameters():
            model_norm_sq += torch.norm(param) ** 2
        stats_dict['model_norm_sq'] = model_norm_sq.item()
        stats_dict['reg_loss'] = args.weight_decay / 2 * model_norm_sq.item()
        return stats_dict

    def get_stats(self, model=None, args=None):
        stats_dict = {}
        for idx in range(self.n_groups):
            stats_dict[f'avg_loss_group:{idx}'] = self.avg_group_loss[idx].item()
            stats_dict[f'exp_avg_loss_group:{idx}'] = self.exp_avg_loss[idx].item()
            stats_dict[f'avg_acc_group:{idx}'] = self.avg_group_acc[idx].item()
            stats_dict[f'processed_data_count_group:{idx}'] = self.processed_data_counts[idx].item()
            stats_dict[f'update_data_count_group:{idx}'] = self.update_data_counts[idx].item()
            stats_dict[f'update_batch_count_group:{idx}'] = self.update_batch_counts[idx].item()

        stats_dict['avg_actual_loss'] = self.avg_actual_loss.item()
        stats_dict['avg_per_sample_loss'] = self.avg_per_sample_loss.item()
        stats_dict['avg_acc'] = self.avg_acc.item()

        # Model stats
        if model is not None:
            assert args is not None
            stats_dict = self.get_model_stats(model, args, stats_dict)

        return stats_dict

    def log_stats(self, logger, is_training):
        if logger is None:
            return

        logger.write(f'Average incurred loss: {self.avg_per_sample_loss.item():.3f}  \n')
        logger.write(f'Average sample loss: {self.avg_actual_loss.item():.3f}  \n')
        logger.write(f'Average acc: {self.avg_acc.item():.3f}  \n')
        for group_idx in range(self.n_groups):
            logger.write(
                f'  {self.group_str(group_idx)}  '
                f'[n = {int(self.processed_data_counts[group_idx])}]:\t'
                f'loss = {self.avg_group_loss[group_idx]:.3f}  '
                f'exp loss = {self.exp_avg_loss[group_idx]:.3f}  '
                f'adjusted loss = {self.exp_avg_loss[group_idx] + self.adj[group_idx]/torch.sqrt(self.group_counts)[group_idx]:.3f}  '
                f'adv prob = {self.adv_probs[group_idx]:3f}   '
                f'acc = {self.avg_group_acc[group_idx]:.3f}\n')
        logger.flush()


def calculate_mse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))

def plot_scatter_and_compute_metrics(actual, predicted, gender, gender_str):
    eval_metric = PearsonCorrCoef()
    gender_specific_actual = actual[gender]
    gender_specific_predicted = predicted[gender]
    
    # Convert tensors to numpy arrays for calculation and plotting
    gender_specific_actual_np = gender_specific_actual.numpy()
    gender_specific_predicted_np = gender_specific_predicted.detach().numpy()
    
    print(f"MSE ({gender_str}): {calculate_mse(gender_specific_actual_np, gender_specific_predicted_np)}")
    print(f"Pearson correlation ({gender_str}): {eval_metric(gender_specific_predicted, gender_specific_actual)}")
    
    # plt.scatter(gender_specific_actual_np, gender_specific_predicted_np, label=gender_str)


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