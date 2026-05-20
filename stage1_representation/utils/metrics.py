"""
评估指标模块

包含预测评估的所有指标：
- 基础指标：MAE, MSE, RSE, CORR
- 排序指标：RankIC, MIRR
- 损失函数：ranking_loss
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List, Union


# ==================== 基础指标函数 ====================

def RSE(pred: Union[np.ndarray, Tensor], true: Union[np.ndarray, Tensor]) -> float:
    """相对平方误差 (Relative Squared Error)"""
    if isinstance(pred, Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(true, Tensor):
        true = true.detach().cpu().numpy()
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred: Union[np.ndarray, Tensor], true: Union[np.ndarray, Tensor]) -> float:
    """相关系数 (Correlation Coefficient)"""
    if isinstance(pred, Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(true, Tensor):
        true = true.detach().cpu().numpy()
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred: Union[np.ndarray, Tensor], true: Union[np.ndarray, Tensor]) -> float:
    """平均绝对误差 (Mean Absolute Error)"""
    if isinstance(pred, Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(true, Tensor):
        true = true.detach().cpu().numpy()
    return np.mean(np.abs(pred - true))


def MSE(pred: Union[np.ndarray, Tensor], true: Union[np.ndarray, Tensor]) -> float:
    """均方误差 (Mean Squared Error)"""
    if isinstance(pred, Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(true, Tensor):
        true = true.detach().cpu().numpy()
    return np.mean((pred - true) ** 2)


def metric(pred, true):
    """计算基础指标"""
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    return mae, mse


# ==================== 排序指标 ====================

def ranking_loss(pred: Tensor, gt: Tensor, reduction: str = 'mean', weight: Tensor = None) -> Tensor:
    """排序损失函数"""
    loss_mat = ranking_loss_matrix(pred, gt)
    loss_mat = F.relu(-loss_mat) # 只有排序错误的部分才有损失，正确的部分为0
    loss_mat = torch.mean(loss_mat, dim=2)

    if weight is not None:
        loss_mat = loss_mat * weight

    return _loss_reduce(loss_mat, reduction)


def ranking_loss_matrix(pred: Tensor, gt: Tensor) -> Tensor:
    """计算排序损失矩阵"""
    num_nodes = pred.shape[1]
    pred_matrix = pred.reshape(-1, num_nodes, 1)
    gt_matrix = gt.reshape(-1, num_nodes, 1)
    one_vector = torch.ones(num_nodes, 1, device=pred.device, dtype=pred.dtype)
    one_row_vector = one_vector.transpose(0, 1)

    pred_diff_matrix = torch.matmul(pred_matrix, one_row_vector) \
                       - torch.matmul(one_vector, pred_matrix.transpose(1, 2)).detach()

    gt_diff_matrix = torch.matmul(gt_matrix, one_row_vector) \
                     - torch.matmul(one_vector, gt_matrix.transpose(1, 2))
    loss_mat = torch.mul(pred_diff_matrix, gt_diff_matrix)
    return loss_mat # 排序正确乘积为正，错误为负


def mirr_top_k(prediction: Tensor, gt: Tensor, k: int) -> float:
    """Incremental Return Rate - Top K"""
    pred_topk_index = torch.topk(prediction, k, dim=1)[1]
    mirr = torch.mean(gt[torch.arange(prediction.shape[0]).unsqueeze(1), pred_topk_index]).item()
    return mirr


def mirr_top1(prediction: Tensor, gt: Tensor) -> float:
    """Mean Incremental Return Rate - Top 1"""
    pred_top1_index = torch.argmax(prediction, dim=1)
    mirr = torch.mean(gt[torch.arange(prediction.shape[0]), pred_top1_index]).item()
    return mirr


def rank_ic(prediction: Tensor, gt: Tensor) -> List[float]:
    """Rank Information Coefficient (排名相关系数) """
    rank_gt = torch.argsort(gt, dim=1, descending=True).float()
    rank_pred = torch.argsort(prediction, dim=1, descending=True).float()
    return _compute_rank_ic(rank_pred, rank_gt)


def correlation(a: Tensor, b: Tensor) -> float:
    """计算两个张量的相关系数"""
    return covariance(a, b) / (a.std(unbiased=False).item() * b.std(unbiased=False).item())


def covariance(a: Tensor, b: Tensor) -> float:
    """计算两个张量的协方差"""
    ab = torch.mul(a, b)
    return ab.mean().item() - a.mean().item() * b.mean().item()


def _compute_rank_ic(pred: Tensor, gt: Tensor) -> List[float]:
    """计算Rank IC列表"""
    rank_ic_list = []
    for i in range(pred.shape[0]):
        rank_ic = correlation(pred[i], gt[i])
        rank_ic_list.append(rank_ic)
    return rank_ic_list


def _loss_reduce(loss: Tensor, reduction: str) -> Tensor:
    """损失函数的reduction操作"""
    if reduction == 'mean':
        return torch.mean(loss)
    elif reduction == 'sum':
        return torch.sum(loss)
    elif reduction == 'none':
        return loss
    else:
        raise NotImplementedError(reduction)


# ==================== 指标类（用于训练过程跟踪） ====================

class Metric:
    """所有指标的基类"""

    def __init__(self, stage='test'):
        self.stage = stage

    def update(self, pred, true):
        """更新指标"""
        raise NotImplementedError

    def reset(self):
        """重置指标"""
        raise NotImplementedError

    @property
    def name(self):
        """指标名称"""
        return f'{self.stage}/{self.__class__.__name__}'


class MAEMetric(Metric):
    """平均绝对误差 (用于训练跟踪)"""

    def __init__(self, stage='test'):
        super().__init__(stage)
        self.values = []

    def update(self, pred, true):
        mae = torch.mean(torch.abs(pred - true)).item()
        self.values.append(mae)

    def reset(self):
        self.values = []

    @property
    def value(self):
        return np.array(self.values)


class MSEMetric(Metric):
    """均方误差 (用于训练跟踪)"""

    def __init__(self, stage='test'):
        super().__init__(stage)
        self.values = []

    def update(self, pred, true):
        mse = torch.mean((pred - true) ** 2).item()
        self.values.append(mse)

    def reset(self):
        self.values = []

    @property
    def value(self):
        return np.array(self.values)


class MIRRTop1Metric(Metric):
    """MIRR Top 1 (用于训练跟踪)"""

    def __init__(self, stage='test'):
        super().__init__(stage)
        self.values = []

    def update(self, pred, true):
        mirr = mirr_top1(pred, true)
        self.values.append(mirr)

    def reset(self):
        self.values = []

    @property
    def value(self):
        return np.array(self.values)


class RankICMetric(Metric):
    """Rank IC (用于训练跟踪)"""

    def __init__(self, stage='test'):
        super().__init__(stage)
        self.values = []

    def update(self, pred, true):
        ic_list = rank_ic(pred, true)
        self.values.extend(ic_list)

    def reset(self):
        self.values = []

    @property
    def value(self):
        return np.array(self.values)
