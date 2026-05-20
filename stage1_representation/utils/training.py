"""
训练工具模块
"""
import numpy as np
import torch
import uuid

# ==================== ID生成 ====================
def generate_id() -> str:
    """
    生成唯一实验ID
    """
    return str(uuid.uuid4())[:8]


# ==================== 学习率调整 ====================
def adjust_learning_rate(optimizer, epoch, args):
    """调整学习率"""
    lr_adjust = {}

    if args.lradj == 'type1':
        # 指数衰减：每adjust_interval个epoch衰减一半
        if epoch % args.adjust_interval == 0:
            lr_adjust = {epoch: args.learning_rate * (0.5 ** ((int(epoch / args.adjust_interval) - 1) // 1))}
    elif args.lradj == 'type2':
        # 预定义的学习率表
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'Updating learning rate to {lr}')


# ==================== 早停机制 ====================
class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        """检查是否应该早停"""
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0 # 重置计数器

    def save_checkpoint(self, val_loss, model, path):
        """保存模型检查点"""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss

    def reset(self):
        """重置早停计数器"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
