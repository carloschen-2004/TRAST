from data.stock_data_handle import DatasetStock_PRED
from exp.exp_basic import Exp_Basic

# 导入所有模型架构
from models.transformer import (
    Transformer_base,
    Transformer_MHA_RoPE_MoE,
    Transformer_MQA_RoPE_MoE,
    Transformer_GQA_RoPE_MoE
)

from utils.training import adjust_learning_rate
from utils.metrics import ranking_loss, MIRRTop1Metric, RankICMetric
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import os
import time
import matplotlib.pyplot as plt

dataset_dict = {
    'stock': DatasetStock_PRED,
}

class Exp_pred(Exp_Basic):
    def __init__(self, args, data_all, id):
        super(Exp_pred, self).__init__(args)
        self.data_all = data_all
        self.id = id

    def _build_model(self):
        arch_type = getattr(self.args, 'arch_type', 'base')

        # 根据架构类型选择模型
        if arch_type == 'base':
            model = Transformer_base(
                self.args.enc_in,
                self.args.dec_in,
                self.args.c_out,
                self.args.d_model,
                self.args.n_heads,
                self.args.e_layers,
                self.args.d_layers,
                self.args.d_ff,
                self.args.dropout,
                self.args.activation
            )
        elif arch_type == 'MHA_RoPE_MoE':
            model = Transformer_MHA_RoPE_MoE(
                self.args.enc_in,
                self.args.dec_in,
                self.args.c_out,
                self.args.d_model,
                self.args.n_heads,
                self.args.e_layers,
                self.args.d_layers,
                self.args.d_ff,
                self.args.num_experts,
                self.args.top_k,
                self.args.dropout,
                self.args.activation
            )
        elif arch_type == 'MQA_RoPE_MoE':
            model = Transformer_MQA_RoPE_MoE(
                self.args.enc_in,
                self.args.dec_in,
                self.args.c_out,
                self.args.d_model,
                self.args.n_heads,
                self.args.e_layers,
                self.args.d_layers,
                self.args.d_ff,
                self.args.num_experts,
                self.args.top_k,
                self.args.dropout,
                self.args.activation
            )
        elif arch_type == 'GQA_RoPE_MoE':
            model = Transformer_GQA_RoPE_MoE(
                self.args.enc_in,
                self.args.dec_in,
                self.args.c_out,
                self.args.d_model,
                self.args.n_heads,
                self.args.n_kv_heads,
                self.args.e_layers,
                self.args.d_layers,
                self.args.d_ff,
                self.args.num_experts,
                self.args.top_k,
                self.args.dropout,
                self.args.activation
            )
        else:
            raise ValueError(f"Unknown architecture type: {arch_type}")

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model.float()

    def _get_data(self, flag):
        args = self.args

        if flag == 'train':
            shuffle_flag = True; drop_last = False; batch_size = args.batch_size
        else:
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size
      
        dataset = dataset_dict[self.args.data_type](self.data_all, type=flag, pred_type=self.args.pred_type)
        
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return dataset, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, metric_builders, stage='test'):
        self.model.eval()
        total_loss = []
        metric_objs = [builder(stage) for builder in metric_builders]

        for i, (batch_x1, batch_x2, batch_y) in enumerate(vali_loader):
            bs, stock_num = batch_x1.shape[0], batch_x1.shape[1]
            batch_x1 = batch_x1.reshape(-1, batch_x1.shape[-2], batch_x1.shape[-1]).float().to(self.device)
            batch_x2 = batch_x2.reshape(-1, batch_x2.shape[-2], batch_x2.shape[-1]).float().to(self.device)
            batch_y = batch_y.squeeze(-1).float().to(self.device)

            _, _, output = self.model(batch_x1, batch_x2)

            # output: [bs*stock_num, pred_len, c_out] -> [bs, stock_num]
            output = output[:, -1, 0:1]  # [bs*stock_num, 1]
            output = output.reshape(bs, stock_num)  # [bs, stock_num]

            loss = criterion(output, batch_y) + self.args.rank_alpha * ranking_loss(output, batch_y)

            total_loss.append(loss.item())

            with torch.no_grad():
                for metric in metric_objs:
                    metric.update(output, batch_y)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss, metric_objs
        
    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'valid')
        test_data, test_loader = self._get_data(flag = 'test')

        metrics_builders = [
            MIRRTop1Metric,
        ]

        # 根据架构类型创建checkpoint目录
        arch_type = getattr(self.args, 'arch_type', 'base')
        path = os.path.join('./checkpoints/', arch_type, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        metric_objs = [builder('train') for builder in metrics_builders]

        valid_loss_global = np.inf
        best_model_index = -1

        # 记录训练历史用于画图
        train_loss_history = []
        valid_loss_history = []
        test_loss_history = []

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            # batch_x1: 编码器输入, batch_x2: 解码器输入 batch_y: 预测目标
            for i, (batch_x1, batch_x2, batch_y) in enumerate(train_loader):
                iter_count += 1
                bs, stock_num = batch_x1.shape[0], batch_x1.shape[1]
                batch_x1 = batch_x1.reshape(-1, batch_x1.shape[-2], batch_x1.shape[-1]).float().to(self.device)
                batch_x2 = batch_x2.reshape(-1, batch_x2.shape[-2], batch_x2.shape[-1]).float().to(self.device)
                batch_y = batch_y.squeeze(-1).float().to(self.device)  # [bs, stock_num, 1] -> [bs, stock_num]

                _,_, output = self.model(batch_x1, batch_x2)

                # output: [bs*stock_num, pred_len, c_out] -> [bs, stock_num]
                # 取最后一个时间步，然后取第一个维度的输出作为预测值
                output = output[:, -1, 0:1]  # [bs*stock_num, 1]
                output = output.reshape(bs, stock_num)  # [bs, stock_num]

                loss = criterion(output, batch_y) + self.args.rank_alpha * ranking_loss(output, batch_y) # 计算损失：MSE损失 + 排名损失
                train_loss.append(loss.item())

                model_optim.zero_grad()
                loss.backward()
                model_optim.step()

                if (i+1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                with torch.no_grad():
                    for metric in metric_objs:
                        metric.update(output, batch_y)


            train_loss = np.average(train_loss)
            valid_loss, valid_metrics = self.vali(vali_data, vali_loader, criterion, metrics_builders, stage='valid')
            test_loss, test_metrics = self.vali(test_data, test_loader, criterion, metrics_builders, stage='test')

            # 记录损失历史
            train_loss_history.append(train_loss)
            valid_loss_history.append(valid_loss)
            test_loss_history.append(test_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Valid Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, valid_loss, test_loss))

            torch.save(self.model.state_dict(), path+'/'+'checkpoint_{0}.pth'.format(epoch+1))

            if valid_loss.item() < valid_loss_global:
                best_model_index = epoch+1

            adjust_learning_rate(model_optim, epoch+1, self.args)

        best_model_path = path+'/'+'checkpoint_{0}.pth'.format(best_model_index)
        self.model.load_state_dict(torch.load(best_model_path))
        print('best model index: ', best_model_index)

        # 绘制收敛曲线
        self._plot_training_curves(train_loss_history, valid_loss_history, test_loss_history, path)

        return self.model

    def _plot_training_curves(self, train_loss_history, valid_loss_history, test_loss_history, save_path):
        """绘制训练收敛曲线"""
        plt.figure(figsize=(12, 5))

        # 子图1: 训练损失
        plt.subplot(1, 2, 1)
        epochs = range(1, len(train_loss_history) + 1)
        plt.plot(epochs, train_loss_history, 'b-', label='Train Loss', linewidth=2)
        plt.plot(epochs, valid_loss_history, 'r-', label='Valid Loss', linewidth=2)
        plt.plot(epochs, test_loss_history, 'g-', label='Test Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (MSE + Ranking)', fontsize=12)
        plt.title('Prediction Fine-tuning Convergence', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        # 子图2: 对数尺度显示
        plt.subplot(1, 2, 2)
        plt.semilogy(epochs, train_loss_history, 'b-', label='Train Loss', linewidth=2)
        plt.semilogy(epochs, valid_loss_history, 'r-', label='Valid Loss', linewidth=2)
        plt.semilogy(epochs, test_loss_history, 'g-', label='Test Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (MSE + Ranking) - Log Scale', fontsize=12)
        plt.title('Prediction Fine-tuning Convergence (Log Scale)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图像
        plot_path = os.path.join(save_path, 'training_curves_supervised.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f'\n✅ Training curves saved to: {plot_path}')
        plt.close()

        # 保存损失数据到numpy文件，便于后续分析
        loss_data_path = os.path.join(save_path, 'loss_history.npz')
        np.savez(loss_data_path,
                 train_loss=train_loss_history,
                 valid_loss=valid_loss_history,
                 test_loss=test_loss_history)
        print(f'✅ Loss history saved to: {loss_data_path}')

    def test(self, setting):
        _, test_loader = self._get_data(flag='test')
        
        self.model.eval()

        metrics_builders = [
            MIRRTop1Metric,
            RankICMetric
        ]
        
        metric_objs = [builder('test') for builder in metrics_builders]
        
        for i, (batch_x1, batch_x2, batch_y) in enumerate(test_loader):
            bs, stock_num = batch_x1.shape[0], batch_x2.shape[1]
            batch_x1 = batch_x1.reshape(-1, batch_x1.shape[-2], batch_x1.shape[-1]).float().to(self.device)
            batch_x2 = batch_x2.reshape(-1, batch_x2.shape[-2], batch_x2.shape[-1]).float().to(self.device)
            batch_y = batch_y.squeeze(-1).float().to(self.device)

            _,_, output = self.model(batch_x1, batch_x2)

            # output: [bs*stock_num, pred_len, c_out] -> [bs, stock_num]
            output = output[:, -1, 0:1]  # [bs*stock_num, 1]
            output = output.reshape(bs, stock_num)  # [bs, stock_num]

            with torch.no_grad():
                for metric in metric_objs:
                    metric.update(output, batch_y)

        all_logs = {
            metric.name: metric.value for metric in metric_objs
        }
        for name, value in all_logs.items():
            print(name, value.mean())

        return 