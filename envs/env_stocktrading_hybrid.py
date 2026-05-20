import numpy as np
import pandas as pd
from gymnasium.utils import seeding
import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'stage1_representation'))
# 导入所有架构的模型
from models.transformer import (
    Transformer_base,
    Transformer_MHA_RoPE_MoE,
    Transformer_MQA_RoPE_MoE,
    Transformer_GQA_RoPE_MoE
)
import torch
from collections import OrderedDict


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        stock_dim,
        hmax,
        initial_amount,
        transaction_cost_pct,
        reward_scaling,
        state_space,
        action_space,
        tech_indicator_list,
        temporal_feature_list,
        additional_list,
        time_window_start, # should be a list
        short_prediction_model_path = None,
        long_prediction_model_path = None,
        step_len = 1000,
        temporal_len = 60,
        figure_path = 'results/',
        logs_path = 'results/',
        csv_path = 'results/',
        mode = "train",
        hidden_channel = 4,
        make_plots = True,
        print_verbosity = 1,
        initial = True,
        model_name = "",
        iteration = "",
        device = 'cuda:0',
        print_additional_flag = 0,
        enc_in = 103,  # Stage1的enc_in
        dec_in = 103,
        arch_type = 'base',  # 默认使用基础架构
        d_model = 64,  # Stage1的模型参数
        n_heads = 2,
        e_layers = 2,
        d_layers = 1,
        d_ff = 128,
        num_experts = 4,
        top_k = 2,
        n_kv_heads = None,
    ):
        # start time
        self.start_day = time_window_start[0]
        self.day = self.start_day
        self.time_window_start = time_window_start
        self.time_windows_point = 0
        self.step_len = step_len

        # 保存模型参数供后续使用
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.arch_type = arch_type
        self.d_model = d_model

        # help file
        self.log_name = logs_path + mode + '.txt'
        self.figure_path = figure_path
        self.csv_path = csv_path
        os.makedirs(logs_path, exist_ok = True)
        os.makedirs(figure_path, exist_ok = True)
        os.makedirs(csv_path, exist_ok = True)

        self.df = df
        # 处理长格式数据：获取唯一日期列表
        self.unique_dates = sorted(df.date.unique())
        self.max_trading_days = len(self.unique_dates)

        self.stock_dim = stock_dim
        self.initial_amount = initial_amount
        self.hmax = hmax
        self.transaction_cost_pct = transaction_cost_pct

        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_dim = action_space
        self.tech_indicator_list = tech_indicator_list
        self.temporal_feature_list = temporal_feature_list
        self.additional_list = additional_list
        self.temporal_len = temporal_len
        # 注意：hidden_channel 会被实际的模型输出维度覆盖
        self.hidden_channel_config = hidden_channel  # 保存配置值

        self.action_space = spaces.Box(low = -1, high = 1, shape = (self.action_dim,))
        # observation_space 会在加载模型后重新计算
        self.observation_space = None
        self.hidden_state_space = None

        # 获取当前时间步的数据（所有股票在这一天的数据）
        current_date = self.unique_dates[self.day]
        self.data = self.df[self.df.date == current_date].reset_index(drop=True)
        self.tic = self.df.tic.unique()
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.initial = initial
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration

        # load model
        # 如果CUDA不可用，自动切换到CPU
        if isinstance(device, str) and 'cuda' in device:
            if not torch.cuda.is_available():
                device = 'cpu'
                if print_verbosity > 0:
                    print("CUDA not available, using CPU for device")
        self.device = device

        # 加载模型 - 使用与Stage1的pred模型一致的参数
        self.short_prediction_model = self.load_model(
            short_prediction_model_path,
            enc_in = enc_in, dec_in = dec_in, c_out = 1,  # 预测未来1天的价格
            d_model = d_model, n_heads = n_heads, e_layers = e_layers,
            d_layers = d_layers, d_ff = d_ff,
            arch_type = arch_type,
            num_experts = num_experts,
            top_k = top_k,
            n_kv_heads = n_kv_heads
        ).to(self.device)

        self.long_prediction_model = self.load_model(
            long_prediction_model_path,
            enc_in = enc_in, dec_in = dec_in, c_out = 1,  # 预测未来5天的价格
            d_model = d_model, n_heads = n_heads, e_layers = e_layers,
            d_layers = d_layers, d_ff = d_ff,
            arch_type = arch_type,
            num_experts = num_experts,
            top_k = top_k,
            n_kv_heads = n_kv_heads
        ).to(self.device)
      
        self.short_prediction_model.eval() # 设置模型为评估模式
        self.long_prediction_model.eval() # 设置模型为评估模式

        # additional list
        self.print_additional_flag = print_additional_flag
        self.short_hidden_feature = []
        self.long_hidden_feature = []

        # initalize state and info
        self.info = self._initiate_info()
        self.state = self._initial_state()

        # 根据实际的模型输出维度设置 observation_space
        # hidden_np1 和 hidden_np2 的形状是 (stock_dim, actual_hidden_dim)
        actual_hidden_dim = self.short_hidden_feature[0].shape[1] if len(self.short_hidden_feature) > 0 else d_model
        self.hidden_channel = actual_hidden_dim  # 使用实际的隐藏层维度

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.state_space, self.state_space + len(self.tech_indicator_list) + 2 * self.hidden_channel + 1)
        )
        self.hidden_state_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.state_space, self.hidden_channel + 1)
        )

        print(f"✅ 观察空间维度设置:")
        print(f"   state_space: {self.state_space}")
        print(f"   tech_indicator_list: {len(self.tech_indicator_list)}")
        print(f"   hidden_channel (实际): {self.hidden_channel}")
        print(f"   observation_space shape: {self.observation_space.shape}")

        # initialize reward
        self.reward = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        # memorize all the total balance change
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.amount_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        # self.reset()
        self._seed()


    def _sell_stock(self, index, action):
        def _do_sell_normal():
            if self.info[index + 1] > 0: # info[index + 1] is the price of the stock
                # Sell only if the price is > 0 (no missing data in this particular date)
                if self.info[index + self.stock_dim + 1] > 0: # info[index + self.stock_dim + 1] is the amount of stock holding
                    # Sell only if current asset is > 0
                    sell_num_shares = min(
                        abs(action), self.info[index + self.stock_dim + 1]
                    )
                    sell_amount = (
                        self.info[index + 1]
                        * sell_num_shares
                        * (1 - self.transaction_cost_pct) # subtract transaction cost 
                    )
                    # update balance
                    self.info[0] += sell_amount # add the sell amount to cash balance
                    self.info[index + self.stock_dim + 1] -= sell_num_shares # subtract the sold shares from holding amount
                    self.cost += (self.info[index + 1] * sell_num_shares * self.transaction_cost_pct) # calculate the transaction cost and add to total cost 
                    self.trades += 1
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0

            return sell_num_shares

        sell_num_shares = _do_sell_normal()
        return sell_num_shares


    def _buy_stock(self, index, action):
        def _do_buy():
            if self.info[index + 1] > 0: # info[index + 1] is the price of the stock
                # Buy only if the price is > 0 (no missing data in this particular date)
                available_amount = self.info[0] // self.info[index + 1]
                # update balance
                buy_num_shares = min(available_amount, action)
                buy_amount = (
                    self.info[index + 1] * buy_num_shares * (1 + self.transaction_cost_pct)
                )
                self.info[0] -= buy_amount # subtract the buy amount from cash balance
                self.info[index + self.stock_dim + 1] += buy_num_shares # add the bought shares to holding amount
                self.cost += (self.info[index + 1] * buy_num_shares * self.transaction_cost_pct) # calculate the transaction cost and add to total cost 
                self.trades += 1
            else:
                buy_num_shares = 0

            return buy_num_shares

        buy_num_shares = _do_buy()
        return buy_num_shares


    def _make_plot(self):
        plt.plot(self.asset_memory, "r")
        plt.savefig(self.figure_path + self.mode + "_account_value_trade_{}.png".format(self.episode))
        plt.close()


    def step(self, actions):
        # 训练模式
        if self.mode == 'train':
            self.terminal = (self.day - self.start_day) >= self.step_len + 1 # episode
        # 测试模式
        else:
            self.terminal = self.day >= self.max_trading_days - 1
        if self.terminal:
            if self.make_plots:
                self._make_plot()
            self.end_total_asset = self.info[0] + sum(
                np.array(self.info[1 : (self.stock_dim + 1)])
                * np.array(self.info[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            ) # calculate the total asset value at the end of the episode
            df_total_value = pd.DataFrame(self.asset_memory)
            tot_reward = (
                self.end_total_asset
                - self.initial_amount
            ) # absolute reward
            df_total_value.columns = ["account_value"]
            df_total_value["date"] = self.date_memory
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(1)
            if df_total_value["daily_return"].std() != 0:
                sharpe = (
                    (252 ** 0.5)
                    * df_total_value["daily_return"].mean()
                    / df_total_value["daily_return"].std()
                )

            self.reward = self.reward + self.reward_scaling * ((self.end_total_asset - self.initial_amount)/(self.initial_amount * 1.0))

            f1 = open(self.log_name, 'a')
            # 保存到日志文件
            f1.write(str(self.end_total_asset) + '\t' + str(self.reward)+ '\t' + str(np.sum(self.rewards_memory)) + '\t' + str(sharpe) + '\t' + str((self.end_total_asset-self.initial_amount)/self.initial_amount) + '\n')
            f1.close()

            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ["account_rewards"]
            df_rewards["date"] = self.date_memory[:-1]

            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {self.end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                if df_total_value["daily_return"].std() != 0:
                    print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")
            # 保存CSV文件
            if (self.model_name != "") and (self.mode != ""):
                df_actions = self.save_action_memory()
                df_actions.to_csv(
                    self.csv_path+"actions_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.episode
                    )
                )
                df_stock_amount = self.save_holding_amount()
                df_stock_amount.to_csv(
                    self.csv_path + "amount_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.episode
                    )
                )
                df_total_value.to_csv(
                    self.csv_path + "account_value_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.episode
                    ),
                    index = False,
                )
                df_rewards.to_csv(
                    self.csv_path + "account_rewards_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.episode
                    ),
                    index = False,
                )
                # 保存资产曲线图
                plt.plot(self.asset_memory, "r")
                plt.savefig(
                    self.figure_path + "account_value_{}_{}_{}.png".format(
                        self.mode, self.model_name, self.episode
                    )
                )
                plt.close()

            terminated = self.terminal
            truncated = False
            info = {}
            return self.state, self.reward, terminated, truncated, info

        else:
            actions = actions * self.hmax  # [-1, 1] -> [-hmax, hmax]
            actions = actions.astype(int) # 转为整数
            begin_total_asset = self.info[0] + sum(
                np.array(self.info[1 : (self.stock_dim + 1)])
                * np.array(self.info[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            ) # 计算初始总资产

            argsort_actions = np.argsort(actions) # 从小到大排序
            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]] # 卖出索引：最小的几个
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]] # 买入索引：最大的几个（倒序）

            for index in sell_index: # 先卖出股票，释放资金，再买入股票
                actions[index] = self._sell_stock(index, actions[index]) * (-1)

            for index in buy_index:
                actions[index] = self._buy_stock(index, actions[index])

            self.actions_memory.append(actions)

            # state: s -> s+1
            self.day += 1
            # 更新当前时间步的数据（所有股票在这一天的数据）
            current_date = self.unique_dates[self.day]
            self.data = self.df[self.df.date == current_date].reset_index(drop=True)
            self.info = self._update_info()
            self.end_total_asset = self.info[0] + sum(
                np.array(self.info[1 : (self.stock_dim + 1)])
                * np.array(self.info[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            self.state = self._update_state()
            self.asset_memory.append(self.end_total_asset)
            self.date_memory.append(self._get_date())
            self.reward = ((self.end_total_asset - begin_total_asset)/(begin_total_asset * 1.0))
            self.rewards_memory.append(self.reward)
            self.amount_memory.append(self.info[-self.stock_dim:])

        # terminated = episode 自然结束
        # truncated = episode 被截断（例如超过步数）
        terminated = self.terminal
        truncated = False  # 我们的环境中没有时间截断
        info = {}
        return self.state, self.reward, terminated, truncated, info


    def _initiate_info(self):
        # eg. for 3 stocks, info = [cash, price1, price2, price3, amount1, amount2, amount3]
        try:
            if isinstance(self.data, pd.DataFrame):
                prices = self.data.price.values.tolist()
            elif isinstance(self.data, pd.Series):
                prices = self.data.values.tolist()
            else:
                prices = list(self.data)
        except Exception as e:
            print(f"Warning: Error accessing prices, using zeros: {e}")
            prices = [0.0] * self.stock_dim

        info = (
                    [self.initial_amount]
                    + prices
                    + [0] * self.stock_dim
            )
        return info


    def _initial_state(self):
        if isinstance(self.data, pd.DataFrame):
            covs = np.array(self.data['cov_list'].values[0])  # (stock_dim, stock_dim)
            technical_indicators = np.array(self.data[self.tech_indicator_list].values.tolist())  # (stock_dim, len(technical_list))
        else:
            covs = self.data['cov_list'][0]  # (stock_dim, stock_dim)
            technical_indicators = np.stack([self.data[ind] for ind in self.tech_indicator_list]).T  # (stock_dim, len(technical_list))

        start_day_idx = max(0, self.day - self.temporal_len + 1)
        target_dates = self.unique_dates[start_day_idx : self.day + 1]

        temporal_feature_data = self.df[self.df.date.isin(target_dates)].sort_values(['date', 'tic'])

        available_dates = len(target_dates)
        actual_temporal_len = min(self.temporal_len, available_dates)

        if available_dates > 0 and len(temporal_feature_data) >= self.stock_dim:
            # 将长格式数据转换为宽格式：(日期数, 股票数, 特征数)
            temporal_feature_list = []
            for date in target_dates:
                date_data = temporal_feature_data[temporal_feature_data.date == date]
                if len(date_data) == self.stock_dim:
                    features = date_data[self.temporal_feature_list].values
                    temporal_feature_list.append(features)
                elif len(date_data) > 0:
                    # 如果某个日期的股票数据不全，用0填充
                    features = np.zeros((self.stock_dim, len(self.temporal_feature_list)))
                    # 按股票顺序填充
                    for idx, tic in enumerate(self.tic):
                        tic_data = date_data[date_data.tic == tic]
                        if len(tic_data) > 0:
                            features[idx] = tic_data[self.temporal_feature_list].values[0]
                    temporal_feature_list.append(features)

            temporal_feature = np.array(temporal_feature_list).transpose(1, 0, 2)  # (stock_dim, 日期数, 特征数)

            # 如果需要，填充到temporal_len
            if available_dates < self.temporal_len:
                # 用第一天的数据填充
                padding = np.tile(temporal_feature[:, :1, :], (1, self.temporal_len - available_dates, 1))
                temporal_feature = np.concatenate([padding, temporal_feature], axis=1)
        else:
            # 如果完全没有数据，创建零矩阵
            temporal_feature = np.zeros((self.stock_dim, self.temporal_len, len(self.temporal_feature_list)))

        # 将协方差矩阵加入到输入中：88个协方差 + 8个技术指标 = 96维
        cov_diagonal = np.diagonal(covs)  # (stock_dim,) = (88,)

        # 为每个时间步添加协方差特征
        # 目标形状: (stock_dim, temporal_len, 96) = (88, temporal_len, 88+8)
        feature_with_cov = []
        for i in range(self.stock_dim):
            # 对于每个股票，将其协方差值和技术指标拼接
            stock_temporal = temporal_feature[i]  # (temporal_len, 8)
            # 复制协方差值 temporal_len 次
            stock_cov = np.tile(cov_diagonal[i:i+1], (self.temporal_len, 1))  # (temporal_len, 1)
            # 拼接所有股票的协方差值（88维）
            all_covs = np.tile(cov_diagonal, (self.temporal_len, 1))  # (temporal_len, 88)
            # 最终拼接: (temporal_len, 88+8) = (temporal_len, 96)
            stock_feature = np.concatenate([all_covs, stock_temporal], axis=1)
            feature_with_cov.append(stock_feature)

        temporal_feature = np.array(feature_with_cov)  # (stock_dim, temporal_len, 96)

        # 如果 self.enc_in > 96，填充零使其匹配
        current_feature_dim = temporal_feature.shape[2]
        if self.enc_in > current_feature_dim:
            padding_dim = self.enc_in - current_feature_dim
            # 在最后一维填充零
            padding = np.zeros((self.stock_dim, self.temporal_len, padding_dim))
            temporal_feature = np.concatenate([temporal_feature, padding], axis=2)
            if hasattr(self, 'verbose') and self.verbose:
                print(f"   填充 temporal_feature: {current_feature_dim} -> {self.enc_in} (填充 {padding_dim} 维)")

        enc_feature = torch.FloatTensor(temporal_feature).to(self.device)
        dec_feature = torch.FloatTensor(temporal_feature[:, -1:, :]).to(self.device)

        _, hidden_short, _ = self.short_prediction_model(enc_feature, dec_feature)
        _, hidden_long, _ = self.long_prediction_model(enc_feature, dec_feature)

        hidden_np1 = hidden_short.detach().cpu().numpy().reshape(self.stock_dim, -1)
        hidden_np2 = hidden_long.detach().cpu().numpy().reshape(self.stock_dim, -1)

        self.short_hidden_feature.append(hidden_np1)
        self.long_hidden_feature.append(hidden_np2)

        holding_amount = np.zeros((self.stock_dim,1), dtype = int)
        state = np.concatenate((
            covs, technical_indicators, hidden_np1, hidden_np2, holding_amount
            ), axis = -1)
        return state


    def _update_info(self):
        # for multiple stock
        info = (
                [self.info[0]]
                + self.data.price.values.tolist()
                + list(self.info[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
        return info


    def _update_state(self):
        # 获取协方差矩阵和技术指标
        covs = np.array(self.data['cov_list'].values[0]) # (stock_dim, stock_dim)
        technical_indicators = np.array(self.data[self.tech_indicator_list].values.tolist()) # (stock_dim, len(technical_list))

        # 提取时序特征（过去temporal_len天）
        start_day_idx = max(0, self.day - self.temporal_len + 1)
        target_dates = self.unique_dates[start_day_idx : self.day + 1]

        # 获取这些日期的所有股票数据
        temporal_feature_data = self.df[self.df.date.isin(target_dates)].sort_values(['date', 'tic'])

        # 将长格式数据转换为宽格式
        temporal_feature_list = []
        for date in target_dates:
            date_data = temporal_feature_data[temporal_feature_data.date == date]
            if len(date_data) == self.stock_dim:
                features = date_data[self.temporal_feature_list].values
                temporal_feature_list.append(features)
            elif len(date_data) > 0:
                # 如果某个日期的股票数据不全，用0填充
                features = np.zeros((self.stock_dim, len(self.temporal_feature_list)))
                for idx, tic in enumerate(self.tic):
                    tic_data = date_data[date_data.tic == tic]
                    if len(tic_data) > 0:
                        features[idx] = tic_data[self.temporal_feature_list].values[0]
                temporal_feature_list.append(features)

        temporal_feature = np.array(temporal_feature_list).transpose(1, 0, 2)  # (stock_dim, 日期数, 特征数)

        # 将协方差矩阵加入到输入中：88个协方差 + 8个技术指标 = 96维
        cov_diagonal = np.diagonal(covs)  # (stock_dim,) = (88,)
        temporal_len = temporal_feature.shape[1]  # 实际的时间长度

        feature_with_cov = []
        for i in range(self.stock_dim):
            stock_temporal = temporal_feature[i]  # (temporal_len, 8)
            # 拼接所有股票的协方差值（88维）
            all_covs = np.tile(cov_diagonal, (temporal_len, 1))  # (temporal_len, 88)
            # 最终拼接: (temporal_len, 88+8) = (temporal_len, 96)
            stock_feature = np.concatenate([all_covs, stock_temporal], axis=1)
            feature_with_cov.append(stock_feature)

        temporal_feature = np.array(feature_with_cov)  # (stock_dim, temporal_len, 96)

        # 如果 self.enc_in > 96，填充零使其匹配
        current_feature_dim = temporal_feature.shape[2]
        if self.enc_in > current_feature_dim:
            padding_dim = self.enc_in - current_feature_dim
            padding = np.zeros((self.stock_dim, temporal_feature.shape[1], padding_dim))
            temporal_feature = np.concatenate([temporal_feature, padding], axis=2)

        enc_feature = torch.FloatTensor(temporal_feature).to(self.device)
        dec_feature = torch.FloatTensor(temporal_feature[:, -1:, :]).to(self.device)

        # 通过双Transformer模型获取隐藏特征
        _, hidden_short, _ = self.short_prediction_model(enc_feature, dec_feature)
        _, hidden_long, _ = self.long_prediction_model(enc_feature, dec_feature)
        hidden_np1 = hidden_short.detach().cpu().numpy().reshape(self.stock_dim, -1)
        hidden_np2 = hidden_long.detach().cpu().numpy().reshape(self.stock_dim, -1)
        self.short_hidden_feature.append(hidden_np1)
        self.long_hidden_feature.append(hidden_np2)
        
        # 计算持仓比例（归一化）
        holding_amount = np.array(self.info[-self.stock_dim : ]) # (stock_dim, 1)
        holding_amount_norm = ((holding_amount * np.array(self.info[1: 1 + self.stock_dim]))/self.end_total_asset).reshape(self.stock_dim, 1)

        state = np.concatenate((covs, technical_indicators, hidden_np1, hidden_np2, holding_amount_norm), axis = -1)

        return state


    def _get_date(self):
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date


    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )
        return df_account_value


    def save_holding_amount(self):
        date_list = self.date_memory[:-1]
        df_date = pd.DataFrame(date_list)
        df_date.columns = ["date"]

        amount_list = self.amount_memory
        df_amount = pd.DataFrame(amount_list)
        df_amount.columns = self.data.tic.values
        return df_amount  


    def save_action_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})
        return df_actions


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def reset(self, seed=None, options=None):
        """重置环境，返回初始状态"""
        super().reset(seed=seed)
        self.day = self.start_day
        self.time_windows_point = 0
        self.terminal = False
        self.episode += 1

        # 重新初始化状态
        self.info = self._initiate_info()
        self.state = self._initial_state()

        # 重置记忆
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.amount_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        self.short_hidden_feature = []
        self.long_hidden_feature = []

        # 重置奖励
        self.reward = 0
        self.cost = 0
        self.trades = 0

        # reset() 应该返回 (observation, info)，不是5个值
        # gymnasium API 要求：reset() -> (obs, info)
        info = {}
        return self.state, info


    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs


    def load_model(self, path, 
                   enc_in = 10, 
                   dec_in = 10, 
                   c_out = 1,
                   d_model = 32, 
                   n_heads = 2, 
                   e_layers = 1, 
                   d_layers = 1, 
                   d_ff = 64,
                   arch_type = 'base', 
                   num_experts = 4, 
                   top_k = 2, 
                   n_kv_heads = None):
        """加载预测模型，支持不同架构和完整的模型参数

        Args:
            path: checkpoint路径
            enc_in, dec_in, c_out: 输入输出维度
            d_model, n_heads, e_layers, d_layers, d_ff: 模型参数
            arch_type: 架构类型 ('base', 'MHA_RoPE_MoE', 'MQA_RoPE_MoE', 'GQA_RoPE_MoE')
            num_experts, top_k, n_kv_heads: MoE和GQA相关参数
        """
        # 根据架构类型选择模型类
        if arch_type == 'base':
            model = Transformer_base(
                enc_in, dec_in, c_out,
                d_model, n_heads, e_layers, d_layers, d_ff,
                dropout = 0.1, activation = 'gelu'
            )
        elif arch_type == 'MHA_RoPE_MoE':
            model = Transformer_MHA_RoPE_MoE(
                enc_in, dec_in, c_out,
                d_model, n_heads, e_layers, d_layers, d_ff,
                num_experts, top_k,
                dropout = 0.1, activation = 'gelu'
            )
        elif arch_type == 'MQA_RoPE_MoE':
            model = Transformer_MQA_RoPE_MoE(
                enc_in, dec_in, c_out,
                d_model, n_heads, e_layers, d_layers, d_ff,
                num_experts, top_k,
                dropout = 0.1, activation = 'gelu'
            )
        elif arch_type == 'GQA_RoPE_MoE':
            model = Transformer_GQA_RoPE_MoE(
                enc_in, dec_in, c_out,
                d_model, n_heads, n_kv_heads, e_layers, d_layers, d_ff,
                num_experts, top_k,
                dropout = 0.1, activation = 'gelu'
            )
        else:
            raise ValueError(f"Unknown architecture type: {arch_type}")

        if path is not None:
            state_dict = torch.load(path, map_location = self.device)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    name = k[7:]
                else:
                    name = k
                # 跳过输出层（projection），因为维度不匹配
                # Stage1的c_out=103，但Stage2需要c_out=1
                if 'projection' in name:
                    continue
                new_state_dict[name] = v

            # 严格模式设置为False，允许部分加载
            model.load_state_dict(new_state_dict, strict=False)
            print(f"✅ Successfully loaded {arch_type} model from {path}")
            print(f"   Model params: d_model={d_model}, n_heads={n_heads}, e_layers={e_layers}, d_ff={d_ff}")
            print(f"   Note: Output layer reinitialized for prediction (c_out={c_out})")
        else:
            print(f"Initialize new {arch_type} model with d_model={d_model}")

        return model
