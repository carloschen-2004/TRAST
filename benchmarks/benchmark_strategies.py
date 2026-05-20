"""
基准策略实现模块
包含常见的股票投资基准策略，用于与TRAST架构进行对比
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os


class BenchmarkStrategy:
    """基准策略基类"""

    def __init__(self, name: str):
        self.name = name
        self.asset_memory = []
        self.actions_memory = []
        self.rewards_memory = []
        self.date_memory = []

    def run(self, df: pd.DataFrame, initial_amount: float = 100000,
            stock_dim: int = None, **kwargs) -> Dict:
        """
        运行基准策略

        Args:
            df: 股票数据DataFrame
            initial_amount: 初始资金
            stock_dim: 股票数量
            **kwargs: 其他参数

        Returns:
            包含结果的字典
        """
        raise NotImplementedError("子类必须实现此方法")

    def get_performance_metrics(self, trading_days_per_year: int = 252) -> Dict:
        """计算绩效指标"""
        if len(self.asset_memory) < 2:
            return {}

        account_value = np.array(self.asset_memory)
        daily_returns = pd.Series(account_value).pct_change().dropna()

        initial_value = account_value[0]
        final_value = account_value[-1]
        total_return = final_value / initial_value - 1

        num_periods = len(account_value) - 1
        annualized_return = (final_value / initial_value) ** (trading_days_per_year / num_periods) - 1

        daily_volatility = daily_returns.std()
        annualized_volatility = daily_volatility * np.sqrt(trading_days_per_year)

        if daily_volatility == 0 or pd.isna(daily_volatility):
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = np.sqrt(trading_days_per_year) * daily_returns.mean() / daily_volatility

        rolling_peak = pd.Series(account_value).cummax()
        drawdown = pd.Series(account_value) / rolling_peak - 1
        max_drawdown = drawdown.min()

        if max_drawdown == 0 or pd.isna(max_drawdown):
            calmar_ratio = 0.0
        else:
            calmar_ratio = annualized_return / abs(max_drawdown)

        return {
            'Initial': initial_value,
            'Final': final_value,
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Annualized Volatility': annualized_volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Calmar Ratio': calmar_ratio,
            'Max Account Value': account_value.max(),
        }

    def save_results(self, save_dir: str):
        """保存结果"""
        os.makedirs(save_dir, exist_ok=True)

        # 保存账户价值
        df_account_value = pd.DataFrame({
            'date': self.date_memory,
            'account_value': self.asset_memory
        })
        df_account_value.to_csv(os.path.join(save_dir, 'account_value.csv'), index=False)

        # 保存动作
        if self.actions_memory:
            df_actions = pd.DataFrame(self.actions_memory)
            # 确保日期长度与动作长度匹配
            dates_for_actions = self.date_memory[:len(df_actions)]
            df_actions.index = pd.to_datetime(dates_for_actions)
            df_actions.to_csv(os.path.join(save_dir, 'actions.csv'))

        # 保存绩效指标
        metrics = self.get_performance_metrics()
        df_metrics = pd.DataFrame([metrics])
        df_metrics.to_csv(os.path.join(save_dir, 'performance_metrics.csv'), index=False)


class BuyAndHoldStrategy(BenchmarkStrategy):
    """买入并持有策略 - 等权重买入所有股票并持有到最后"""

    def __init__(self):
        super().__init__("Buy & Hold")

    def run(self, df: pd.DataFrame, initial_amount: float = 100000,
            stock_dim: int = None, price_col: str = 'price', **kwargs) -> Dict:
        """
        运行买入并持有策略

        Args:
            df: 股票数据DataFrame
            initial_amount: 初始资金
            stock_dim: 股票数量
            price_col: 价格列名
            **kwargs: 其他参数

        Returns:
            包含结果的字典
        """
        if stock_dim is None:
            stock_dim = len(df.tic.unique())

        unique_dates = sorted(df.date.unique())
        ticker_list = df.tic.unique().tolist()

        # 第一天等权重买入所有股票
        first_date = unique_dates[0]
        first_day_data = df[df.date == first_date].set_index('tic')

        # 计算每只股票的买入金额
        amount_per_stock = initial_amount / stock_dim

        # 记录每只股票的持仓数量
        holdings = {}
        total_value = 0

        for ticker in ticker_list:
            if ticker in first_day_data.index:
                price = first_day_data.loc[ticker, price_col]
                if price > 0:
                    shares = amount_per_stock / price
                    holdings[ticker] = shares
                    total_value += shares * price
                else:
                    holdings[ticker] = 0
            else:
                holdings[ticker] = 0

        # 现金余额（如果有）
        cash = initial_amount - total_value
        self.asset_memory.append(initial_amount)
        self.date_memory.append(first_date)

        # 记录初始动作
        initial_actions = np.zeros(stock_dim)
        for i, ticker in enumerate(ticker_list):
            if holdings[ticker] > 0:
                initial_actions[i] = 1  # 买入
        self.actions_memory.append(initial_actions)

        # 持有并每天计算价值
        for date in unique_dates[1:]:
            day_data = df[df.date == date].set_index('tic')

            portfolio_value = cash
            for ticker in ticker_list:
                if ticker in day_data.index and ticker in holdings:
                    price = day_data.loc[ticker, price_col]
                    if price > 0:
                        portfolio_value += holdings[ticker] * price

            daily_return = (portfolio_value - self.asset_memory[-1]) / self.asset_memory[-1]
            self.asset_memory.append(portfolio_value)
            self.date_memory.append(date)
            self.rewards_memory.append(daily_return)

            # 持有期间动作为0
            self.actions_memory.append(np.zeros(stock_dim))

        return {
            'asset_memory': self.asset_memory,
            'actions_memory': self.actions_memory,
            'rewards_memory': self.rewards_memory,
            'date_memory': self.date_memory,
            'final_value': self.asset_memory[-1],
            'total_return': (self.asset_memory[-1] - initial_amount) / initial_amount
        }


class EqualWeightStrategy(BenchmarkStrategy):
    """等权重策略 - 每个交易日都保持等权重配置"""

    def __init__(self, rebalance_freq: int = 1):
        """
        Args:
            rebalance_freq: 再平衡频率（天数）
        """
        super().__init__(f"Equal Weight (Rebalance Every {rebalance_freq} Day)")
        self.rebalance_freq = rebalance_freq

    def run(self, df: pd.DataFrame, initial_amount: float = 100000,
            stock_dim: int = None, price_col: str = 'price', **kwargs) -> Dict:
        """
        运行等权重策略

        Args:
            df: 股票数据DataFrame
            initial_amount: 初始资金
            stock_dim: 股票数量
            price_col: 价格列名
            **kwargs: 其他参数

        Returns:
            包含结果的字典
        """
        if stock_dim is None:
            stock_dim = len(df.tic.unique())

        unique_dates = sorted(df.date.unique())
        ticker_list = df.tic.unique().tolist()

        # 初始化
        holdings = {}
        cash = initial_amount

        # 第一天等权重买入
        first_date = unique_dates[0]
        first_day_data = df[df.date == first_date].set_index('tic')

        amount_per_stock = initial_amount / stock_dim
        total_invested = 0

        for ticker in ticker_list:
            if ticker in first_day_data.index:
                price = first_day_data.loc[ticker, price_col]
                if price > 0:
                    shares = amount_per_stock / price
                    holdings[ticker] = shares
                    total_invested += shares * price
                else:
                    holdings[ticker] = 0
            else:
                holdings[ticker] = 0

        cash = initial_amount - total_invested

        # 计算第一天价值
        first_day_value = cash
        for ticker in ticker_list:
            if ticker in first_day_data.index:
                price = first_day_data.loc[ticker, price_col]
                if price > 0 and ticker in holdings:
                    first_day_value += holdings[ticker] * price

        self.asset_memory.append(first_day_value)
        self.date_memory.append(first_date)
        self.actions_memory.append(np.ones(stock_dim))  # 全部买入

        # 每个交易日处理
        for day_idx, date in enumerate(unique_dates[1:], 1):
            day_data = df[df.date == date].set_index('tic')

            # 计算当前组合价值
            portfolio_value = cash
            for ticker in ticker_list:
                if ticker in day_data.index and ticker in holdings:
                    price = day_data.loc[ticker, price_col]
                    if price > 0:
                        portfolio_value += holdings[ticker] * price

            # 再平衡
            if day_idx % self.rebalance_freq == 0:
                amount_per_stock = portfolio_value / stock_dim

                new_holdings = {}
                total_new_invested = 0

                for ticker in ticker_list:
                    if ticker in day_data.index:
                        price = day_data.loc[ticker, price_col]
                        if price > 0:
                            shares = amount_per_stock / price
                            new_holdings[ticker] = shares
                            total_new_invested += shares * price
                        else:
                            new_holdings[ticker] = 0
                    else:
                        new_holdings[ticker] = 0

                holdings = new_holdings
                cash = portfolio_value - total_new_invested

                # 计算调整后的价值
                portfolio_value = cash
                for ticker in ticker_list:
                    if ticker in day_data.index:
                        price = day_data.loc[ticker, price_col]
                        if price > 0 and ticker in holdings:
                            portfolio_value += holdings[ticker] * price

                self.actions_memory.append(np.ones(stock_dim))  # 再平衡
            else:
                self.actions_memory.append(np.zeros(stock_dim))  # 持有

            daily_return = (portfolio_value - self.asset_memory[-1]) / self.asset_memory[-1]
            self.asset_memory.append(portfolio_value)
            self.date_memory.append(date)
            self.rewards_memory.append(daily_return)

        return {
            'asset_memory': self.asset_memory,
            'actions_memory': self.actions_memory,
            'rewards_memory': self.rewards_memory,
            'date_memory': self.date_memory,
            'final_value': self.asset_memory[-1],
            'total_return': (self.asset_memory[-1] - initial_amount) / initial_amount
        }


class MomentumStrategy(BenchmarkStrategy):
    """动量策略 - 买入近期表现好的股票"""

    def __init__(self, lookback_days: int = 20, top_n: int = None):
        """
        Args:
            lookback_days: 回顾天数
            top_n: 选择表现最好的前N只股票，None表示全部
        """
        super().__init__(f"Momentum ({lookback_days}d Lookback)")
        self.lookback_days = lookback_days
        self.top_n = top_n

    def run(self, df: pd.DataFrame, initial_amount: float = 100000,
            stock_dim: int = None, price_col: str = 'price', **kwargs) -> Dict:
        """
        运行动量策略

        Args:
            df: 股票数据DataFrame
            initial_amount: 初始资金
            stock_dim: 股票数量
            price_col: 价格列名
            **kwargs: 其他参数

        Returns:
            包含结果的字典
        """
        if stock_dim is None:
            stock_dim = len(df.tic.unique())

        unique_dates = sorted(df.date.unique())
        ticker_list = df.tic.unique().tolist()

        # 如果数据不够，使用买入并持有
        if len(unique_dates) <= self.lookback_days:
            buy_hold = BuyAndHoldStrategy()
            return buy_hold.run(df, initial_amount, stock_dim, price_col, **kwargs)

        holdings = {}
        cash = initial_amount

        # 等待积累足够的历史数据
        for date_idx, date in enumerate(unique_dates):
            if date_idx < self.lookback_days:
                continue

            day_data = df[df.date == date].set_index('tic')

            # 计算当前组合价值
            portfolio_value = cash
            for ticker in ticker_list:
                if ticker in day_data.index and ticker in holdings:
                    price = day_data.loc[ticker, price_col]
                    if price > 0:
                        portfolio_value += holdings[ticker] * price

            if date_idx == self.lookback_days:
                # 第一次购买：基于动量选股
                lookback_start = unique_dates[0]
                lookback_end = unique_dates[self.lookback_days - 1]

                lookback_data = df[(df.date >= lookback_start) & (df.date <= lookback_end)]

                # 计算每只股票的收益率
                stock_returns = {}
                for ticker in ticker_list:
                    ticker_data = lookback_data[lookback_data.tic == ticker]
                    if len(ticker_data) >= 2:
                        start_price = ticker_data.iloc[0][price_col]
                        end_price = ticker_data.iloc[-1][price_col]
                        if start_price > 0 and end_price > 0:
                            stock_returns[ticker] = (end_price - start_price) / start_price

                # 选择表现最好的股票
                if self.top_n is None or self.top_n >= len(stock_returns):
                    selected_stocks = list(stock_returns.keys())
                else:
                    selected_stocks = sorted(stock_returns.items(),
                                           key=lambda x: x[1], reverse=True)[:self.top_n]
                    selected_stocks = [t[0] for t in selected_stocks]

                # 等权重买入
                if selected_stocks:
                    amount_per_stock = portfolio_value / len(selected_stocks)

                    new_holdings = {}
                    total_invested = 0

                    for ticker in selected_stocks:
                        if ticker in day_data.index:
                            price = day_data.loc[ticker, price_col]
                            if price > 0:
                                shares = amount_per_stock / price
                                new_holdings[ticker] = shares
                                total_invested += shares * price
                            else:
                                new_holdings[ticker] = 0
                        else:
                            new_holdings[ticker] = 0

                    holdings = new_holdings
                    cash = portfolio_value - total_invested

                    # 记录初始动作
                    initial_actions = np.zeros(stock_dim)
                    for i, ticker in enumerate(ticker_list):
                        if ticker in selected_stocks and holdings.get(ticker, 0) > 0:
                            initial_actions[i] = 1
                    self.actions_memory.append(initial_actions)

                    # 记录初始价值
                    initial_value = cash
                    for ticker in selected_stocks:
                        if ticker in day_data.index:
                            price = day_data.loc[ticker, price_col]
                            if price > 0 and ticker in holdings:
                                initial_value += holdings[ticker] * price

                    self.asset_memory.append(initial_value)
                    self.date_memory.append(date)
                else:
                    self.asset_memory.append(portfolio_value)
                    self.date_memory.append(date)
                    self.actions_memory.append(np.zeros(stock_dim))
            else:
                daily_return = (portfolio_value - self.asset_memory[-1]) / self.asset_memory[-1]
                self.asset_memory.append(portfolio_value)
                self.date_memory.append(date)
                self.rewards_memory.append(daily_return)
                self.actions_memory.append(np.zeros(stock_dim))  # 持有

        return {
            'asset_memory': self.asset_memory,
            'actions_memory': self.actions_memory,
            'rewards_memory': self.rewards_memory,
            'date_memory': self.date_memory,
            'final_value': self.asset_memory[-1],
            'total_return': (self.asset_memory[-1] - initial_amount) / initial_amount
        }


def run_all_benchmarks(df: pd.DataFrame, initial_amount: float = 100000,
                      stock_dim: int = None, save_dir: str = 'benchmarks/results') -> Dict[str, BenchmarkStrategy]:
    """
    运行所有基准策略

    Args:
        df: 股票数据DataFrame
        initial_amount: 初始资金
        stock_dim: 股票数量
        save_dir: 结果保存目录

    Returns:
        包含所有基准策略结果的字典
    """
    strategies = {
        'BuyAndHold': BuyAndHoldStrategy(),
        'EqualWeight': EqualWeightStrategy(rebalance_freq=5),
        'Momentum': MomentumStrategy(lookback_days=20, top_n=stock_dim//4 if stock_dim else None),
    }

    results = {}
    for name, strategy in strategies.items():
        print(f"Running {strategy.name}...")
        _ = strategy.run(df, initial_amount, stock_dim)  # 忽略返回值
        results[name] = strategy

        # 保存结果
        strategy_save_dir = os.path.join(save_dir, name)
        strategy.save_results(strategy_save_dir)

        # 打印绩效
        metrics = strategy.get_performance_metrics()
        print(f"  Total Return: {metrics['Total Return'] * 100:.2f}%")
        print(f"  Sharpe Ratio: {metrics['Sharpe Ratio']:.4f}")
        print(f"  Max Drawdown: {metrics['Max Drawdown'] * 100:.2f}%")

    return results