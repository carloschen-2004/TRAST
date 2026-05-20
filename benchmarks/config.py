"""
基准策略配置文件
"""

# 初始资金
INITIAL_AMOUNT = 100000

# 交易成本（目前与TRAST环境一致，设为0）
TRANSACTION_COST_PCT = 0

# 年化交易日数
TRADING_DAYS_PER_YEAR = 252

# 基准策略配置
BENCHMARK_CONFIGS = {
    'buy_and_hold': {
        'name': 'Buy & Hold',
        'description': '在第一个交易日等权重买入所有股票，并持有到最后一个交易日',
        'class': 'BuyAndHoldStrategy',
        'params': {}
    },
    'equal_weight': {
        'name': 'Equal Weight',
        'description': '每个交易日都保持等权重配置，定期再平衡',
        'class': 'EqualWeightStrategy',
        'params': {
            'rebalance_freq': 5  # 每5天再平衡一次
        }
    },
    'momentum': {
        'name': 'Momentum',
        'description': '基于历史表现选择股票，买入近期表现最好的股票',
        'class': 'MomentumStrategy',
        'params': {
            'lookback_days': 20,  # 回顾20天
            'top_n': None  # None表示选择所有有数据的股票
        }
    }
}

# 绩效指标配置
METRICS_CONFIG = {
    'total_return': {
        'name': 'Total Return',
        'unit': '%',
        'description': '总收益率',
        'higher_better': True
    },
    'annualized_return': {
        'name': 'Annualized Return',
        'unit': '%',
        'description': '年化收益率',
        'higher_better': True
    },
    'annualized_volatility': {
        'name': 'Annualized Volatility',
        'unit': '%',
        'description': '年化波动率',
        'higher_better': False
    },
    'sharpe_ratio': {
        'name': 'Sharpe Ratio',
        'unit': '',
        'description': '夏普比率（风险调整后收益）',
        'higher_better': True
    },
    'max_drawdown': {
        'name': 'Max Drawdown',
        'unit': '%',
        'description': '最大回撤',
        'higher_better': False
    },
    'calmar_ratio': {
        'name': 'Calmar Ratio',
        'unit': '',
        'description': '卡玛比率（年化收益/最大回撤）',
        'higher_better': True
    }
}

# 图表配置
PLOT_CONFIG = {
    'colors': {
        'Buy & Hold': '#95a5a6',           # 灰色
        'Equal Weight': '#e74c3c',        # 红色
        'Momentum': '#3498db',            # 蓝色
        'TRAST (base)': '#2ecc71',        # 绿色
        'TRAST (MHA_RoPE_MoE)': '#9b59b6', # 紫色
        'TRAST (MQA_RoPE_MoE)': '#1abc9c', # 青色
        'TRAST (GQA_RoPE_MoE)': '#e91e63', # 粉色
    },
    'line_width': {
        'benchmark': 2.0,
        'trast': 3.0
    },
    'alpha': {
        'benchmark': 0.7,
        'trast': 1.0
    },
    'figure_size': {
        'main': (16, 9),
        'return': (16, 6),
        'metrics': (18, 5)
    }
}

# 数据集配置
DATASET_CONFIG = {
    'train': {
        'start_date': '2011-01-17',
        'end_date': '2018-12-28',
        'description': '训练集'
    },
    'eval': {
        'start_date': '2019-01-02',
        'end_date': '2021-12-31',
        'description': '验证集'
    },
    'test': {
        'start_date': '2022-01-01',
        'end_date': '2022-12-31',
        'description': '测试集'
    }
}

# 路径配置
PATH_CONFIG = {
    'data_dir': 'data/CSI/',
    'trast_results_dir': 'stage2_policy/results/',
    'benchmark_results_dir': 'benchmarks/results/',
    'temp_dir': 'temp/'
}

# 日志配置
LOG_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'benchmarks/logs/benchmark.log'
}