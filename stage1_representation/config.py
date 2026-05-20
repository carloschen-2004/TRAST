import torch

# ========== 数据配置 ==========
# 使用所有CSI 
USE_CSI_300_TICKET = [
 '600519.SS',
 '601318.SS',
 '600036.SS',
 '000858.SZ',
 '600276.SS',
 '601166.SS',
 '601888.SS',
 '300059.SZ',
 '000651.SZ',
 '600900.SS',
 '600887.SS',
 '000001.SZ',
 '000725.SZ',
 '600030.SS',
 '300015.SZ',
 '601398.SS',
 '000568.SZ',
 '600031.SS',
 '600309.SS',
 '000002.SZ',
 '600809.SS',
 '601919.SS',
 '002142.SZ',
 '600436.SS',
 '601328.SS',
 '601899.SS',
 '002304.SZ',
 '002352.SZ',
 '002230.SZ',
 '300014.SZ',
 '600000.SS',
 '600438.SS',
 '600837.SS',
 '000661.SZ',
 '000100.SZ',
 '000063.SZ',
 '002241.SZ',
 '002271.SZ',
 '600585.SS',
 '600690.SS',
 '601601.SS',
 '601668.SS',
 '002027.SZ',
 '600016.SS',
 '600763.SS',
 '600196.SS',
 '000338.SZ',
 '600048.SS',
 '600703.SS',
 '002129.SZ',
 '600050.SS',
 '601688.SS',
 '600660.SS',
 '600104.SS',
 '600570.SS',
 '601766.SS',
 '601169.SS',
 '600999.SS',
 '002311.SZ',
 '002371.SZ',
 '600019.SS',
 '002049.SZ',
 '600406.SS',
 '601088.SS',
 '601988.SS',
 '000538.SZ',
 '000625.SZ',
 '600745.SS',
 '600028.SS',
 '600893.SS',
 '600346.SS',
 '601628.SS',
 '600588.SS',
 '601009.SS',
 '601390.SS',
 '601857.SS',
 '600009.SS',
 '600132.SS',
 '600584.SS',
 '000776.SZ',
 '000895.SZ',
 '002001.SZ',
 '600111.SS',
 '600426.SS',
 '601939.SS',
 '000166.SZ',
 '002050.SZ',
 '002179.SZ']

# 使用全部股票
use_ticker_dict = {'CSI': USE_CSI_300_TICKET}
use_ticker_dict = {'CSI':USE_CSI_300_TICKET, 'TEST': USE_CSI_300_TICKET[:5]}
CSI_date = ['20110419', '20181228', '20180102', '20201231',  '20190402', '20211231']
date_dict = {'CSI': CSI_date, 'TEST': CSI_date}
ticker_list = USE_CSI_300_TICKET

# ========== 技术指标 ==========
# stockstats技术指标（8个基础指标）
STOCKSTATS_INDICATORS = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]
# 子午投资内部因子
ZIYOU_INDICATORS = [
    "close_volume_cor",   # 量价相关性
    "capital_flow",       # 资金流向
    "weighted_skew",      # 加权偏度
]
# RDA因子
RDA_INDICATORS = [
    "vmon_20",            # 成交量变化率20日
    "vmon_50",            # 成交量变化率50日
    "vmon_100",           # 成交量变化率100日
    "klen",               # K线长度
]

# 合并所有技术指标
TECHNICAL_INDICATORS_LIST = STOCKSTATS_INDICATORS + ZIYOU_INDICATORS + RDA_INDICATORS

TEMPORAL_FEATURE = [
    'open', 'close', 'high', 'low', 'volume'
]

ADDITIONAL_FEATURE = [
    'label_short_term',
    'label_long_term'
]

INF = 1100

# ========== 阶段1：表征学习 ==========
STAGE1_CPU_PARAMS = {
    # 数据
    'seq_len': 60,             
    'label_len': 1,
    'pred_len': 1,

    # 模型规模
    'dec_in': 108,
    'c_out': 108,
    'd_model': 64,            
    'n_heads': 4,            
    'e_layers': 2,         
    'd_layers': 1,          
    'd_ff': 128,          
    'dropout': 0.1,

    # 训练参数
    'batch_size': 16,        
    'learning_rate': 0.0001,    
    'train_epochs': 10,         
    'patience': 3,
    'itr': 1,

    # 预测配置
    'short_term_len': 1,
    'long_term_len': 5,
}

# ========== 阶段2：策略学习 ==========
STAGE2_CPU_PARAMS = {
    # SAC参数
    'batch_size': 32,
    'buffer_size': 20000,  # 经验回放缓冲区大小
    'learning_rate': 0.0001,
    'learning_starts': 100,
    'ent_coef': "auto_0.05",  # 降低探索率，从0.1到0.05，加速收敛
    'total_timesteps': 500,  # 训练步数

    # Transformer参数 - 匹配Stage 1训练的模型
    'enc_in': 103,             # 15个技术指标 + 88个协方差 = 103
    'dec_in': 103,
    'c_out_construction': 103,
    'c_out_prediction': 1,
    'd_model': 32,             # 匹配Stage 1: 32
    'n_heads': 2,              # 匹配Stage 1: 2
    'e_layers': 1,             # 匹配Stage 1: 1
    'd_layers': 1,
    'd_ff': 64,                # 匹配Stage 1: 64
    'dropout': 0.1,
    'pred_len': 1,
    'seq_len': 60,

    # 环境参数
    'step_len': 500,
    'temporal_len': 60,
    'hidden_channel': 32,      # 匹配 d_model = 32
}

# MAE-SAC 模型参数 (基于STAGE2_CPU_PARAMS)
MAESAC_PARAMS = {
    "batch_size": STAGE2_CPU_PARAMS['batch_size'],
    "buffer_size": STAGE2_CPU_PARAMS['buffer_size'],
    "learning_rate": STAGE2_CPU_PARAMS['learning_rate'],
    "learning_starts": STAGE2_CPU_PARAMS['learning_starts'],
    "ent_coef": STAGE2_CPU_PARAMS['ent_coef'],
    "enc_in": STAGE2_CPU_PARAMS['enc_in'],
    "dec_in": STAGE2_CPU_PARAMS['dec_in'],
    "c_out_construction": STAGE2_CPU_PARAMS['c_out_construction'],
    "d_model": STAGE2_CPU_PARAMS['d_model'],
    "d_ff": STAGE2_CPU_PARAMS['d_ff'],
    "n_heads": STAGE2_CPU_PARAMS['n_heads'],
    "e_layers": STAGE2_CPU_PARAMS['e_layers'],
    "d_layers": STAGE2_CPU_PARAMS['d_layers'],
    "dropout": STAGE2_CPU_PARAMS['dropout'],
    "transformer_path": None,  # 运行时设置
}

# 路径配置
TRAINED_MODEL_DIR = "trained_models"
RESULTS_DIR = "results"

# 自动检测CPU
device = torch.device('cpu')
print(f" Using device: {device}")
print(f"  CPU模型优化模式:")
print(f"   - 股票数量: {len(USE_CSI_300_TICKET)} (保持全部)")
print(f"   - 序列长度: {STAGE1_CPU_PARAMS['seq_len']}")
print(f"   - 模型维度: {STAGE1_CPU_PARAMS['d_model']}")
print(f"   - 编码器层数: {STAGE1_CPU_PARAMS['e_layers']}")
print(f"   - 解码器层数: {STAGE1_CPU_PARAMS['d_layers']}")
print(f"   - 训练轮数: {STAGE1_CPU_PARAMS['train_epochs']}")
print(f"   - RL训练步数: {STAGE2_CPU_PARAMS['total_timesteps']}")
