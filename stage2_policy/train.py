import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('..')
import stage2_policy.config as config
from stage2_policy.preprocess import FeatureEngineer, data_split
from stage2_policy.models.DRLAgent import DRLAgent
from stable_baselines3.common.vec_env import VecMonitor
from envs.env_stocktrading_hybrid import StockTradingEnv as Env
from sklearn.preprocessing import StandardScaler
from stage2_policy.config import TECHNICAL_INDICATORS_LIST, INF
import time
import random
import torch
torch.set_num_threads(4)  # 限制CPU线程数

fix_seed = 2026
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# 添加命令行参数
parser = argparse.ArgumentParser(description='Stage2 Training with Different Architectures')
parser.add_argument('--arch_type', type=str, default='base',
                    choices=['base', 'MHA_RoPE_MoE', 'MQA_RoPE_MoE', 'GQA_RoPE_MoE'],
                    help='Architecture type from stage1')
args = parser.parse_args()

arch_type = args.arch_type
version = 'CSI/'
model_name = f'TRAST_{arch_type}/'

# 查找对应架构的checkpoint
def find_best_checkpoint(arch_type, exp_type='mae'):
    """
    找到指定架构和实验类型的最佳checkpoint

    Args:
        arch_type: 架构类型 (base, MHA_RoPE_MoE, etc.)
        exp_type: 实验类型 (mae 或 pred)

    Returns:
        checkpoint_path: 最佳模型路径
    """
    base_path = f'../stage1_representation/checkpoints/{arch_type}/'
    if not os.path.exists(base_path):
        raise ValueError(f"Architecture directory not found: {base_path}")

    # 查找所有 {exp_type}_* 目录
    exp_dirs = [d for d in os.listdir(base_path)
                if d.startswith(f'{exp_type}_') and os.path.isdir(os.path.join(base_path, d))]

    if not exp_dirs:
        raise ValueError(f"No {exp_type.upper()} checkpoint found in {base_path}")

    # 使用最新的目录
    exp_dirs.sort()
    latest_dir = exp_dirs[-1]
    checkpoint_dir = os.path.join(base_path, latest_dir)

    # 读取训练日志找到最佳模型索引
    loss_history_path = os.path.join(checkpoint_dir, 'loss_history.npz')
    if os.path.exists(loss_history_path):
        # 从loss_history中找到验证集损失最小的epoch
        data = np.load(loss_history_path)
        valid_loss = data['valid_loss']
        best_epoch = np.argmin(valid_loss) + 1  # epoch从1开始
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{best_epoch}.pth')
        print(f"✅ 找到 {arch_type} 的最佳 {exp_type.upper()} 模型 (epoch {best_epoch}): {checkpoint_path}")
    else:
        # 如果没有loss_history，查找checkpoint_10.pth（默认最佳）
        checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_10.pth')

        if not os.path.exists(checkpoint_path):
            # 如果没有checkpoint_10，查找最大的编号
            checkpoints = [f for f in os.listdir(checkpoint_dir)
                          if f.startswith('checkpoint_') and f.endswith('.pth')]
            if checkpoints:
                checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
                checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])
            else:
                raise ValueError(f"No checkpoint found in {checkpoint_dir}")

        print(f"✅ 找到 {arch_type} 的最佳 {exp_type.upper()} 模型: {checkpoint_path}")

    return checkpoint_path

# 获取MAE模型路径（用于特征提取）
mae_model_path = find_best_checkpoint(arch_type, exp_type='mae')

# 获取PRED模型路径（用于预测）
pred_model_path = find_best_checkpoint(arch_type, exp_type='pred')
# Stage2使用MAE进行特征提取，使用PRED进行收益预测
short_prediction_model_path = pred_model_path  # 短期预测模型
long_prediction_model_path = pred_model_path   # 长期预测模型

# 数据路径 - 使用全部股票
full_stock_dir = '../data/CSI/'
ticker_list = config.USE_CSI_300_TICKET
prediction_len = [1,5]

# 创建按架构分离的输出目录
trained_model_dir = os.path.join(config.TRAINED_MODEL_DIR, arch_type)
results_dir = os.path.join(config.RESULTS_DIR, arch_type)

if not os.path.exists(trained_model_dir):
    os.makedirs(trained_model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

print("="*80)
print(f" StockFormer Stage2 Training - Architecture: {arch_type}")
print("="*80)
print(f" Configuration:")
print(f"  - Architecture: {arch_type}")
print(f"  - Stock数量: {len(ticker_list)}")
print(f"  - 训练步数: {config.STAGE2_CPU_PARAMS['total_timesteps']}")
print(f"  - Batch size: {config.STAGE2_CPU_PARAMS['batch_size']}")
print(f"  - Buffer size: {config.STAGE2_CPU_PARAMS['buffer_size']}")
print(f"  - 模型维度: {config.STAGE2_CPU_PARAMS['d_model']}")
print(f"  - MAE模型: {mae_model_path}")
print(f"  - 输出目录: {results_dir}")
print("="*80)

# 加载数据
dfs = []
for ticker in ticker_list:
    temp_df = pd.read_csv(os.path.join(full_stock_dir,ticker+'.csv'),
                          usecols = ['date', 'open', 'close', 'high', 'low', 'volume', 'dopen', 'dclose', 'dhigh', 'dlow', 'dvolume', 'price'])
    temp_df['date'] = temp_df['date'].apply(lambda x:str(x))
    temp_df['date'] = pd.to_datetime(temp_df['date'])
    temp_df['label_short_term'] = temp_df['close'].pct_change(periods = prediction_len[0]).shift(periods = (-1*prediction_len[0]))
    temp_df['label_long_term'] = temp_df['close'].pct_change(periods = prediction_len[1]).shift(periods = (-1*prediction_len[1]))
    temp_df['tic'] = pd.Series([ticker]*len(temp_df))
    dfs.append(temp_df)

df = pd.concat(dfs, ignore_index=True)

df = df.sort_values(by = ['date','tic'])

# 特征工程
fe = FeatureEngineer(
    use_technical_indicator = True,
    tech_indicator_list = config.TECHNICAL_INDICATORS_LIST,
    use_ziyou_factors = True,
    use_rda_factors = True)

print(" 生成技术指标...")
df = fe.preprocess_data(df)

# 添加协方差矩阵（保持完整lookback）
df = df.sort_values(['date','tic'],ignore_index = True)
df.index = df.date.factorize()[0]

cov_list = []
return_list = []

# 保持1年lookback（252个交易日）
lookback = 252
for i in range(lookback,len(df.index.unique())):
    data_lookback = df.loc[i-lookback:i, :]
    price_lookback = data_lookback.pivot_table(
        index = 'date',
        columns = 'tic',
        values = 'close'
    )
    return_lookback = price_lookback.pct_change().dropna()
    return_list.append(return_lookback)
    covs = return_lookback.cov().values
    cov_list.append(covs)

df_cov = pd.DataFrame({'date' : df.date.unique()[lookback:], 'cov_list' : cov_list, 'return_list' : return_list})
df = df.merge(df_cov, on = 'date')
df = df.sort_values(['date','tic']).reset_index(drop=True)

# 数据标准化 - 包含所有特征因子
feature_columns = TECHNICAL_INDICATORS_LIST.copy()

print(f" 标准化特征数量: {len(feature_columns)}")
print(f"  - 全部特征: {len(TECHNICAL_INDICATORS_LIST)} (包含8个技术指标 + 3个子午因子 + 4个RDA因子)")

# 执行标准化
scaler = StandardScaler()
df_data = df[feature_columns]
df_data = df_data.replace([np.inf], INF)
df_data = df_data.replace([-np.inf], INF*(-1))
data = scaler.fit_transform(df_data.values)
df[feature_columns] = data

# 数据分割（标准三段划分）
train = data_split(df, '2011-01-17', '2018-12-28')
eval = data_split(df, '2019-01-02', '2021-12-31')
test = data_split(df, '2022-01-01', '2022-12-31')

train = train.reset_index(drop=True)
eval = eval.reset_index(drop=True)
test = test.reset_index(drop=True)

stock_dimension = len(train.tic.unique())
state_space = stock_dimension
print(f"✅ 数据加载完成:")
print(f"  - 股票维度: {stock_dimension}")
print(f"  - 状态空间: {state_space}")

# 环境参数
env_kwargs = {
    "hmax": 100,
    "initial_amount": 100000,
    "transaction_cost_pct": 0,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": feature_columns,  # 环境状态使用完整特征列表（15个）
    "temporal_feature_list": TECHNICAL_INDICATORS_LIST,
    "additional_list": config.ADDITIONAL_FEATURE,
    "action_space": stock_dimension,
    "reward_scaling": 10,
    "figure_path": results_dir + '/figures/' + version + model_name,
    "logs_path": results_dir + '/logs/' + version + model_name,
    "csv_path": results_dir + '/csv/' + version + model_name,
    "mode": 'train',
    "time_window_start": [0],  # 应该是列表格式
    "step_len": config.STAGE2_CPU_PARAMS['step_len'],
    "temporal_len": config.STAGE2_CPU_PARAMS['temporal_len'],
    "hidden_channel": config.STAGE2_CPU_PARAMS['hidden_channel'],
    "model_name": model_name[:-1],
    "short_prediction_model_path": short_prediction_model_path, # 加载阶段1训练的短期预测模型
    "long_prediction_model_path": long_prediction_model_path, # 加载阶段1训练的长期预测模型
    "enc_in": 103,  # Stage1的enc_in (covariates + technical_indicators)
    "dec_in": 103,
    "arch_type": arch_type,  # 传递架构类型
    "d_model": 64,  # Stage1的模型参数
    "n_heads": 2,
    "e_layers": 2,
    "d_layers": 1,
    "d_ff": 128,
    "num_experts": 4,
    "top_k": 2,
    "n_kv_heads": None if arch_type in ['base', 'MHA_RoPE_MoE', 'MQA_RoPE_MoE'] else 1  # GQA使用n_kv_heads=1
}

# 创建不同的环境配置
env_kwargs_eval = env_kwargs.copy()
env_kwargs_eval['mode'] = 'eval'

env_kwargs_test = env_kwargs.copy()
env_kwargs_test['mode'] = 'test'

# 创建环境
ck_dir = os.path.join(trained_model_dir, version[:-1], model_name[:-1])
log_dir = os.path.join(results_dir, version[:-1], model_name[:-1])

os.makedirs(log_dir, exist_ok = True)
os.makedirs(ck_dir, exist_ok = True)

print("  初始化环境...")
eval_trade_gym = Env(df = eval, **env_kwargs_eval)
env_eval, _ = eval_trade_gym.get_sb_env()
env_eval_sac = VecMonitor(env_eval, log_dir + '_eval')

test_trade_gym = Env(df = test, **env_kwargs_test)
env_test, _ = test_trade_gym.get_sb_env()
test_eval_sac = VecMonitor(env_test, log_dir + '_test')

e_train_gym = Env(df = train, **env_kwargs)
env_train, _ = e_train_gym.get_sb_env()
env_train_sac = VecMonitor(env_train, log_dir + '_train')

# 打印环境的observation_space，用于调试
print(f"\n🔍 环境observation_space调试信息:")
print(f"  训练环境: {e_train_gym.observation_space.shape}")
print(f"  展平后维度: {e_train_gym.observation_space.shape[0] * e_train_gym.observation_space.shape[1]}")
print(f"  验证环境: {eval_trade_gym.observation_space.shape}")
print(f"  测试环境: {test_trade_gym.observation_space.shape}")
print()

# 创建智能体
agent = DRLAgent(env = env_train_sac)
MAESAC_PARAMS = config.MAESAC_PARAMS.copy()
MAESAC_PARAMS["transformer_path"] = mae_model_path  # 加载阶段1.1训练的MAE模型（用于特征提取）

# 更新MAESAC_PARAMS以使用与Stage1一致的模型参数
MAESAC_PARAMS["enc_in"] = 103  # Stage1的enc_in
MAESAC_PARAMS["dec_in"] = 103
MAESAC_PARAMS["d_model"] = 64  # Stage1的模型参数
MAESAC_PARAMS["d_ff"] = 128
MAESAC_PARAMS["n_heads"] = 2
MAESAC_PARAMS["e_layers"] = 2
MAESAC_PARAMS["d_layers"] = 1
MAESAC_PARAMS["arch_type"] = arch_type  # 传递架构类型
MAESAC_PARAMS["num_experts"] = 4
MAESAC_PARAMS["top_k"] = 2
MAESAC_PARAMS["n_kv_heads"] = None if arch_type in ['base', 'MHA_RoPE_MoE', 'MQA_RoPE_MoE'] else 1

print("  创建MAE-SAC模型...")
model_sac = agent.get_model("maesac",
                           model_kwargs = MAESAC_PARAMS,
                           seed = fix_seed)
print("  开始训练...")
start = time.time()

trained_sac = agent.train_model(model = model_sac,
                                check_freq = 500,
                                log_dir = log_dir,
                                ck_dir = ck_dir,
                                eval_env = env_eval_sac,
                                total_timesteps = config.STAGE2_CPU_PARAMS['total_timesteps'])

end = time.time()
print(f"✅ 训练完成！用时: {end-start:.2f}秒")

model_path = os.path.join(trained_model_dir, version[:-1], model_name[:-1], 'best_model.zip')

if os.path.exists(model_path):
    start = time.time()
    print("\n在测试集上评估...")
    results = DRLAgent.DRL_prediction_load_from_file(
        model_name = 'maesac',
        environment = test_trade_gym,  # 使用test环境（2022年数据）
        cwd = model_path)
    end = time.time()
    print(f"✅ 测试完成！用时: {end-start:.2f}秒")
    df_root = results_dir + '/df_print/' + version + model_name
    os.makedirs(df_root, exist_ok = True)
    assets_his, df_actions = results[1], results[2]
    df_actions.to_csv(df_root + 'df_actions_test.csv')
    assets_his.to_csv(df_root + 'df_assets_test.csv')
    print(f"  结果已保存到: {df_root}")
else:
    print(f"⚠️  模型文件不存在: {model_path}")
    print("请检查训练是否成功完成")

print("="*80)
print(f" Stage2 Training Complete - Architecture: {arch_type}")
print("="*80)
