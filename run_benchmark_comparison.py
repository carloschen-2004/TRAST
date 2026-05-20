"""
运行基准策略并与TRAST架构进行对比
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import sys
import argparse

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('..')

from benchmarks.benchmark_strategies import (
    BuyAndHoldStrategy,
    EqualWeightStrategy,
    MomentumStrategy,
    run_all_benchmarks
)
from stage1_representation.config import USE_CSI_300_TICKET

# 设置字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

# 常量
TRADING_DAYS_PER_YEAR = 252
INITIAL_AMOUNT = 100000


def calculate_performance_metrics(account_value: np.array) -> dict:
    """计算绩效指标"""
    if len(account_value) < 2:
        return {}

    daily_returns = pd.Series(account_value).pct_change().dropna()

    initial_value = account_value[0]
    final_value = account_value[-1]
    total_return = final_value / initial_value - 1

    num_periods = len(account_value) - 1
    annualized_return = (final_value / initial_value) ** (TRADING_DAYS_PER_YEAR / num_periods) - 1

    daily_volatility = daily_returns.std()
    annualized_volatility = daily_volatility * np.sqrt(TRADING_DAYS_PER_YEAR)

    if daily_volatility == 0 or pd.isna(daily_volatility):
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = np.sqrt(TRADING_DAYS_PER_YEAR) * daily_returns.mean() / daily_volatility

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


def load_trast_results(arch_type: str, mode: str = 'test') -> dict:
    """加载TRAST结果"""
    result_path = f'stage2_policy/results/{arch_type}/csv/CSI/TRAST_{arch_type}/account_value_{mode}_TRAST_{arch_type}_2.csv'

    if not os.path.exists(result_path):
        print(f"Warning: TRAST result not found: {result_path}")
        return None

    df = pd.read_csv(result_path)
    df['date'] = pd.to_datetime(df['date'])

    return df


def prepare_benchmark_data(df: pd.DataFrame, mode: str = 'test') -> pd.DataFrame:
    """准备基准策略数据"""
    if mode == 'train':
        df = df[(df.date >= '2011-01-17') & (df.date <= '2018-12-28')]
    elif mode == 'eval':
        df = df[(df.date >= '2019-01-02') & (df.date <= '2021-12-31')]
    elif mode == 'test':
        df = df[(df.date >= '2022-01-01') & (df.date <= '2022-12-31')]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description='Run Benchmarks vs TRAST')
    parser.add_argument('--mode', type=str, default='test',
                       choices=['train', 'eval', 'test'],
                       help='Data split mode')
    parser.add_argument('--arch_type', type=str, default='MHA_RoPE_MoE',
                       choices=['base', 'MHA_RoPE_MoE', 'MQA_RoPE_MoE', 'GQA_RoPE_MoE'],
                       help='TRAST architecture type')
    parser.add_argument('--data_dir', type=str, default='data/CSI/',
                       help='Data directory')
    args = parser.parse_args()

    print("="*80)
    print(f" 基准策略 vs TRAST对比 - Mode: {args.mode}, Arch: {args.arch_type}")
    print("="*80)

    # 创建结果目录
    benchmark_dir = f'benchmarks/results/{args.mode}_{args.arch_type}'
    os.makedirs(benchmark_dir, exist_ok=True)

    # 加载原始数据
    print(f"\n加载股票数据...")
    dfs = []
    for ticker in USE_CSI_300_TICKET:
        data_path = os.path.join(args.data_dir, f'{ticker}.csv')
        if os.path.exists(data_path):
            temp_df = pd.read_csv(data_path,
                                  usecols=['date', 'open', 'close', 'high', 'low', 'volume', 'price'])
            temp_df['date'] = pd.to_datetime(temp_df['date'])
            temp_df['tic'] = ticker
            dfs.append(temp_df)

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(by=['date', 'tic']).reset_index(drop=True)

    # 准备数据
    df = prepare_benchmark_data(df, args.mode)
    stock_dim = len(df.tic.unique())

    print(f"  股票数量: {stock_dim}")
    print(f"  交易天数: {len(df.date.unique())}")
    print(f"  时间范围: {df.date.min()} 到 {df.date.max()}")

    # 运行基准策略
    print(f"\n运行基准策略...")
    benchmarks = {
        'Buy & Hold': BuyAndHoldStrategy(),
        'Equal Weight (5d)': EqualWeightStrategy(rebalance_freq=5),
        'Momentum (20d)': MomentumStrategy(lookback_days=20, top_n=stock_dim//4),
    }

    benchmark_data = {}
    benchmark_metrics = []

    for name, strategy in benchmarks.items():
        print(f"\n  运行 {name}...")
        result = strategy.run(df, initial_amount=INITIAL_AMOUNT, stock_dim=stock_dim)

        metrics = strategy.get_performance_metrics()
        benchmark_data[name] = {
            'dates': strategy.date_memory,
            'values': strategy.asset_memory,
            'actions': strategy.actions_memory
        }

        benchmark_metrics.append({
            'Strategy': name,
            **metrics
        })

        # 保存基准策略结果
        save_dir = os.path.join(benchmark_dir, name.replace(' ', '_'))
        os.makedirs(save_dir, exist_ok=True)

        df_result = pd.DataFrame({
            'date': strategy.date_memory,
            'account_value': strategy.asset_memory
        })
        df_result.to_csv(os.path.join(save_dir, 'account_value.csv'), index=False)

        print(f"    Total Return: {metrics['Total Return'] * 100:.2f}%")
        print(f"    Sharpe Ratio: {metrics['Sharpe Ratio']:.4f}")
        print(f"    Max Drawdown: {metrics['Max Drawdown'] * 100:.2f}%")

    # 加载TRAST结果
    print(f"\n加载TRAST结果...")
    trast_result = load_trast_results(args.arch_type, args.mode)

    if trast_result is not None:
        trast_data = {
            'dates': trast_result['date'].tolist(),
            'values': trast_result['account_value'].tolist()
        }

        trast_metrics = calculate_performance_metrics(trast_result['account_value'].values)
        trast_metrics['Strategy'] = f'TRAST ({args.arch_type})'

        print(f"  TRAST ({args.arch_type}):")
        print(f"    Total Return: {trast_metrics['Total Return'] * 100:.2f}%")
        print(f"    Sharpe Ratio: {trast_metrics['Sharpe Ratio']:.4f}")
        print(f"    Max Drawdown: {trast_metrics['Max Drawdown'] * 100:.2f}%")
    else:
        trast_data = None
        trast_metrics = None

    # 创建对比图表
    print(f"\n生成对比图表...")

    # 颜色方案
    colors = {
        'Buy & Hold': '#95a5a6',           # 灰色
        'Equal Weight (5d)': '#e74c3c',    # 红色
        'Momentum (20d)': '#3498db',       # 蓝色
        f'TRAST ({args.arch_type})': '#2ecc71'  # 绿色
    }

    # 图1：账户价值对比
    fig, ax = plt.subplots(figsize=(16, 9))

    # 绘制基准策略
    for name, data in benchmark_data.items():
        ax.plot(data['dates'], data['values'],
                label=name,
                color=colors.get(name, '#333333'),
                linewidth=2.0,
                alpha=0.7)

    # 绘制TRAST
    if trast_data is not None:
        ax.plot(trast_data['dates'], trast_data['values'],
                label=f'TRAST ({args.arch_type})',
                color=colors.get(f'TRAST ({args.arch_type})', '#2ecc71'),
                linewidth=3.0,
                alpha=1.0)

    # 初始值参考线
    ax.axhline(y=INITIAL_AMOUNT, color='black', linestyle='--',
              linewidth=1, alpha=0.3, label=f'Initial Capital (${INITIAL_AMOUNT:,})')

    # 设置标题和标签
    ax.set_title(f'Benchmark vs TRAST Comparison - {args.mode.upper()} Set',
                fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax.set_ylabel('Account Value (USD)', fontsize=14, fontweight='bold')

    # 格式化Y轴
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

    # 格式化X轴
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=0)

    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # 设置图例
    legend = ax.legend(loc='upper left', fontsize=11, framealpha=0.95,
                      fancybox=True, shadow=True, frameon=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#cccccc')

    # 设置背景
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')

    plt.tight_layout()
    output_path = os.path.join(benchmark_dir, 'benchmark_vs_trast_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  对比图已保存: {output_path}")
    plt.close(fig)

    # 图2：累积收益率对比
    fig2, ax2 = plt.subplots(figsize=(16, 6))

    for name, data in benchmark_data.items():
        cumulative_return = (np.array(data['values']) / data['values'][0] - 1) * 100
        ax2.plot(data['dates'], cumulative_return,
                label=name,
                color=colors.get(name, '#333333'),
                linewidth=2.0,
                alpha=0.7)

    if trast_data is not None:
        trast_cumulative_return = (np.array(trast_data['values']) / trast_data['values'][0] - 1) * 100
        ax2.plot(trast_data['dates'], trast_cumulative_return,
                label=f'TRAST ({args.arch_type})',
                color=colors.get(f'TRAST ({args.arch_type})', '#2ecc71'),
                linewidth=3.0,
                alpha=1.0)

    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
    ax2.set_title('Cumulative Return Comparison - Benchmarks vs TRAST',
                fontsize=20, fontweight='bold', pad=20)
    ax2.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Cumulative Return (%)', fontsize=14, fontweight='bold')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=0)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax2.set_facecolor('#f8f9fa')
    fig2.patch.set_facecolor('white')

    # 添加收益率区域填充
    for name, data in benchmark_data.items():
        cumulative_return = (np.array(data['values']) / data['values'][0] - 1) * 100
        color = colors.get(name, '#333333')
        ax2.fill_between(data['dates'], 0, cumulative_return,
                        where=(cumulative_return >= 0),
                        color=color, alpha=0.1)
        ax2.fill_between(data['dates'], 0, cumulative_return,
                        where=(cumulative_return < 0),
                        color=color, alpha=0.1)

    if trast_data is not None:
        trast_cumulative_return = (np.array(trast_data['values']) / trast_data['values'][0] - 1) * 100
        color = colors.get(f'TRAST ({args.arch_type})', '#2ecc71')
        ax2.fill_between(trast_data['dates'], 0, trast_cumulative_return,
                        where=(trast_cumulative_return >= 0),
                        color=color, alpha=0.2)
        ax2.fill_between(trast_data['dates'], 0, trast_cumulative_return,
                        where=(trast_cumulative_return < 0),
                        color=color, alpha=0.2)

    plt.tight_layout()
    output_path2 = os.path.join(benchmark_dir, 'cumulative_return_comparison.png')
    plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  收益率对比图已保存: {output_path2}")
    plt.close(fig2)

    # 生成绩效汇总表格
    print(f"\n生成绩效汇总...")
    all_metrics = benchmark_metrics.copy()

    if trast_metrics is not None:
        all_metrics.append(trast_metrics)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df['Total Return (%)'] = metrics_df['Total Return'] * 100
    metrics_df['Annualized Return (%)'] = metrics_df['Annualized Return'] * 100
    metrics_df['Annualized Volatility (%)'] = metrics_df['Annualized Volatility'] * 100
    metrics_df['Max Drawdown (%)'] = metrics_df['Max Drawdown'] * 100

    # 按总收益率排序
    metrics_df = metrics_df.sort_values('Total Return', ascending=False).reset_index(drop=True)

    # 选择要显示的列
    display_cols = [
        'Strategy', 'Initial', 'Final', 'Total Return (%)',
        'Annualized Return (%)', 'Sharpe Ratio',
        'Max Drawdown (%)', 'Calmar Ratio'
    ]

    display_df = metrics_df[display_cols].copy()

    # 保存到CSV
    summary_path = os.path.join(benchmark_dir, 'performance_summary.csv')
    display_df.to_csv(summary_path, index=False)
    print(f"  绩效汇总已保存: {summary_path}")

    # 打印表格
    print("\n" + "="*100)
    print("PERFORMANCE SUMMARY: Benchmarks vs TRAST")
    print("="*100)
    print(
        f"{'Strategy':<25} {'Total Ret':>12} {'Annual Ret':>13} "
        f"{'Sharpe':>10} {'Max DD':>11} {'Calmar':>10}"
    )
    print("-"*100)

    for _, row in display_df.iterrows():
        print(
            f"{row['Strategy']:<25} "
            f"{row['Total Return (%)']:>11.2f}% "
            f"{row['Annualized Return (%)']:>12.2f}% "
            f"{row['Sharpe Ratio']:>10.4f} "
            f"{row['Max Drawdown (%)']:>10.2f}% "
            f"{row['Calmar Ratio']:>10.4f}"
        )

    print("="*100)

    # 绘制绩效指标柱状图
    metric_configs = [
        ('Total Return (%)', 'Total Return (%)', '{:.1f}%'),
        ('Sharpe Ratio', 'Sharpe Ratio', '{:.2f}'),
        ('Max Drawdown (%)', 'Max Drawdown (%)', '{:.1f}%'),
    ]

    fig3, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (column_name, title, value_format) in enumerate(metric_configs):
        ax = axes[idx]
        metric_values = metrics_df[column_name].values
        strategy_names = metrics_df['Strategy'].tolist()

        bar_colors = [colors.get(name, '#333333') for name in strategy_names]

        bars = ax.bar(strategy_names, metric_values,
                     color=bar_colors, width=0.6,
                     edgecolor='white', linewidth=1.5, alpha=0.85)

        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.grid(True, axis='y', linestyle='--', linewidth=0.6, alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_facecolor('#fbfbfb')

        ax.tick_params(axis='x', rotation=15, labelsize=9)
        ax.tick_params(axis='y', labelsize=9)

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_color('#d0d7de')
        ax.spines['bottom'].set_color('#d0d7de')

        # 添加数值标签
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            if value >= 0:
                va = 'bottom'
                y_pos = height + max(metric_values) * 0.02
            else:
                va = 'top'
                y_pos = height - max(metric_values) * 0.02

            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                   value_format.format(value),
                   ha='center', va=va, fontsize=8, fontweight='semibold')

    fig3.suptitle(f'Performance Metrics Comparison - {args.mode.upper()} Set',
                 fontsize=16, fontweight='bold', y=1.02)
    fig3.patch.set_facecolor('white')
    plt.tight_layout()

    output_path3 = os.path.join(benchmark_dir, 'performance_metrics_comparison.png')
    plt.savefig(output_path3, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  绩效指标对比图已保存: {output_path3}")
    plt.close(fig3)

    print(f"\n✅ 所有对比完成！结果保存在: {benchmark_dir}")


if __name__ == '__main__':
    main()