"""
绘制不同架构在 stage2 训练后的账户价值对比图
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

# 定义架构名称和对应的文件路径
architectures = {
    'Baseline': 'stage2_policy/results/base/csv/CSI/TRAST_base/account_value_eval_TRAST_base_2.csv',
    'MHA + RoPE + MoE': 'stage2_policy/results/MHA_RoPE_MoE/csv/CSI/TRAST_MHA_RoPE_MoE/account_value_eval_TRAST_MHA_RoPE_MoE_2.csv',
    'MQA + RoPE + MoE': 'stage2_policy/results/MQA_RoPE_MoE/csv/CSI/TRAST_MQA_RoPE_MoE/account_value_eval_TRAST_MQA_RoPE_MoE_2.csv',
    'GQA + RoPE + MoE': 'stage2_policy/results/GQA_RoPE_MoE/csv/CSI/TRAST_GQA_RoPE_MoE/account_value_eval_TRAST_GQA_RoPE_MoE_2.csv'
}

# 定义颜色方案（专业配色）
colors = {
    'Baseline': '#95a5a6',      # 灰色
    'MHA + RoPE + MoE': '#3498db',  # 蓝色
    'MQA + RoPE + MoE': '#e74c3c',  # 红色
    'GQA + RoPE + MoE': '#2ecc71'   # 绿色
}

TRADING_DAYS_PER_YEAR = 252

def calculate_performance_metrics(df):
    account_value = df['account_value'].astype(float)
    daily_returns = account_value.pct_change().dropna()

    initial_value = account_value.iloc[0]
    final_value = account_value.iloc[-1]
    total_return = final_value / initial_value - 1

    num_periods = len(account_value) - 1
    if num_periods > 0:
        annualized_return = (final_value / initial_value) ** (TRADING_DAYS_PER_YEAR / num_periods) - 1
    else:
        annualized_return = np.nan

    daily_volatility = daily_returns.std()
    annualized_volatility = daily_volatility * np.sqrt(TRADING_DAYS_PER_YEAR) if not pd.isna(daily_volatility) else np.nan
    if daily_returns.empty or pd.isna(daily_volatility) or np.isclose(daily_volatility, 0.0):
        sharpe_ratio = np.nan
    else:
        sharpe_ratio = np.sqrt(TRADING_DAYS_PER_YEAR) * daily_returns.mean() / daily_volatility

    rolling_peak = account_value.cummax()
    drawdown = account_value / rolling_peak - 1
    max_drawdown = drawdown.min()

    if pd.isna(max_drawdown) or np.isclose(max_drawdown, 0.0):
        calmar_ratio = np.nan
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

# 读取所有数据
data = {}
performance_summary = []
for arch_name, file_path in architectures.items():
    try:
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        data[arch_name] = df
        metrics = calculate_performance_metrics(df)
        performance_summary.append({
            'Architecture': arch_name,
            **metrics,
        })
        print(f"✓ Successfully loaded {arch_name}: {len(df)} trading days")
        print(f"  Initial: ${metrics['Initial']:,.2f}")
        print(f"  Final: ${metrics['Final']:,.2f}")
        print(f"  Total Return: {metrics['Total Return'] * 100:.2f}%")
        print(f"  Annualized Return: {metrics['Annualized Return'] * 100:.2f}%")
        print(f"  Sharpe Ratio: {metrics['Sharpe Ratio']:.4f}")
        print(f"  Max Drawdown: {metrics['Max Drawdown'] * 100:.2f}%")
        print(f"  Calmar Ratio: {metrics['Calmar Ratio']:.4f}")
    except Exception as e:
        print(f"✗ Error loading {arch_name}: {e}")

# 创建图表
fig, ax = plt.subplots(figsize=(16, 9))

# 绘制每个架构的曲线
for arch_name, df in data.items():
    ax.plot(df['date'], df['account_value'],
            label=arch_name,
            color=colors.get(arch_name, '#333333'),
            linewidth=2.5,
            alpha=0.85)

    # 添加终点标记
    final_value = df['account_value'].iloc[-1]
    final_date = df['date'].iloc[-1]
    ax.scatter([final_date], [final_value],
              color=colors.get(arch_name, '#333333'),
              s=100,
              zorder=5,
              edgecolors='white',
              linewidths=2)

# 添加初始值参考线
initial_value = 100000
ax.axhline(y=initial_value, color='black', linestyle='--',
          linewidth=1, alpha=0.3, label='Initial Capital ($100,000)')

# 设置标题和标签
ax.set_title('Account Value Comparison: Different Transformer Architectures',
             fontsize=20, fontweight='bold', pad=20)
ax.set_xlabel('Date', fontsize=14, fontweight='bold')
ax.set_ylabel('Account Value (USD)', fontsize=14, fontweight='bold')

# 设置Y轴格式
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

# 设置X轴日期格式
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=0)  # 日期不倾斜

# 添加网格
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

# 设置图例
legend = ax.legend(loc='upper left', fontsize=12, framealpha=0.95,
                  fancybox=True, shadow=True, frameon=True)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('#cccccc')

# 设置背景色
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')

# 调整布局
plt.tight_layout()

# 保存图片
output_path = 'stage2_policy/results/account_value_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n📊 图表已保存到: {output_path}")

# 创建统计摘要表格
print("\n" + "="*80)
print("PERFORMANCE SUMMARY")
print("="*80)
print(
    f"{'Architecture':<20} {'Total Ret':>10} {'Annual Ret':>11} "
    f"{'Ann Vol':>10} {'Sharpe':>9} {'Max DD':>10} {'Calmar':>9}"
)
print("-"*80)

for item in performance_summary:
    print(
        f"{item['Architecture']:<20} "
        f"{item['Total Return'] * 100:>9.2f}% "
        f"{item['Annualized Return'] * 100:>10.2f}% "
        f"{item['Annualized Volatility'] * 100:>9.2f}% "
        f"{item['Sharpe Ratio']:>9.4f} "
        f"{item['Max Drawdown'] * 100:>9.2f}% "
        f"{item['Calmar Ratio']:>9.4f}"
    )

print("="*80)

summary_df = pd.DataFrame(performance_summary)
summary_df['Total Return (%)'] = summary_df['Total Return'] * 100
summary_df['Annualized Return (%)'] = summary_df['Annualized Return'] * 100
summary_df['Annualized Volatility (%)'] = summary_df['Annualized Volatility'] * 100
summary_df['Max Drawdown (%)'] = summary_df['Max Drawdown'] * 100
summary_df = summary_df[
    [
        'Architecture',
        'Initial',
        'Final',
        'Max Account Value',
        'Total Return (%)',
        'Annualized Return (%)',
        'Annualized Volatility (%)',
        'Sharpe Ratio',
        'Max Drawdown (%)',
        'Calmar Ratio',
    ]
]
summary_output_path = 'stage2_policy/results/performance_summary.csv'
summary_df.to_csv(summary_output_path, index=False)
print(f"\n📄 绩效指标汇总已保存到: {summary_output_path}")

# 显示图表
plt.close(fig)

# 创建第二个图：收益率对比
fig2, ax2 = plt.subplots(figsize=(16, 6))

for arch_name, df in data.items():
    cumulative_return = (df['account_value'] / df['account_value'].iloc[0] - 1) * 100
    ax2.plot(df['date'], cumulative_return,
            label=arch_name,
            color=colors.get(arch_name, '#333333'),
            linewidth=2.5,
            alpha=0.85)

ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
ax2.set_title('Cumulative Return Comparison', fontsize=20, fontweight='bold', pad=20)
ax2.set_xlabel('Date', fontsize=14, fontweight='bold')
ax2.set_ylabel('Cumulative Return (%)', fontsize=14, fontweight='bold')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=0)  # 日期不倾斜
ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax2.legend(loc='upper left', fontsize=12, framealpha=0.95)
ax2.set_facecolor('#f8f9fa')
fig2.patch.set_facecolor('white')

# 添加收益率区域填充
for arch_name, df in data.items():
    cumulative_return = (df['account_value'] / df['account_value'].iloc[0] - 1) * 100
    color = colors.get(arch_name, '#333333')
    ax2.fill_between(df['date'], 0, cumulative_return,
                    where=(cumulative_return >= 0),
                    color=color, alpha=0.15)
    ax2.fill_between(df['date'], 0, cumulative_return,
                    where=(cumulative_return < 0),
                    color=color, alpha=0.15)

plt.tight_layout()
output_path2 = 'stage2_policy/results/cumulative_return_comparison.png'
plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n📊 收益率对比图已保存到: {output_path2}")
plt.close(fig2)

# 创建第三个图：绩效指标柱状图
metric_configs = [
    ('Total Return (%)', 'Total Return (%)', '{:.1f}%'),
    ('Annualized Return (%)', 'Annualized Return (%)', '{:.1f}%'),
    ('Annualized Volatility (%)', 'Annualized Volatility (%)', '{:.1f}%'),
    ('Sharpe Ratio', 'Sharpe Ratio', '{:.2f}'),
    ('Max Drawdown (%)', 'Max Drawdown (%)', '{:.1f}%'),
    ('Calmar Ratio', 'Calmar Ratio', '{:.2f}'),
]

fig3, axes = plt.subplots(2, 3, figsize=(17, 9))
axes = axes.flatten()
bar_colors = [colors.get(arch, '#333333') for arch in summary_df['Architecture']]
architecture_labels = summary_df['Architecture'].tolist()

for idx, (column_name, title, value_format) in enumerate(metric_configs):
    ax = axes[idx]
    metric_values = summary_df[column_name].values
    bars = ax.bar(
        architecture_labels,
        metric_values,
        color=bar_colors,
        width=0.62,
        edgecolor='white',
        linewidth=1.2,
        alpha=0.92,
    )

    ax.set_title(title, fontsize=15, fontweight='bold', pad=10)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.6, alpha=0.22)
    ax.set_axisbelow(True)
    ax.set_facecolor('#fbfbfb')
    ax.tick_params(axis='x', rotation=12, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#d0d7de')
    ax.spines['bottom'].set_color('#d0d7de')

    if column_name == 'Max Drawdown (%)':
        ax.axhline(y=0, color='#7f8c8d', linewidth=1.0, alpha=0.6)

    value_range = np.nanmax(metric_values) - np.nanmin(metric_values)
    offset = max(value_range * 0.04, 0.03 if 'Ratio' in column_name else 1.0)
    for bar, value in zip(bars, metric_values):
        x = bar.get_x() + bar.get_width() / 2
        if value >= 0:
            y = value + offset
            va = 'bottom'
        else:
            y = value - offset
            va = 'top'
        ax.text(
            x,
            y,
            value_format.format(value),
            ha='center',
            va=va,
            fontsize=10,
            color='#2c3e50',
            fontweight='semibold',
        )

fig3.suptitle(
    'Performance Metrics Comparison Across Transformer Architectures',
    fontsize=20,
    fontweight='bold',
    y=0.98
)
fig3.patch.set_facecolor('white')
plt.tight_layout(rect=[0, 0, 1, 0.96])

output_path3 = 'stage2_policy/results/performance_metrics_comparison.png'
plt.savefig(output_path3, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n📊 绩效指标柱状图已保存到: {output_path3}")
plt.close(fig3)
