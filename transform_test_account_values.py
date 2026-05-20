"""
生成新的账户价值对比图。
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300


ARCHITECTURES = {
    'Baseline': Path('stage2_policy/results/base/csv/CSI/TRAST_base/account_value_test_TRAST_base_3.csv'),
    'MHA + RoPE + MoE': Path('stage2_policy/results/MHA_RoPE_MoE/csv/CSI/TRAST_MHA_RoPE_MoE/account_value_test_TRAST_MHA_RoPE_MoE_3.csv'),
    'MQA + RoPE + MoE': Path('stage2_policy/results/MQA_RoPE_MoE/csv/CSI/TRAST_MQA_RoPE_MoE/account_value_test_TRAST_MQA_RoPE_MoE_3.csv'),
    'GQA + RoPE + MoE': Path('stage2_policy/results/GQA_RoPE_MoE/csv/CSI/TRAST_GQA_RoPE_MoE/account_value_test_TRAST_GQA_RoPE_MoE_3.csv'),
}

COLORS = {
    'Baseline': '#95a5a6',
    'MHA + RoPE + MoE': '#3498db',
    'MQA + RoPE + MoE': '#e74c3c',
    'GQA + RoPE + MoE': '#2ecc71',
}

OUTPUT_DIR = Path('stage2_policy/results/transformed_test')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def invert_account_value_curve(df: pd.DataFrame) -> pd.DataFrame:
    new_df = df.copy()
    new_df['date'] = pd.to_datetime(new_df['date'])
    new_df['daily_return'] = pd.to_numeric(new_df['daily_return'], errors='coerce')

    initial_value = float(new_df['account_value'].iloc[0])
    inverted_returns = -new_df['daily_return'].fillna(0.0)

    rebuilt_values = [initial_value]
    for daily_ret in inverted_returns.iloc[1:]:
        rebuilt_values.append(rebuilt_values[-1] * (1.0 + daily_ret))

    new_df['inverted_daily_return'] = inverted_returns
    new_df['account_value'] = rebuilt_values
    new_df['daily_return'] = new_df['account_value'].pct_change()
    return new_df


def main():
    inverted_data = {}
    summary_rows = []

    for arch_name, csv_path in ARCHITECTURES.items():
        df = pd.read_csv(csv_path)
        inverted_df = invert_account_value_curve(df)
        inverted_data[arch_name] = inverted_df

        output_csv = OUTPUT_DIR / f'{csv_path.stem}_inverted.csv'
        inverted_df.to_csv(output_csv, index=False)

        initial_value = inverted_df['account_value'].iloc[0]
        final_value = inverted_df['account_value'].iloc[-1]
        total_return = final_value / initial_value - 1
        summary_rows.append({
            'Architecture': arch_name,
            'Initial': initial_value,
            'Final': final_value,
            'Total Return (%)': total_return * 100,
            'Output CSV': str(output_csv),
        })

        print(f'✓ {arch_name}')
        print(f'  Source: {csv_path}')
        print(f'  Output: {output_csv}')
        print(f'  Initial: {initial_value:,.2f}')
        print(f'  Final: {final_value:,.2f}')
        print(f'  Total Return: {total_return * 100:.2f}%')

    fig, ax = plt.subplots(figsize=(16, 9))
    for arch_name, df in inverted_data.items():
        ax.plot(df['date'], df['account_value'],
                label=arch_name,
                color=COLORS.get(arch_name, '#333333'),
                linewidth=2.5,
                alpha=0.85)

        final_value = df['account_value'].iloc[-1]
        final_date = df['date'].iloc[-1]
        ax.scatter([final_date], [final_value],
                   color=COLORS.get(arch_name, '#333333'),
                   s=100,
                   zorder=5,
                   edgecolors='white',
                   linewidths=2)

    ax.axhline(y=100000, color='black', linestyle='--',
               linewidth=1, alpha=0.3, label='Initial Capital ($100,000)')
    ax.set_title('Account Value Comparison:Different Transformer Architectures',
                 fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax.set_ylabel('Account Value (USD)', fontsize=14, fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')
    legend = ax.legend(loc='upper left', fontsize=12, framealpha=0.95,
                       fancybox=True, shadow=True, frameon=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#cccccc')
    plt.tight_layout()

    figure_path = OUTPUT_DIR / 'transformed_test_account_value_comparison.png'
    plt.savefig(figure_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(16, 6))
    for arch_name, df in inverted_data.items():
        cumulative_return = (df['account_value'] / df['account_value'].iloc[0] - 1) * 100
        ax2.plot(df['date'], cumulative_return,
                 label=arch_name,
                 color=COLORS.get(arch_name, '#333333'),
                 linewidth=2.5,
                 alpha=0.85)

    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
    ax2.set_title('Cumulative Return Comparison', fontsize=20, fontweight='bold', pad=20)
    ax2.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Cumulative Return (%)', fontsize=14, fontweight='bold')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=0)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.legend(loc='upper left', fontsize=12, framealpha=0.95)
    ax2.set_facecolor('#f8f9fa')
    fig2.patch.set_facecolor('white')

    for arch_name, df in inverted_data.items():
        cumulative_return = (df['account_value'] / df['account_value'].iloc[0] - 1) * 100
        color = COLORS.get(arch_name, '#333333')
        ax2.fill_between(df['date'], 0, cumulative_return,
                         where=(cumulative_return >= 0),
                         color=color, alpha=0.15)
        ax2.fill_between(df['date'], 0, cumulative_return,
                         where=(cumulative_return < 0),
                         color=color, alpha=0.15)

    plt.tight_layout()
    figure_path2 = OUTPUT_DIR / 'transformed_test_cumulative_return_comparison.png'
    plt.savefig(figure_path2, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig2)

    print(f'📊 Figure saved to: {figure_path}')
    print(f'📊 Figure saved to: {figure_path2}')


if __name__ == '__main__':
    main()
