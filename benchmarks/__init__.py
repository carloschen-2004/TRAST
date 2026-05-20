"""
基准策略模块
提供常见的股票投资基准策略，用于与TRAST架构进行对比评估
"""

from .benchmark_strategies import (
    BenchmarkStrategy,
    BuyAndHoldStrategy,
    EqualWeightStrategy,
    MomentumStrategy,
    run_all_benchmarks
)

__all__ = [
    'BenchmarkStrategy',
    'BuyAndHoldStrategy',
    'EqualWeightStrategy',
    'MomentumStrategy',
    'run_all_benchmarks'
]