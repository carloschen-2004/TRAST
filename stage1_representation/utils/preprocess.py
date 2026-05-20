import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from stockstats import StockDataFrame as Sdf

class FeatureEngineer:
    """Provides methods for preprocessing the stock price data"""

    def __init__(
        self,
        use_technical_indicator = True,
        tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
        use_ziyou_factors = True,  
        use_rda_factors = True,       
    ):
        self.use_technical_indicator = use_technical_indicator
        self.tech_indicator_list = tech_indicator_list
        self.use_ziyou_factors = use_ziyou_factors
        self.use_rda_factors = use_rda_factors

    def preprocess_data(self, df):
        """main method to do the feature engineering"""
        df = self.clean_data(df)

        # 使用统一的因子管理器
        df = FactorManager.add_all_factors(
            df,
            use_technical_indicator = self.use_technical_indicator,
            use_ziyou = self.use_ziyou_factors,
            use_rda = self.use_rda_factors,
        )

        df = df.ffill().bfill()
        return df

    def clean_data(self, data):
        """clean the raw data & deal with missing values"""
        df = data.copy()
        df = df.sort_values(["date", "tic"], ignore_index=True)
        df.index = df.date.factorize()[0]
        merged_closes = df.pivot_table(index="date", columns="tic", values="close")
        merged_closes = merged_closes.dropna(axis=1)
        tics = merged_closes.columns
        df = df[df.tic.isin(tics)]
        return df


"""因子库 - 按来源分类组织"""
def apply_factors_by_ticker(df, factor_funcs):
    df = df.copy()
    df = df.sort_values(by=["tic", "date"])
    results = []

    for tic, sub_df in df.groupby("tic"):
        sub_df = sub_df.copy()
        
        for func in factor_funcs:
            try:
                sub_df = func(sub_df)
            except Exception as e:
                print(f"Error in {func.__name__} for {tic}: {e}")
        
        results.append(sub_df)

    df = pd.concat(results, ignore_index=True)
    df = df.sort_values(by=["date", "tic"])
    return df

# ==================== (1) stockstats技术指标 ====================

class TechnicalFactors:
    """基于stockstats库的技术指标因子"""
    @staticmethod
    def add_stockstats_indicators(df, indicator_list):
        """从stockstats库添加技术指标"""
        df = df.copy()
        df = df.sort_values(by=["tic", "date"])

        results = []
        for tic, sub_df in df.groupby("tic"):
            # 保存原始索引和关键列
            sub_df = sub_df.copy().reset_index(drop=True)
            try:
                stock = Sdf.retype(sub_df.copy())
                for indicator in indicator_list:
                    try:
                        # 计算技术指标并添加到sub_df
                        sub_df[indicator] = stock[indicator].values
                    except Exception as e:
                        print(f"Error calculating {indicator} for {tic}: {e}")
                        sub_df[indicator] = np.nan

            except Exception as e:
                print(f"Error processing ticker {tic}: {e}")
                # 如果出错，确保所有指标列都存在（填充NaN）
                for indicator in indicator_list:
                    if indicator not in sub_df.columns:
                        sub_df[indicator] = np.nan

            # 确保关键列存在
            results.append(sub_df)

        df = pd.concat(results, ignore_index=True)
        df = df.sort_values(by=["date", "tic"])

        return df

# ==================== (2) 子午投资内部投研因子 ====================

class ZiyouInternalFactors:
    """子午投资内部投研因子类"""
    @staticmethod
    def add_factors(df):
        factor_funcs = [
            ZiyouInternalFactors._close_volume_cor,
            ZiyouInternalFactors._capital_flow,
            ZiyouInternalFactors._weighted_skew,
        ]
        return apply_factors_by_ticker(df, factor_funcs)
    
    @staticmethod
    def _close_volume_cor(df):
        """量价相关性"""
        close = df["close"]
        volume = df["volume"]
        cor = close.rolling(window=20).corr(volume)
        df['close_volume_cor'] = cor
        return df
    
    @staticmethod
    def _capital_flow(df):
        """资金流向"""
        close = df["close"]
        volume = df["volume"]
        ret = close.pct_change()
        direction = np.sign(ret)
        nu = (volume * direction).rolling(window=20).sum()
        de = volume.rolling(window=20).sum()
        df['capital_flow'] = nu / de
        return df
    
    @staticmethod
    def _weighted_skew(df):
        """加权偏度：成交量加权的价格偏度"""
        close = df["close"]
        volume = df["volume"]

        window = 20
        roll_close = close.rolling(window=window)
        mean = roll_close.mean()
        std = roll_close.std()
        total_volume = volume.rolling(window=window).sum()
        # 标准化价格
        centered = close - mean
        # 三阶矩（偏度分子）
        cubed = centered ** 3
        # 成交量权重
        weight = volume / total_volume
        # 加权偏度 = sum(weight * (x-mean)^3) / std^3
        nu = (weight * cubed).rolling(window=window).sum()
        de = std ** 3

        df['weighted_skew'] = nu / de
        return df
    
# ==================== (3) RDA因子 ====================

class RDAFactors:
    """RDA因子类"""

    @staticmethod
    def add_factors(df):
        factor_funcs = [
            RDAFactors._vmon,
            RDAFactors._klen,
        ]
        return apply_factors_by_ticker(df, factor_funcs)
    
    @staticmethod
    def _vmon(df):
        windows = [20, 50, 100]
        v = df["volume"]
        for window in windows:
            v_shift = v.shift(window)
            factor = (v - v_shift) / v_shift.replace(0, np.nan)
            df[f'vmon_{window}'] = factor
        return df
    
    @staticmethod
    def _klen(df):
        high = df["high"]
        low = df["low"]
        open_price = df["open"]  
        klen = (high - low) / open_price
        df['klen'] = klen.fillna(0)
        return df

# ==================== 统一的因子管理接口 ====================

class FactorManager:
    """统一的因子添加接口"""

    @staticmethod
    def add_all_factors(df, use_technical_indicator = True, use_ziyou = True, use_rda = True):
        print("="*60)
        print("因子处理流水线")
        print("="*60)

        # 分离不同类型的指标
        # stockstats只能处理8个基础指标，需要精确匹配
        stockstats_indicators = config.STOCKSTATS_INDICATORS  

        # (1) stockstats技术指标 - 传递stockstats能识别的8个指标
        if use_technical_indicator and stockstats_indicators:
            print("\n[1/3] 添加stockstats技术指标...")
            df = TechnicalFactors.add_stockstats_indicators(df, stockstats_indicators)
            print(f"  ✅ 完成！添加了 {len(stockstats_indicators)} 个stockstats指标")

        # (2) 子午投资内部因子 - 由ZiyouInternalFactors处理
        if use_ziyou:
            print("\n[2/3] 添加子午投资内部投研因子...")
            df = ZiyouInternalFactors.add_factors(df)
            print(f"  ✅ 完成！添加了 {len(config.ZIYOU_INDICATORS)} 个子午投资因子")

        # (3) RDA因子 - 由RDAFactors处理
        if use_rda:
            print("\n[3/3] 添加RDA因子...")
            df = RDAFactors.add_factors(df)
            print(f"  ✅ 完成！添加了 {len(config.RDA_INDICATORS)} 个RDA因子")

        print("\n" + "="*60)
        print(f"因子处理完成！")
        print(f"   当前数据列数: {len(df.columns)}")
        print(f"   当前数据行数: {len(df)}")
        print(f"   当前数据列: {df.columns.tolist()} ")
        print("="*60)

        return df


