# DRL models from Stable Baselines 3
import numpy as np
import sys
import os
# 计算项目路径
_models_dir = os.path.dirname(os.path.abspath(__file__))
_stage2_policy_dir = os.path.dirname(_models_dir)
_project_root = os.path.dirname(_stage2_policy_dir)
_stage1_dir = os.path.join(_project_root, 'stage1_representation')
# 添加必要的路径
if _stage1_dir not in sys.path:
    sys.path.insert(0, _stage1_dir)

from stage2_policy import config
from stage2_policy.sac.MAE_SAC import SAC as SAC_MAE  
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.sac.policies import MlpPolicy, SACPolicy


MODELS = {"maesac": SAC_MAE}

MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}

# Policy 映射：字符串 -> 实际的 Policy 类
POLICIES = {
    "MlpPolicy": MlpPolicy,
    "SACPolicy": SACPolicy,
}

NOISE = {
    "normal": NormalActionNoise,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
}

# 计数器
class CheckCallback(BaseCallback):
    def __init__(self, check_freq: int, verbose: int = 1):
        super(CheckCallback, self).__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            print(self.n_calls)

# 监控奖励 自动保存
class oursTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(oursTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
    
    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              mean_reward = np.mean(y[-50:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # 如果新模型更好 保存
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path + 'model.zip')
        return True


class DRLAgent:
    """Provides implementations for DRL algorithms

    Attributes
    ----------
        env: gym environment class
            user-defined class

    Methods
    -------
        get_model()
            setup DRL algorithms
        train_model()
            train DRL algorithms in a train dataset
            and output the trained model
        DRL_prediction()
            make a prediction in a test dataset and get results
    """

    def __init__(self, env):
        self.env = env

    def get_model(
        self,
        model_name,
        policy = "MlpPolicy",
        policy_kwargs = None,
        model_kwargs = None,
        verbose = 1,
        seed = None,
    ):
        # 检查模型是否支持
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        # 将字符串 policy 转换为实际的类
        if isinstance(policy, str):
            if policy not in POLICIES:
                raise ValueError(f"Policy {policy} not found. Available: {list(POLICIES.keys())}")
            policy = POLICIES[policy]

        # 获取模型参数
        if model_kwargs is None:
            model_kwargs = MODEL_KWARGS[model_name]

        # 添加动作噪声（如果配置了）
        if "action_noise" in model_kwargs:
            n_actions = self.env.action_space.shape[-1]
            model_kwargs["action_noise"] = NOISE[model_kwargs["action_noise"]](
                mean = np.zeros(n_actions), sigma = 0.1 * np.ones(n_actions)
            )
        print(model_kwargs)

        # 创建模型
        model = MODELS[model_name](
            policy = policy,
            env = self.env,
            verbose = verbose,
            policy_kwargs = policy_kwargs,
            seed = seed,
            **model_kwargs,
        )
        return model

    def train_model(self, model, check_freq, ck_dir, log_dir, eval_env, total_timesteps = 5000, deterministic = True):
        eval_callback = EvalCallback(
                            eval_env, 
                            best_model_save_path = ck_dir, # 保存最佳模型的路径
                            log_path = log_dir, 
                            eval_freq = check_freq, 
                            n_eval_episodes = 1, 
                            deterministic = deterministic, 
                            render = False)
        callback = eval_callback

        model = model.learn(
            total_timesteps = total_timesteps,
            callback = callback
        )
        return model

    @staticmethod
    def DRL_prediction(model, environment, deterministic = True):
        test_env, test_obs = environment.get_sb_env()
        """make a prediction"""
        account_memory = []
        actions_memory = []
        test_env.reset()
        for i in range(len(environment.df.index.unique())):
            # 预测动作
            action, _ = model.predict(test_obs, deterministic = deterministic)
            # 执行动作
            test_obs, _, dones, _ = test_env.step(action)
            # 保存结果（最后一天/结束时）
            if i == (len(environment.df.index.unique()) - 2):
                account_memory = test_env.env_method(method_name = "save_asset_memory")
                actions_memory = test_env.env_method(method_name = "save_action_memory")
            if dones[0]:
                account_memory = test_env.env_method(method_name = "save_asset_memory")
                actions_memory = test_env.env_method(method_name = "save_action_memory")
                print("hit end!")
                break
            
        return account_memory[0], actions_memory[0]#, universal_results[0]

    @staticmethod
    def DRL_prediction_load_from_file(model_name, environment, cwd, deterministic = True):
        test_env, _ = environment.get_sb_env()
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")
        try:
            # load agent - 传递env参数以确保observation_space匹配
            model = MODELS[model_name].load(cwd, env=test_env)
            print("Successfully load model", cwd)
        except BaseException:
            raise ValueError("Fail to load agent!")

        # test on the testing env
        state, _ = environment.reset()  # gymnasium API: 返回 (obs, info)
        episode_returns = list()  # the cumulative_return / initial_account
        episode_total_assets = list()
        episode_total_assets.append(environment.initial_amount)
        done = False
        while not done:
            action = model.predict(state, deterministic = deterministic)[0]
            state, _, terminated, truncated, _ = environment.step(action)  # gymnasium API
            done = terminated or truncated  # 合并terminated和truncated

            total_asset = environment.end_total_asset

            episode_total_assets.append(total_asset)
            episode_return = total_asset / environment.initial_amount
            episode_returns.append(episode_return)

        print("episode_return", episode_return)
        print("Test Finished!")

        account_memory = test_env.env_method(method_name = "save_asset_memory")
        actions_memory = test_env.env_method(method_name = "save_action_memory")

        return episode_total_assets, account_memory[0], actions_memory[0]
