from typing import Any, Dict, List, Optional, Tuple, Type, Union
import gymnasium as gym
import numpy as np
import torch as th
from torch.nn import functional as F
from collections import OrderedDict

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from .off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac.policies import SACPolicy
import sys
import os
# 计算到项目根目录的路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
_stage2_dir = os.path.dirname(_current_dir)
_project_root = os.path.dirname(_stage2_dir)
_stage1_dir = os.path.join(_project_root, 'stage1_representation')
if _stage1_dir not in sys.path:
    sys.path.append(_stage1_dir)
# 导入所有架构的模型
from models.transformer import (
    Transformer_base,
    Transformer_MHA_RoPE_MoE,
    Transformer_MQA_RoPE_MoE,
    Transformer_GQA_RoPE_MoE
)
from .policy_transformer import policy_transformer_stock_atten2 as policy_transformer_attn2


class SAC(OffPolicyAlgorithm):
    """
    Soft Actor-Critic (SAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup), from the softlearning repo
    (https://github.com/rail-berkeley/softlearning/)
    and from Stable Baselines (https://github.com/hill-a/stable-baselines)
    Paper: https://arxiv.org/abs/1801.01290
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

    Note: we use double q target and not value target as discussed
    in https://github.com/hill-a/stable-baselines/issues/270

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
    :param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        enc_in = 96,
        dec_in = 96,
        c_out_construction = 96,
        d_model = 128,
        n_heads = 4,
        e_layers = 2,
        d_layers = 1,
        d_ff = 256,
        dropout = 0.05,
        transformer_device = 'cuda:0',
        transformer_path = None,
        critic_alpha = 1,
        actor_alpha = 0,
        arch_type = 'MHA_RoPE_MoE',
        num_experts = 4,
        top_k = 2,
        n_kv_heads = None,
    ):

        super(SAC, self).__init__(
            policy,
            env,
            SACPolicy,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class = replay_buffer_class,
            replay_buffer_kwargs = replay_buffer_kwargs,
            policy_kwargs = policy_kwargs,
            tensorboard_log = tensorboard_log,
            verbose = verbose,
            device = device,
            create_eval_env = create_eval_env,
            seed = seed,
            use_sde = use_sde,
            sde_sample_freq = sde_sample_freq,
            use_sde_at_warmup = use_sde_at_warmup,
            optimize_memory_usage = optimize_memory_usage,
            supported_action_spaces = (gym.spaces.Box),
        )

        self.target_entropy = target_entropy
        self.log_ent_coef = None 
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer = None

        if _init_setup_model:
            self._setup_model()

        # 如果CUDA不可用，自动切换到CPU
        if isinstance(transformer_device, str) and 'cuda' in transformer_device:
            if not th.cuda.is_available():
                transformer_device = 'cpu'
                if verbose > 0:
                    print("CUDA not available, using CPU for transformer_device")

        self.transformer_device = transformer_device

        # 根据架构类型创建 state_transformer
        # 使用 __init__ 参数而不是从 model_kwargs 获取
        if arch_type == 'base':
            self.state_transformer = Transformer_base(
                enc_in, dec_in, c_out_construction,
                d_model, n_heads, e_layers, d_layers, d_ff, dropout
            ).to(transformer_device)
        elif arch_type == 'MHA_RoPE_MoE':
            self.state_transformer = Transformer_MHA_RoPE_MoE(
                enc_in, dec_in, c_out_construction,
                d_model, n_heads, e_layers, d_layers, d_ff,
                num_experts, top_k, dropout
            ).to(transformer_device)
        elif arch_type == 'MQA_RoPE_MoE':
            self.state_transformer = Transformer_MQA_RoPE_MoE(
                enc_in, dec_in, c_out_construction,
                d_model, n_heads, e_layers, d_layers, d_ff,
                num_experts, top_k, dropout
            ).to(transformer_device)
        elif arch_type == 'GQA_RoPE_MoE':
            self.state_transformer = Transformer_GQA_RoPE_MoE(
                enc_in, dec_in, c_out_construction,
                d_model, n_heads, n_kv_heads, e_layers, d_layers, d_ff,
                num_experts, top_k, dropout
            ).to(transformer_device)
        else:
            raise ValueError(f"Unknown architecture type: {arch_type}")

        if verbose > 0:
            print(f"Created {arch_type} state_transformer")
        
        # 加载stage1预训练权重
        if transformer_path is not None:
            map_location = transformer_device
            state_dict = th.load(transformer_path, map_location = map_location)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    name = k[7:]  
                else:
                    name = k
                new_state_dict[name] = v
            # 打印前几个键进行调试
            if verbose > 0:
                print(f"Loaded {len(new_state_dict)} parameters from {transformer_path}")
                print(f"Sample keys: {list(new_state_dict.keys())[:3]}")

            # 使用 strict=False 允许部分加载
            self.state_transformer.load_state_dict(new_state_dict, strict =False)
            print("Successfully load pretrained model...", transformer_path)
        else:
            print("Successfully initialize transformer model...")
        self.transformer_optim = th.optim.Adam(self.state_transformer.parameters(), lr = learning_rate)
        self.transformer_criteria = th.nn.MSELoss()
        self.critic_alpha = critic_alpha
        self.actor_alpha = actor_alpha

        # Actor/Critic 专用的Transformer
        self.actor_transformer = policy_transformer_attn2(d_model = d_model, dropout = dropout, lr = learning_rate).to(transformer_device)
        self.critic_transformer = policy_transformer_attn2(d_model = d_model, dropout = dropout, lr = learning_rate).to(transformer_device)
        
        self.in_feat = enc_in


    def _setup_model(self) -> None:
        super(SAC, self)._setup_model()
        self._create_aliases()
        if self.target_entropy == "auto":
            self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
        else:
            self.target_entropy = float(self.target_entropy)

        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            self.log_ent_coef = th.log(th.ones(1, device = self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr = self.lr_schedule(1))
        else:
            self.ent_coef_tensor = th.tensor(float(self.ent_coef)).to(self.device)


    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target


    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        self.state_transformer.train()
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer, self.actor_transformer.optimizer, self.critic_transformer.optimizer, self.transformer_optim] 
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        transformer_losses = []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env = self._vec_normalize_env)
            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # 使用Actor 采样动作
            state, temporal_feature_short, temporal_feature_long, holding_stocks, loss_s = self._state_transfer(replay_data.observations) # [bs, num_nodes, cov_list\technial\temporal_feature(60day)\label\holding] 通过MAE Transformer进行特征提取
            actions_pi, log_prob = self.actor.action_log_prob(
                self.actor_transformer(state.detach(), temporal_feature_short, temporal_feature_long, holding_stocks)
                ) # actor 接收 actor transformer的输出，输出动作和log概率
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            # 计算Critic的目标Q值
            next_state, next_temporal_feature_short, next_temporal_feature_long, next_holding_stocks, loss_ns = self._state_transfer(replay_data.next_observations)
            with th.no_grad():
                # 下一状态的动作
                next_actions, next_log_prob = self.actor.action_log_prob(
                    self.actor_transformer(next_state, next_temporal_feature_short, next_temporal_feature_long, next_holding_stocks)
                    )
                
                # 目标Q值
                next_q_values = th.cat(
                    self.critic_target(
                        self.critic_transformer(next_state, next_temporal_feature_short, next_temporal_feature_long, next_holding_stocks), 
                        next_actions), dim = 1
                )
                next_q_values, _ = th.min(next_q_values, dim = 1, keepdim = True)
                
                # SAC核心：Q值 - 熵奖励
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                
                # TD目标
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # 当前Q值
            current_q_values = self.critic(
                self.critic_transformer(state, temporal_feature_short, temporal_feature_long, holding_stocks), 
                replay_data.actions
            )

            # Critic损失（MSE）
            critic_loss = 0.5 * sum([
                F.mse_loss(current_q, target_q_values) 
                for current_q in current_q_values
                ])
            critic_losses.append(critic_loss.item())

            # 更新Critic
            self.critic.optimizer.zero_grad()
            self.critic_transformer.optimizer.zero_grad()
            self.transformer_optim.zero_grad()
            critic_loss.backward()

            self.critic.optimizer.step()
            self.critic_transformer.optimizer.step()
            self.transformer_optim.step()

            # 计算当前策略的Q值
            alpha = 0
            q_values_pi = th.cat(
                self.critic.forward(
                    self.critic_transformer(state, temporal_feature_short, temporal_feature_long, holding_stocks).detach(),
                    actions_pi
                    ), dim = 1
            )
            min_qf_pi, _ = th.min(q_values_pi, dim = 1, keepdim = True)

            # Actor损失 SAC核心：最大化（Q值 + 熵奖励）
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean() + alpha * th.abs(th.mean(th.sum(replay_data.actions, dim = -1))-1)
            actor_losses.append(actor_loss.item())

            # 更新Actor
            self.actor.optimizer.zero_grad()
            self.actor_transformer.optimizer.zero_grad()
            actor_loss.backward()
            
            self.actor.optimizer.step()
            self.actor_transformer.optimizer.step()

            transformerloss = (loss_s + loss_ns) / 2
            transformer_losses.append(transformerloss.item())

            # 软更新目标网络
            if gradient_step % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(), 
                    self.critic_target.parameters(), 
                    self.tau
                )

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude = "tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/transformer_loss", np.mean(transformer_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))


    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "SAC",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        return super(SAC, self).learn(
            total_timesteps = total_timesteps,
            callback = callback,
            log_interval = log_interval,
            eval_env = eval_env,
            eval_freq = eval_freq,
            n_eval_episodes = n_eval_episodes,
            tb_log_name = tb_log_name,
            eval_log_path = eval_log_path,
            reset_num_timesteps = reset_num_timesteps,
        )


    def predict(self, test_obs: np.ndarray, deterministic: bool = False, state: np.ndarray = None, episode_start: bool = None) -> OffPolicyAlgorithm:
        flag = 0
        if len(test_obs.shape) == 2:
            test_obs = np.expand_dims(test_obs, axis = 0)
            flag = 1

        self.state_transformer.eval()
        with th.no_grad():
            obs = th.FloatTensor(test_obs).to(self.transformer_device)
            # 状态转换（不实用MAE掩码）
            obs_tensor, temporal_short, temporal_long, holding = self._state_transfer_predict(obs)
            # 通过 Actor Transformer
            state_tensor = self.actor_transformer(
                obs_tensor, temporal_short, temporal_long, holding
            )
            obs_array = state_tensor.detach().cpu().numpy()

        if flag:
            obs_array = obs_array.squeeze(0)
        # 调用父类的predict方法
        return super(SAC, self).predict(observation = obs_array, deterministic = deterministic)


    def _excluded_save_params(self) -> List[str]:
        return super(SAC, self)._excluded_save_params() + ["actor", "critic", "critic_target"]


    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables
    

    def _state_transfer_predict(self, x):
        batch_enc1 = x[:, :, :self.in_feat] # [cov+technical_list]
        # 直接通过Transformer
        enc_out, _, _ = self.state_transformer(batch_enc1, batch_enc1)
        hidden_channel = enc_out.shape[-1]
        # 提取时间信息和持仓信息
        temporal_feature_short = x[:, :, self.in_feat: hidden_channel + self.in_feat]
        temporal_feature_long = x[:, :, hidden_channel + self.in_feat: hidden_channel * 2 + self.in_feat]
        holding = x[:, :, -1:]

        return enc_out, temporal_feature_short, temporal_feature_long, holding


    def _state_transfer(self, x):
        """MAE掩码"""
        bs, stock_num = x.shape[0], x.shape[1]

        batch_enc1 = x[:, :, :self.in_feat] # [cov+technical_list]
        mask = th.ones_like(batch_enc1) # 创建MAE掩码
        rand_indices = th.rand(bs, stock_num).argsort(dim = -1)
        mask_indices = rand_indices[:, :int(stock_num/2)]
        batch_range = th.arange(bs)[:, None]
        mask[batch_range, mask_indices, stock_num:] = 0 
        enc_inp = mask * batch_enc1

        # 通过MAE编码器
        enc_out, _, output = self.state_transformer(enc_inp, enc_inp)
        hidden_channel = enc_out.shape[-1]

        # 计算MAE重建损失
        pred = output[batch_range, mask_indices, stock_num:]
        true = batch_enc1[batch_range, mask_indices, stock_num:]
        loss = self.transformer_criteria(pred, true)

        # 提取时间信息和持仓信息
        temporal_feature_short = x[:, :, self.in_feat: hidden_channel + self.in_feat]
        temporal_feature_long = x[:, :, hidden_channel + self.in_feat: hidden_channel * 2 + self.in_feat]
        holding = x[:, :, -1:]

        return enc_out, temporal_feature_short, temporal_feature_long, holding, loss
