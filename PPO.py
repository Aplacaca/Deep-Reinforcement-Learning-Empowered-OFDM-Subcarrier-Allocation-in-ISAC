import argparse
import os
import random
import time
from distutils.util import strtobool
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions import MultivariateNormal
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
import pickle
from utils import MultiCategoricalDistribution
from torch.utils.tensorboard import SummaryWriter

from env.ofdm_env import ISAC_BS
import pdb

ACTION_DIM = 4 # N_s
CATEGORICAL_DIM = 8 # N_c
STATE_DIM = 2*CATEGORICAL_DIM + ACTION_DIM

def parse_args(args = None):
    # fmt: off
    parser = argparse.ArgumentParser()
    # Env Config
    parser.add_argument('--subcarrier_number', type=int, default=8, help='subcarrier_number')
    parser.add_argument('--cu_number', type=int, default=8, help='cu_number')
    parser.add_argument('--su_number', type=int, default=2, help='su_number')
    parser.add_argument('--action-dim', type=int, default=2, help='action-dim')
    parser.add_argument('--categorical-dim', type=int, default=3, help='categorical-dim of each action')
    parser.add_argument('--state-dim', type=int, default=18, help='DIM OF STATE')
    
    # Exiperiment Args
    parser.add_argument('--daytime', type=str, default=datetime.datetime.now().strftime('TD_%Y-%m-%d-%H-%M-%S'),
        help='the time of this experiment')
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
        help='the name of this experiment')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=777,
        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=600000,
        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, cuda will be enabled by default')

    # Algorithm specific arguments
    parser.add_argument('--num-steps', type=int, default=2048,
        help='the number of steps to run in each environment per policy rollout')
    parser.add_argument('--anneal-lr', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument('--gae', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='Use GAE for advantage computation')
    parser.add_argument('--gamma', type=float, default=0.99,
        help='the discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
        help='the lambda for the general advantage estimation')
    parser.add_argument('--num-minibatches', type=int, default=32,
        help='the number of mini-batches')
    parser.add_argument('--update-epochs', type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument('--norm-adv', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help="Toggles advantages normalization")
    parser.add_argument('--clip-coef', type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument('--clip-vloss', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='Toggles wheter or not to use a clipped loss for the value function, as per the paper.')
    parser.add_argument('--ent-coef', type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument('--vf-coef', type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
        help='the maximum norm for the gradient clipping')
    parser.add_argument('--target-kl', type=float, default=None,
        help='the target KL divergence threshold')
    
    if args is None:
        # 从命令行读取参数
        parsed_args = parser.parse_args()
    else:
        # 使用 parse_known_args 传递参数列表
        parsed_args, _ = parser.parse_known_args(args)
    args = parsed_args
    args.batch_size = int(1 * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, state_dim = STATE_DIM, action_dim = ACTION_DIM, cate_dim = CATEGORICAL_DIM):
        super(Agent, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.cate_dim = cate_dim
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=0.1),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim*cate_dim), std=0.01),
            # nn.ReLU(), # 4.21 W SIGMOID
            nn.Sigmoid(), # 4.21 W SIGMOID
            # nn.Softmax(dim=-1), # 4.21 W SIGMOID
        )
    
    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        if torch.any(torch.isnan(action_mean)):
            print("caught nan")
            pdb.set_trace()
        # probs = MultiCategoricalDistribution(action_dims=[8]*ACTION_DIM+[2]*ACTION_DIM)
        probs = MultiCategoricalDistribution(action_dims=[self.cate_dim]*self.action_dim)
        probs.actions_from_params(action_logits=action_mean, deterministic=False)
        
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class trainer(object):
    def __init__(self, args):
        self.device = torch.device("cuda:0")
        self.args = args
        # TRY NOT TO MODIFY: seeding
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic
        # Agent init
        self.agent = Agent(self.args.state_dim,self.args.action_dim,self.args.categorical_dim).to(self.device)
        # self.agent.load_state_dict(torch.load("/home/dhy/final/ISAC_OFDM/models/PPO_TD_2023-04-26-08-35-38/update_100.mo"))
        # ENV init
        self.env = ISAC_BS(N=args.subcarrier_number, N_c=self.args.cu_number, N_r=self.args.su_number, seed=args.seed)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=args.learning_rate, eps=1e-5)
        self.run_name = f"PPO_SCHED__{args.exp_name}__{args.seed}__{args.daytime}"

    def train(self, tensorboard=False):
        #
        results = dict()
        results["reward"]=[]
        results["reward_epoch_mean"]=[]
        results["action"]=[]
        results["baseline_reward"]=[]
        results["baseline_reward_epoch_mean"]=[]
        results["baseline_action"]=[]
        # results[""]=[]
        ##
        args = self.args
        run_name = self.run_name
        agent = self.agent
        optimizer = self.optimizer
        device = self.device
        env = self.env
        if tensorboard:
            writer = SummaryWriter(f"runs/{run_name}")
            writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
            )
        # ALGO Logic: Storage setup
        obs = torch.zeros((args.num_steps, args.state_dim)).to(device)
        actions = torch.zeros((args.num_steps, args.action_dim)).to(device)
        logprobs = torch.zeros((args.num_steps, 1)).to(device)
        rewards = torch.zeros((args.num_steps, 1)).to(device)
        dones = torch.zeros((args.num_steps, 1)).to(device)
        values = torch.zeros((args.num_steps, 1)).to(device)

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        next_state,done = env.reset()
        next_obs = torch.Tensor(next_state).to(device)
        next_done = torch.Tensor([done]).to(device)
        num_updates = args.total_timesteps // args.batch_size

        for update in range(1, num_updates + 1):
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            episode_reward = 0.0
            episode_reward_raw = 0.0
            bl_episode_reward = 0.0
            for step in range(0, args.num_steps):
                global_step += 1
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob
                # TRY NOT TO MODIFY: execute the game and log data.
                
                next_state,reward,done,EE_C,EE_R,bl_reward,bl_EE_C,bl_EE_R,reward_raw,_ = env.step(action.cpu().numpy(),freeze=False)
                # next_obs, reward, done = env.step()
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor([done]).to(device)
                log_action = action.cpu().numpy()
                if tensorboard:
                    writer.add_scalar("reward/reward", reward, global_step)
                    writer.add_scalar("reward/reward_raw", reward_raw, global_step)
                    writer.add_scalar("reward/baseline_random", bl_reward, global_step)
                    for k in range(args.action_dim):
                        writer.add_scalar(f"action/sc{k}", log_action[k], global_step)
                    writer.add_scalar("env/EE_C", EE_C, global_step)
                    writer.add_scalar("env/EE_R", EE_R, global_step)
                    writer.add_scalar("env/bl_EE_C", bl_EE_C, global_step)
                    writer.add_scalar("env/bl_EE_R", bl_EE_R, global_step)
                episode_reward += reward
                episode_reward_raw += reward_raw
                bl_episode_reward += bl_reward
                results["reward"].append(reward)
                results["reward_epoch_mean"].append(episode_reward/args.num_steps)
                # results["action"].append(action)
                results["baseline_reward"].append(bl_reward)
                results["baseline_reward_epoch_mean"].append(bl_episode_reward/args.num_steps)
            if tensorboard:
                writer.add_scalar("reward/epoch_mean", episode_reward/args.num_steps, global_step)
                writer.add_scalar("reward/epoch_mean_raw", episode_reward_raw/args.num_steps, global_step)
                writer.add_scalar("reward/bl_epoch_mean", bl_episode_reward/args.num_steps, global_step)
                
            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                if args.gae:
                    advantages = torch.zeros_like(rewards).to(device)
                    lastgaelam = 0
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            nextvalues = values[t + 1]
                        delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                        advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    returns = advantages + values
                else:
                    returns = torch.zeros_like(rewards).to(device)
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            next_return = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            next_return = returns[t + 1]
                        returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                    advantages = returns - values

            # flatten the batch
            b_obs = obs.reshape((-1,args.state_dim))
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,args.action_dim))
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizaing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    
                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        # old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)         
                    optimizer.step()
                    
                if args.target_kl is not None:
                    if approx_kl > args.target_kl:
                        break
            if update%50 == 0:
                if not os.path.exists(f'{os.path.dirname(__file__)}/models/PPO_{args.daytime}/'):
                    os.mkdir(f'{os.path.dirname(__file__)}/models/PPO_{args.daytime}/')
                torch.save(agent.state_dict(), \
                    f'{os.path.dirname(__file__)}/models/PPO_{args.daytime}/update_{update}.mo')
                
                if not os.path.exists(f'{os.path.dirname(__file__)}/logs/PPO_{args.daytime}/'):
                    os.mkdir(f'{os.path.dirname(__file__)}/logs/PPO_{args.daytime}/')
                f = open(f'{os.path.dirname(__file__)}/logs/PPO_{args.daytime}.pkl',"wb")
                pickle.dump(results,f)
                f.close()

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            if tensorboard:
                # TRY NOT TO MODIFY: record rewards for plotting purposes
                writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
                writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
                writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
                writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
                writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
                writer.add_scalar("losses/explained_variance", explained_var, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        if tensorboard:
            writer.close()
        
        torch.save(agent.state_dict(), f'{os.path.dirname(__file__)}/models/PPO_{args.daytime}_update_{update}.mo')
        return results


if __name__ == "__main__":
    args = parse_args()
    ppo_trainer = trainer(args)
    results = ppo_trainer.train(tensorboard=True)
    # results = ppo_trainer.train(tensorboard=False)
    
    
    
    # run_name = f"PPO_SCHED__{args.exp_name}__{args.seed}__{args.daytime}"
    # writer = SummaryWriter(f"runs/{run_name}")
    # writer.add_text(
    #     "hyperparameters",
    #     "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    # )

    
    # # TRY NOT TO MODIFY: seeding
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = args.torch_deterministic

    # # device = torch.device("cpu")
    # device = torch.device("cuda:0")


    # env = ISAC_BS(N=8, N_c=CATEGORICAL_DIM, N_r=ACTION_DIM, seed=777)
    # agent = Agent().to(device)
    # optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)



    # # ALGO Logic: Storage setup
    # obs = torch.zeros((args.num_steps, STATE_DIM)).to(device)
    # actions = torch.zeros((args.num_steps, ACTION_DIM)).to(device)
    # logprobs = torch.zeros((args.num_steps, 1)).to(device)
    # rewards = torch.zeros((args.num_steps, 1)).to(device)
    # dones = torch.zeros((args.num_steps, 1)).to(device)
    # values = torch.zeros((args.num_steps, 1)).to(device)

    # # TRY NOT TO MODIFY: start the game
    # global_step = 0
    # start_time = time.time()
    # next_state,done = env.reset()
    # next_obs = torch.Tensor(next_state).to(device)
    # next_done = torch.Tensor([done]).to(device)
    # num_updates = args.total_timesteps // args.batch_size

    # for update in range(1, num_updates + 1):
    #     # Annealing the rate if instructed to do so.
    #     if args.anneal_lr:
    #         frac = 1.0 - (update - 1.0) / num_updates
    #         lrnow = frac * args.learning_rate
    #         optimizer.param_groups[0]["lr"] = lrnow

    #     episode_reward = 0.0
    #     bl_episode_reward = 0.0
    #     for step in range(0, args.num_steps):
    #         global_step += 1
    #         obs[step] = next_obs
    #         dones[step] = next_done

    #         # ALGO LOGIC: action logic
    #         with torch.no_grad():
    #             action, logprob, _, value = agent.get_action_and_value(next_obs)
    #             values[step] = value.flatten()
    #         actions[step] = action
    #         logprobs[step] = logprob
    #         # TRY NOT TO MODIFY: execute the game and log data.
            
    #         next_obs, reward, done, bl_reward = env.step(action.cpu().numpy(),freeze=False)
    #         # next_obs, reward, done = env.step()
    #         rewards[step] = torch.tensor(reward).to(device).view(-1)
    #         next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor([done]).to(device)
    #         writer.add_scalar("reward/reward", reward, global_step)
    #         writer.add_scalar("reward/baseline_random", bl_reward, global_step)
    #         # writer.add_scalars("action/action_ue", {f"UE{i}":action[i] for i in range(8)}, global_step)
    #         episode_reward += reward
    #         bl_episode_reward += bl_reward
    #     writer.add_scalar("reward/epoch_mean", episode_reward/args.num_steps, global_step)
    #     writer.add_scalar("reward/bl_epoch_mean", bl_episode_reward/args.num_steps, global_step)
        
            
    #     # bootstrap value if not done
    #     with torch.no_grad():
    #         next_value = agent.get_value(next_obs).reshape(1, -1)
    #         if args.gae:
    #             advantages = torch.zeros_like(rewards).to(device)
    #             lastgaelam = 0
    #             for t in reversed(range(args.num_steps)):
    #                 if t == args.num_steps - 1:
    #                     nextnonterminal = 1.0 - next_done
    #                     nextvalues = next_value
    #                 else:
    #                     nextnonterminal = 1.0 - dones[t + 1]
    #                     nextvalues = values[t + 1]
    #                 delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
    #                 advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
    #             returns = advantages + values
    #         else:
    #             returns = torch.zeros_like(rewards).to(device)
    #             for t in reversed(range(args.num_steps)):
    #                 if t == args.num_steps - 1:
    #                     nextnonterminal = 1.0 - next_done
    #                     next_return = next_value
    #                 else:
    #                     nextnonterminal = 1.0 - dones[t + 1]
    #                     next_return = returns[t + 1]
    #                 returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
    #             advantages = returns - values

    #     # flatten the batch
    #     b_obs = obs.reshape((-1,STATE_DIM))
    #     b_logprobs = logprobs.reshape(-1)
    #     b_actions = actions.reshape((-1,ACTION_DIM))
    #     b_advantages = advantages.reshape(-1)
    #     b_returns = returns.reshape(-1)
    #     b_values = values.reshape(-1)

    #     # Optimizaing the policy and value network
    #     b_inds = np.arange(args.batch_size)
    #     clipfracs = []
    #     for epoch in range(args.update_epochs):
    #         np.random.shuffle(b_inds)
    #         for start in range(0, args.batch_size, args.minibatch_size):
    #             end = start + args.minibatch_size
    #             mb_inds = b_inds[start:end]

    #             _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
    #             logratio = newlogprob - b_logprobs[mb_inds]
    #             ratio = logratio.exp()
                
    #             with torch.no_grad():
    #                 # calculate approx_kl http://joschu.net/blog/kl-approx.html
    #                 # old_approx_kl = (-logratio).mean()
    #                 approx_kl = ((ratio - 1) - logratio).mean()
    #                 clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

    #             mb_advantages = b_advantages[mb_inds]
    #             if args.norm_adv:
    #                 mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

    #             # Policy loss
    #             pg_loss1 = -mb_advantages * ratio
    #             pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    #             pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    #             # Value loss
    #             newvalue = newvalue.view(-1)
    #             if args.clip_vloss:
    #                 v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
    #                 v_clipped = b_values[mb_inds] + torch.clamp(
    #                     newvalue - b_values[mb_inds],
    #                     -args.clip_coef,
    #                     args.clip_coef,
    #                 )
    #                 v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
    #                 v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
    #                 v_loss = 0.5 * v_loss_max.mean()
    #             else:
    #                 v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

    #             entropy_loss = entropy.mean()
    #             loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

    #             # assert(torch.isnan(loss).sum() == 0), print("pg_loss-",pg_loss,"entropy_loss-",entropy_loss,"v_loss-",v_loss)
                
    #             # for check_key in agent.critic.state_dict().keys():
    #             #     prm = agent.critic.state_dict()[check_key]
    #             #     assert(torch.isnan(prm).sum() == 0),print("critic nan before backward")
                    
    #             # for check_key in agent.actor_mean.state_dict().keys():
    #             #     prm = agent.actor_mean.state_dict()[check_key]
    #             #     assert(torch.isnan(prm).sum() == 0),print("actor_mean nan before backward")
                    
    #             # assert(torch.isnan(agent.actor_logstd).sum() == 0),print("actor_logstd nan before backward")
                
    #             optimizer.zero_grad()
    #             loss.backward()
    #             nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)         
    #             optimizer.step()
                
    #             # for check_key in agent.critic.state_dict().keys():
    #             #     prm = agent.critic.state_dict()[check_key]
    #             #     assert(torch.isnan(prm).sum() == 0),pdb.set_trace()

    #             # for check_key in agent.actor_mean.state_dict().keys():
    #             #     prm = agent.actor_mean.state_dict()[check_key]
    #             #     assert(torch.isnan(prm).sum() == 0),print("actor_mean nan after step")
                
    #             # assert(torch.isnan(agent.actor_logstd).sum() == 0),print("actor_logstd nan after step")
    #     if update%100 == 0:
    #         if not os.path.exists(f'{os.path.dirname(__file__)}/models/PPO_{args.daytime}/'):
    #             os.mkdir(f'{os.path.dirname(__file__)}/models/PPO_{args.daytime}/')
    #         torch.save(agent.state_dict(), \
    #             f'{os.path.dirname(__file__)}/models/PPO_{args.daytime}/update_{update}.mo')
    #         if args.target_kl is not None:
    #             if approx_kl > args.target_kl:
    #                 break
        

    #     y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    #     var_y = np.var(y_true)
    #     explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
    #     # TRY NOT TO MODIFY: record rewards for plotting purposes
    #     writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
    #     writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    #     writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    #     writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
    #     writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    #     writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
    #     writer.add_scalar("losses/explained_variance", explained_var, global_step)
    #     print("SPS:", int(global_step / (time.time() - start_time)))
    #     writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
    # writer.close()
    
    # torch.save(agent.state_dict(), f'{os.path.dirname(__file__)}/models/PPO_{args.daytime}_update_{update}.mo')