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
from utils import MultiCategoricalDistribution
from torch.utils.tensorboard import SummaryWriter

from env.ofdm_env import ISAC_BS
import pdb

STATE_DIM = 144
ACTION_DIM = 80

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--daytime', type=str, default=datetime.datetime.now().strftime('TD_%Y-%m-%d-%H-%M-%S'),
        help='the time of this experiment')
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
        help='the name of this experiment')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=777,
        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=2000000,
        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, cuda will be enabled by default')
    parser.add_argument('--track', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, this experiment will be tracked with Weights and Biases')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='weather to capture videos of the agent performances (check out `videos` folder)')

    # Algorithm specific arguments
    parser.add_argument('--num-envs', type=int, default=1,
        help='the number of parallel game environments')
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
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, ):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(STATE_DIM, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(STATE_DIM, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, ACTION_DIM*2), std=0.01),
            nn.Sigmoid(), # 4.21 W SIGMOID
        )
        # self.actor_logstd = nn.Parameter(torch.zeros(1, ACTION_DIM))
        # self.actor_logstd = 0.2*(torch.zeros(1, 8)).to(torch.device("cuda:0"))
    
    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        if torch.any(torch.isnan(action_mean)):
            print("caught nan")
            pdb.set_trace()
        # probs = Bernoulli(logits = action_mean)
        # probs = Categorical(logits = action_mean)
        probs = MultiCategoricalDistribution(action_dims=[2]*ACTION_DIM)
        probs.actions_from_params(action_logits=action_mean, deterministic=False)
        
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(-1), probs.entropy().sum(-1), self.critic(x)


if __name__ == "__main__":
    args = parse_args()
    run_name = f"PPO_SCHED__{args.exp_name}__{args.seed}__{args.daytime}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda:0")


    env = ISAC_BS(N=8, N_c=8, N_r=2, seed=777)
    agent = Agent().to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)



    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, STATE_DIM)).to(device)
    actions = torch.zeros((args.num_steps, ACTION_DIM)).to(device)
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
            
            next_obs, reward, done = env.step(action.cpu().numpy(),freeze=True)
            # next_obs, reward, done = env.step()
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor([done]).to(device)
            writer.add_scalar("reward/reward", reward, global_step)
            # writer.add_scalar("reward/baseline", bl_reward, global_step)
            # writer.add_scalars("action/action_ue", {f"UE{i}":action[i] for i in range(8)}, global_step)
            episode_reward += reward
            # bl_episode_reward += bl_reward
        writer.add_scalar("reward/epoch_mean", episode_reward/args.num_steps, global_step)
        # writer.add_scalar("reward/bl_epoch_mean", bl_episode_reward/args.num_steps, global_step)
        
            
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
        b_obs = obs.reshape((-1,STATE_DIM))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,ACTION_DIM))
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

                # assert(torch.isnan(loss).sum() == 0), print("pg_loss-",pg_loss,"entropy_loss-",entropy_loss,"v_loss-",v_loss)
                
                # for check_key in agent.critic.state_dict().keys():
                #     prm = agent.critic.state_dict()[check_key]
                #     assert(torch.isnan(prm).sum() == 0),print("critic nan before backward")
                    
                # for check_key in agent.actor_mean.state_dict().keys():
                #     prm = agent.actor_mean.state_dict()[check_key]
                #     assert(torch.isnan(prm).sum() == 0),print("actor_mean nan before backward")
                    
                # assert(torch.isnan(agent.actor_logstd).sum() == 0),print("actor_logstd nan before backward")
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)         
                optimizer.step()
                
                # for check_key in agent.critic.state_dict().keys():
                #     prm = agent.critic.state_dict()[check_key]
                #     assert(torch.isnan(prm).sum() == 0),pdb.set_trace()

                # for check_key in agent.actor_mean.state_dict().keys():
                #     prm = agent.actor_mean.state_dict()[check_key]
                #     assert(torch.isnan(prm).sum() == 0),print("actor_mean nan after step")
                
                # assert(torch.isnan(agent.actor_logstd).sum() == 0),print("actor_logstd nan after step")
        if update%100 == 0:
            if not os.path.exists(f'{os.path.dirname(__file__)}/models/PPO_{args.daytime}/'):
                os.mkdir(f'{os.path.dirname(__file__)}/models/PPO_{args.daytime}/')
            torch.save(agent.state_dict(), \
                f'{os.path.dirname(__file__)}/models/PPO_{args.daytime}/update_{update}.mo')
            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break
        

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
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
    writer.close()
    
    torch.save(agent.state_dict(), f'{os.path.dirname(__file__)}/models/PPO_{args.daytime}_update_{update}.mo')