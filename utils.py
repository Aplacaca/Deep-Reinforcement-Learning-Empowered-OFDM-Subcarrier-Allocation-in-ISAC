import numpy as np
import torch
import torch.nn as nn

class MultiCategoricalDistribution():

	def __init__(self, action_dims):
		"""Initialization
		"""
		super(MultiCategoricalDistribution, self).__init__()
		self.action_dims = action_dims

	def proba_distribution_net(self, latent_dim):
		"""
		Create the layer that represents the distribution. 
		It will be the logits (flattened) of the MultiCategorical distribution.
		You can then get probabilities using a softmax on each sub-space.
		"""
		action_logits = nn.Linear(latent_dim, sum(self.action_dims))
		return action_logits

	def proba_distribution(self, action_logits):
		"""Create a list of categorical distribution for each dimension
		"""
		self.distribution = [torch.distributions.Categorical(logits=split) for split in torch.split(action_logits, tuple(self.action_dims), dim=-1)]
		return self

	def log_prob(self, actions):
		"""Extract each discrete action and compute log prob for their respective distributions
		"""
		return torch.stack(
			[dist.log_prob(action) for dist, action in zip(self.distribution, torch.unbind(actions, dim=-1))], dim=-1
		).sum(dim=-1)

	def entropy(self):
		"""Computes sum of entropy of individual caterogical dist
		"""
		return torch.stack([dist.entropy() for dist in self.distribution], dim=-1).sum(dim=-1)

	def sample(self):
		"""Samples actions from each individual categorical dist
		"""
		return torch.stack([dist.sample() for dist in self.distribution], dim=-1)

	def mode(self):
		"""Computes mode of each categorical dist.
		"""
		return torch.stack([torch.argmax(dist.probs, dim=-1) for dist in self.distribution], dim=-1)

	def get_actions(self, deterministic=False):
		"""Return actions according to the probability distribution.  
		"""
		if deterministic:
			return self.mode()
		return self.sample()

	def actions_from_params(self, action_logits, deterministic=False):
		"""Update the proba distribution
		"""
		self.proba_distribution(action_logits)
		return self.get_actions(deterministic=deterministic)

	def log_prob_from_params(self, action_logits):
		"""Compute log-probability of actions
		"""
		actions = self.actions_from_params(action_logits)
		log_prob = self.log_prob(actions)
		return actions, log_prob

