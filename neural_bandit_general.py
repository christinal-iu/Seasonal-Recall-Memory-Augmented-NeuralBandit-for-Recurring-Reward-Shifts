import numpy as np
import torch
from torch import nn

# reward networks for one hidden size and lr
class RewardModels:
    def __init__(self, input_size, num_actions, hidden_size, lr):
        self.models = []
        self.optimizers = []

        # each action has its own network to predict reward
        for _ in range(num_actions):
            model = nn.Sequential(
                nn.Linear(input_size, hidden_size, bias=False),
                nn.Sigmoid(),
                nn.Linear(hidden_size, 1, bias=False),
                nn.Sigmoid(),
            )
            # initialize weights uniformly between -0.5 and 0.5
            for weights in model.parameters():
                nn.init.uniform_(weights, -0.5, 0.5)

            self.models.append(model)
            self.optimizers.append(
                torch.optim.SGD(model.parameters(), lr=lr)
            )

    def update(self, context, action, reward, prob):
        model = self.models[action - 1]
        optimizer = self.optimizers[action - 1]

        prediction = model(context).squeeze()
        # weight sq error by inverse action prob
        loss = 0.5 * (prediction - reward) ** 2 / prob

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class NeuralBandit3:
    def __init__(
        self,
        rng,
        input_size=94,
        num_actions=7,
        gamma=0.005,
    ):
        self.rng = rng
        self.input_size = input_size
        self.num_actions = num_actions
        self.gamma = gamma
        self.gamma_model = 0.1
        self.bandits = []

        # combine five hidden sizes w/ 3 lr
        for hidden_size in [1, 5, 25, 50, 100]:
            for lr in [0.01, 0.1, 1.0]:
                self.bandits.append(
                    RewardModels(
                        input_size,
                        num_actions,
                        hidden_size,
                        lr,
                    )
                )

        self.log_weights = np.zeros(
            (num_actions, len(self.bandits))
        )

    def choose_action(self, context):
        scores = []
        model_indices = []
        model_probs = []
        num_models = len(self.bandits)

        for action_index in range(self.num_actions):
            # keep separate config selection weights for each action
            log_weights = self.log_weights[action_index]
            # subtract max to avoid overflow when exponentiating
            weights = np.exp(log_weights - log_weights.max())
            # mix weighted configuration selection w uniform exploration
            probs = (
                (1 - self.gamma_model)
                * weights
                / weights.sum()
            )
            probs += self.gamma_model / num_models

            model_index = self.rng.choice(
                num_models, p=probs
            )

            model_indices.append(model_index)
            model_probs.append(probs[model_index])

            model = self.bandits[model_index].models[action_index]

            with torch.no_grad():
                scores.append(model(context).item())

        # favor highest pred reward while exploring every action
        best_index = int(np.argmax(scores))

        action_probs = np.full(
            self.num_actions,
            self.gamma / self.num_actions,
        )
        action_probs[best_index] += 1 - self.gamma

        action_index = self.rng.choice(
            self.num_actions, p=action_probs
        )

        self.chosen_models = model_indices
        self.chosen_model_probs = model_probs

        return action_index + 1, action_probs[action_index]

    def update(self, context, action, reward, prob):
        # update played action's network in all 15 configurations
        for member in self.bandits:
            member.update(context, action, reward, prob)

        num_models = len(self.bandits)

        # use the observed reward to update each action's sampled config
        for action_index in range(self.num_actions):
            model_index = self.chosen_models[action_index]
            model_prob = self.chosen_model_probs[action_index]

            estimated_reward = reward / model_prob

            self.log_weights[action_index, model_index] += (
                self.gamma_model
                * estimated_reward
                / num_models
            )