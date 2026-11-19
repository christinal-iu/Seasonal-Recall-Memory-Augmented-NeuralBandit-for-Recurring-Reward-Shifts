import numpy as np
import torch
from torch import nn

class NeuralBandit1:
    def __init__(self, hidden_size, lr, rng):
        self.rng = rng
        self.gamma = 0.005
        self.models = []
        self.optimizers = []

        # make one network for each class
        for _ in range(7):
            model = nn.Sequential(
                nn.Linear(94, hidden_size, bias=False),
                nn.Sigmoid(),
                nn.Linear(hidden_size, 1, bias=False),
                nn.Sigmoid(),
            )

            for weights in model.parameters():
                nn.init.uniform_(weights, -0.5, 0.5)

            self.models.append(model)
            self.optimizers.append(torch.optim.SGD(model.parameters(), lr=lr))

    def choose_action(self, context):
        # predict reward for each action
        with torch.no_grad():
            scores = [model(context).item() for model in self.models]

        best_index = int(np.argmax(scores))
        
        # give every action small chance of being chosen
        probs = np.full(7, self.gamma / 7)
        probs[best_index] += 1 - self.gamma
        index = self.rng.choice(7, p=probs)

        return index + 1, probs[index]

    def update(self, context, action, reward, prob):
        # update only network for action that was chosen
        model = self.models[action - 1]
        optimizer = self.optimizers[action - 1]
        prediction = model(context).squeeze()
        loss = 0.5 * (prediction - reward) ** 2 / prob
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

class NeuralBandit2:
    def __init__(self, rng):
        self.rng = rng
        self.gamma_model = 0.1
        self.bandits = []

        # make one bandit for each combo of settings
        for hidden_size in [1, 5, 25, 50, 100]:
            for lr in [0.01, 0.1, 1.0]:
                self.bandits.append(NeuralBandit1(hidden_size, lr, rng))

        self.log_weights = np.zeros(len(self.bandits))

    def choose_action(self, context):
        # get exp3 prob of choosing each bandit
        weights = np.exp(self.log_weights - self.log_weights.max())
        num_bandits = len(self.bandits)
        probs = (1 - self.gamma_model) * weights / weights.sum()
        probs += self.gamma_model / num_bandits

        self.chosen_index = self.rng.choice(num_bandits, p=probs)
        self.chosen_prob = probs[self.chosen_index]

        # let chosen bandit choose action
        return self.bandits[self.chosen_index].choose_action(context)

    def update(self, context, action, reward, prob):
        # train every bandit on action that was played
        for bandit in self.bandits:
            bandit.update(context, action, reward, prob)

        # update exp3 weights of chosen bandit
        num_bandits = len(self.bandits)
        estimated_reward = reward / self.chosen_prob
        self.log_weights[self.chosen_index] += (self.gamma_model * estimated_reward / num_bandits)

class NeuralBandit3:
    def __init__(self, rng):
        self.rng = rng
        self.gamma = 0.005
        self.gamma_model = 0.1
        self.bandits = []

        for hidden_size in [1, 5, 25, 50, 100]:
            for lr in [0.01, 0.1, 1.0]:
                self.bandits.append(NeuralBandit1(hidden_size, lr, rng))

        self.log_weights = np.zeros((7, len(self.bandits)))

    def choose_action(self, context):
        scores = []
        model_indices = []
        model_probs = []
        num_bandits = len(self.bandits)

        for action_index in range(7):
            # pick configuration for action
            log_weights = self.log_weights[action_index]
            weights = np.exp(log_weights - log_weights.max())
            probs = (1 - self.gamma_model) * weights / weights.sum()
            probs += self.gamma_model / num_bandits

            model_index = self.rng.choice(num_bandits, p=probs)
            model_indices.append(model_index)
            model_probs.append(probs[model_index])

            model = self.bandits[model_index].models[action_index]
            with torch.no_grad():
                scores.append(model(context).item())

        # choose action from seven scores
        best_index = int(np.argmax(scores))
        action_probs = np.full(7, self.gamma / 7)
        action_probs[best_index] += 1 - self.gamma
        action_index = self.rng.choice(7, p=action_probs)

        self.chosen_models = model_indices
        self.chosen_model_probs = model_probs
        return action_index + 1, action_probs[action_index]

    def update(self, context, action, reward, prob):
        # train played action's network in every configuration
        for bandit in self.bandits:
            bandit.update(context, action, reward, prob)

        # update each exp3 selector using the observed reward
        num_bandits = len(self.bandits)

        for action_index in range(7):
            model_index = self.chosen_models[action_index]
            model_prob = self.chosen_model_probs[action_index]
            estimated_reward = reward / model_prob

            self.log_weights[action_index, model_index] += (self.gamma_model * estimated_reward / num_bandits)