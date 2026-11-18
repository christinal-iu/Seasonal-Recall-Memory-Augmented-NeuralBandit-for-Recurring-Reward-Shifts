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