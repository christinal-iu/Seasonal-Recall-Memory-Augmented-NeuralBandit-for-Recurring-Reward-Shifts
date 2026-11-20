import numpy as np

class LinUCB:
    def __init__(self, rng, alpha=1.0):
        self.rng = rng
        self.alpha = alpha

        # keep one model per class
        self.inverse = np.array([np.eye(94) for _ in range(7)])
        self.reward_sum = np.zeros((7, 94))

    def choose_action(self, context):
        context = np.asarray(context, dtype=np.float64)
        scores = []

        for index in range(7):
            inverse = self.inverse[index]
            weights = np.dot(inverse, self.reward_sum[index])

            # predict reward and calculate uncertainty
            prediction = np.dot(context, weights)
            product = np.dot(inverse, context)
            uncertainty = np.dot(context, product)

            bonus = self.alpha * np.sqrt(max(uncertainty, 0.0))
            scores.append(prediction + bonus)

        # randomly choose among actions w highest score
        scores = np.array(scores)
        best_indices = np.flatnonzero(scores == scores.max())
        index = self.rng.choice(best_indices)

        return index + 1, 1.0 / len(best_indices)

    def update(self, context, action, reward, prob):
        context = np.asarray(context, dtype=np.float64)
        index = action - 1

        # update inverse for the chosen action
        product = np.dot(self.inverse[index], context)
        denominator = 1 + np.dot(context, product)
        self.inverse[index] -= np.outer(product, product) / denominator

        # update reward info
        self.reward_sum[index] += reward * context

class CTS(LinUCB):
    def __init__(self, rng, scale=1.0):
        super().__init__(rng)
        self.scale = scale

    def choose_action(self, context):
        context = np.asarray(context, dtype=np.float64)
        scores = []

        for index in range(7):
            inverse = self.inverse[index]
            weights = np.dot(inverse, self.reward_sum[index])

            # calculate predicted reward and uncertainty
            prediction = np.dot(context, weights)
            product = np.dot(inverse, context)
            variance = np.dot(context, product)
            std = self.scale * np.sqrt(max(variance, 0.0))

            # sample a possible reward estimate for this action
            score = self.rng.normal(prediction, std)
            scores.append(score)

        index = int(np.argmax(scores))
        return index + 1, None

class Banditron:
    def __init__(self, rng):
        self.rng = rng
        self.gamma = 0.005
        self.weights = np.zeros((7, 94))

    def choose_action(self, context):
        context = np.asarray(context, dtype=np.float64)

        # find class with the highest linear score
        scores = np.dot(self.weights, context)
        self.best_index = int(np.argmax(scores))

        # give every class a small chance of being chosen
        probs = np.full(7, self.gamma / 7)
        probs[self.best_index] += 1 - self.gamma

        index = self.rng.choice(7, p=probs)
        return index + 1, probs[index]

    def update(self, context, action, reward, prob):
        context = np.asarray(context, dtype=np.float64)
        index = action - 1

        # subtract context from original best class
        self.weights[self.best_index] -= context

        # add weighted context when chosen class is correct
        if reward == 1:
            self.weights[index] += context / prob