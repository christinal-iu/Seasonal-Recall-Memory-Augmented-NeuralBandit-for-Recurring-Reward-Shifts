import copy
import numpy as np
import torch
from neural_bandit_general import NeuralBandit3

class SeasonalRecallShared(NeuralBandit3):
    def __init__(
        self,
        rng,
        input_size=94,
        num_actions=7,
        gamma=0.05,
        memory_enabled=True,
        archive_mode="diverse",
    ):
        super().__init__(rng, input_size, num_actions, gamma)

        self.memory_enabled = memory_enabled
        self.archive_mode = archive_mode
        self.memory_limit = 3
        self.archive_interval = 7000
        self.signature_size = 128
        self.min_disagreement = 0.1

        self.reference_contexts = []
        self.memory = []
        self.experts = []

        self.router_weights = np.zeros(1)
        self.router_rng = np.random.default_rng(0)
        self.router_gamma = 0.1
        self.router_discount = 0.999

        self.round = 0
        self.memory_used_rounds = 0

    def action_advice(self, policy, context):
        action, _ = NeuralBandit3.choose_action(
            policy, context
        )

        with torch.no_grad():
            scores = [
                policy.bandits[
                    policy.chosen_models[a]
                ].models[a](context).item()
                for a in range(self.num_actions)
            ]

        probs = np.full(
            self.num_actions,
            self.gamma / self.num_actions,
        )
        probs[int(np.argmax(scores))] += 1 - self.gamma

        return probs, action

    def choose_action(self, context):
        # calculate the model selection probs
        weights = np.exp(
            self.router_weights - self.router_weights.max()
        )
        weights = (
            (1 - self.router_gamma)
            * weights
            / weights.sum()
        )
        weights += self.router_gamma / len(weights)

        current_advice, current_action = self.action_advice(
            self, context
        )

        advice = [current_advice]

        for expert in self.experts:
            probs, _ = self.action_advice(expert, context)
            advice.append(probs)

        self.advice = np.array(advice)

        # calc action probs across all models
        behavior = np.sum(
            weights[:, None] * self.advice,
            axis=0,
        )

        self.chosen_expert = int(
            self.router_rng.choice(len(weights), p=weights)
        )
        self.expert_prob = float(
            weights[self.chosen_expert]
        )

        if len(weights) == 1:
            action = current_action
        else:
            action = int(
                self.rng.choice(
                    self.num_actions,
                    p=self.advice[self.chosen_expert],
                )
            ) + 1

        if self.chosen_expert > 0:
            self.memory_used_rounds += 1

        return action, float(behavior[action - 1])

    def update(self, context, action, reward, prob):
        self.round += 1

        # keep past contexts for comparing saved policies
        if len(self.reference_contexts) < self.signature_size:
            self.reference_contexts.append(
                context.detach().clone()
            )

        # keep training current learner
        for member in self.bandits:
            member.update(context, action, reward, prob)

        # share observed reward using action probs
        estimated_rewards = (
            reward * self.advice[:, action - 1] / prob
        )

        for a in range(self.num_actions):
            selected = self.chosen_models[a]
            selected_prob = self.chosen_model_probs[a]

            self.log_weights[a, selected] += (
                self.gamma_model
                * estimated_rewards[0]
                / (selected_prob * len(self.bandits))
            )

        # gradually reduce influence of older feedback
        self.router_weights *= self.router_discount
        self.router_weights += (
            self.router_gamma
            * estimated_rewards
            / len(self.router_weights)
        )

        if (
            self.memory_enabled
            and self.round % self.archive_interval == 0
            and len(self.reference_contexts) == self.signature_size
        ):
            self.archive_current()

    def policy_signature(self):
        contexts = torch.stack(self.reference_contexts)

        scores = np.zeros(
            (len(contexts), self.num_actions)
        )

        # compare decisions on the same past contexts
        with torch.no_grad():
            for a in range(self.num_actions):
                weights = np.exp(
                    self.log_weights[a]
                    - self.log_weights[a].max()
                )

                probs = (
                    (1 - self.gamma_model)
                    * weights
                    / weights.sum()
                )
                probs += (
                    self.gamma_model / len(self.bandits)
                )

                for index, member in enumerate(self.bandits):
                    outputs = member.models[a](contexts)
                    outputs = outputs.squeeze(-1)

                    scores[:, a] += (
                        probs[index] * outputs.cpu().numpy()
                    )

        return scores.argmax(axis=1)

    def archive_current(self):
        signature = self.policy_signature()
        drop = None

        if self.memory:
            if self.archive_mode == "fifo":
                # remove oldest memory when full
                if len(self.memory) >= self.memory_limit:
                    drop = 0

            else:
                distances = [
                    np.mean(signature != item["signature"])
                    for item in self.memory
                ]

                nearest = int(np.argmin(distances))

                if distances[nearest] < self.min_disagreement:
                    # refresh a memory w similar decisions
                    drop = nearest

                elif len(self.memory) >= self.memory_limit:
                    signatures = [
                        item["signature"]
                        for item in self.memory
                    ]
                    signatures.append(signature)

                    closest_distance = float("inf")

                    for left in range(len(signatures)):
                        for right in range(
                            left + 1, len(signatures)
                        ):
                            distance = np.mean(
                                signatures[left]
                                != signatures[right]
                            )

                            if distance < closest_distance:
                                closest_distance = distance

                                # left is the older memory
                                drop = left

        # preserve preferences for surviving memories
        old_weights = self.router_weights.copy()

        old_memories = {
            item["round"]: old_weights[index + 1]
            for index, item in enumerate(self.memory)
        }

        maximum = old_weights.max()
        new_prior = maximum + np.log(
            np.exp(old_weights - maximum).mean()
        )

        if drop is not None:
            self.memory.pop(drop)
            self.experts.pop(drop)

        expert = object.__new__(NeuralBandit3)

        expert.rng = self.rng
        expert.input_size = self.input_size
        expert.num_actions = self.num_actions
        expert.gamma = self.gamma
        expert.gamma_model = self.gamma_model
        expert.bandits = copy.deepcopy(self.bandits)
        expert.log_weights = self.log_weights.copy()

        self.experts.append(expert)

        self.memory.append({
            "round": self.round,
            "signature": signature,
        })

        self.router_weights = np.array([
            old_weights[0],
            *[
                old_memories.get(item["round"], new_prior)
                for item in self.memory
            ],
        ])

        self.router_weights -= self.router_weights.max()