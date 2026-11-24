import os
import time
import numpy as np
import torch
from load_dataset import load_dataset, encode_features
from load_digits_dataset import load_digits_dataset
from neural_bandit_general import NeuralBandit3
from seasonal_recall_shared import SeasonalRecallShared

# timing check
pilot = False
version = "final_v1"

def run_experiment(
    features,
    labels,
    dataset,
    method,
    setting,
    seed,
    rounds,
    interval,
    folder,
):
    filename = os.path.join(
        folder,
        f"{version}_{dataset}_{method}_{setting}"
        f"_rounds{rounds}_interval{interval}_seed{seed}.npz",
    )

    # check saved results before reusing them
    if os.path.exists(filename):
        with np.load(filename) as data:
            if data["rewards"].shape != (rounds,):
                raise ValueError(f"incomplete run: {filename}")

            expected = {
                "dataset": dataset,
                "method": method,
                "setting": setting,
                "seed": seed,
                "rounds": rounds,
                "interval": interval,
                "version": version,
            }

            for name, value in expected.items():
                if data[name].item() != value:
                    raise ValueError(
                        f"saved {name} does not match: {filename}"
                    )

        print(f"skipping {filename}", flush=True)
        return

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # match context order & starting position across methods
    order = np.random.default_rng(67).permutation(len(labels))
    start = rng.integers(len(order))

    num_actions = len(np.unique(labels))
    input_size = features.shape[1]

    if method == "original":
        gamma = 0.005
    else:
        gamma = 0.05
    # compare memory against baseline w matching exploration
    if method in ["original", "explore"]:
        bandit = NeuralBandit3(
            rng=rng,
            input_size=input_size,
            num_actions=num_actions,
            gamma=gamma,
        )
    else:
        archive_mode = "fifo" if method == "fifo" else "diverse"

        bandit = SeasonalRecallShared(
            rng=rng,
            input_size=input_size,
            num_actions=num_actions,
            gamma=gamma,
            archive_mode=archive_mode,
        )

    rewards = np.zeros(rounds, dtype=np.uint8)
    actions = np.zeros(rounds, dtype=np.uint8)
    action_probs = np.zeros(rounds)
    # -1 indicates current learner was selected
    checkpoints = np.full(rounds, -1, dtype=np.int64)

    print(
        f"starting {dataset}, {method}, "
        f"{setting}, seed {seed}",
        flush=True,
    )
    # measure learning loop, including memory archiving
    started = time.perf_counter()
    progress_interval = min(10000, interval)

    for step in range(rounds):
        # repeat shuffled dataset when all rows have been used
        row = order[(start + step) % len(order)]
        context = features[row]
        segment = step // interval

        # construct fixed, recurring, or previously unseen label shifts
        if setting == "stationary":
            shift = 0
        elif setting == "recurring":
            shift = segment % 2
        elif setting == "novel":
            shift = segment
        else:
            raise ValueError("unknown setting")

        label = (
            (int(labels[row]) - 1 + shift)
            % num_actions
            + 1
        )

        action, prob = bandit.choose_action(context)

        # record the selected memory before archive changes
        if (
            hasattr(bandit, "chosen_expert")
            and bandit.chosen_expert > 0
        ):
            checkpoints[step] = bandit.memory[
                bandit.chosen_expert - 1
            ]["round"]

        reward = int(action == label)

        rewards[step] = reward
        actions[step] = action
        action_probs[step] = prob

        # reveal only the reward for chosen action
        bandit.update(context, action, reward, prob)

        if (step + 1) % progress_interval == 0:
            recent = rewards[
                step + 1 - progress_interval:step + 1
            ]

            print(
                f"finished {step + 1:,} / {rounds:,}, "
                f"recent reward: {recent.mean():.4f}",
                flush=True,
            )

    elapsed = time.perf_counter() - started

    np.savez_compressed(
        filename,
        rewards=rewards,
        actions=actions,
        action_probs=action_probs,
        chosen_checkpoints=checkpoints,
        dataset=dataset,
        method=method,
        setting=setting,
        seed=seed,
        rounds=rounds,
        interval=interval,
        gamma=gamma,
        input_size=input_size,
        num_actions=num_actions,
        shuffle_seed=67,
        start_row=start,
        elapsed_seconds=elapsed,
        torch_threads=torch.get_num_threads(),
        version=version,
    )

    print(
        f"overall reward: {rewards.mean():.4f}",
        flush=True,
    )
    print(
        f"elapsed seconds: {elapsed:.1f}",
        flush=True,
    )
    print(f"saved {filename}", flush=True)


def main():
    if pilot:
        rounds = 10000
        interval = 2500
        seeds = [90]
        folder = "pilot_results"

        experiments = [
            ("original", "recurring"),
            ("explore", "recurring"),
            ("memory", "recurring"),
        ]

    else:
        rounds = 100000
        interval = 25000
        seeds = [90, 91, 92, 93, 94]
        folder = "final_results"

        experiments = [
            (method, setting)
            for setting in [
                "stationary",
                "recurring",
                "novel",
            ]
            for method in [
                "original",
                "explore",
                "memory",
            ]
        ]

        # compare diverse and fifo archives on recurring patterns
        experiments.append(("fifo", "recurring"))

    os.makedirs(folder, exist_ok=True)

    for dataset in ["covertype", "digits"]:
        if dataset == "covertype":
            features, labels = load_dataset()
            features = encode_features(features)
        else:
            features, labels = load_digits_dataset()

        features = torch.from_numpy(features)

        for seed in seeds:
            for method, setting in experiments:
                run_experiment(
                    features,
                    labels,
                    dataset,
                    method,
                    setting,
                    seed,
                    rounds,
                    interval,
                    folder,
                )

    print("all runs finished", flush=True)

if __name__ == "__main__":
    main()