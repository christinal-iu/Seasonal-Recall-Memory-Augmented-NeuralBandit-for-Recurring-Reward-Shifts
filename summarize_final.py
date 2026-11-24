import csv
import os
import numpy as np
from scipy.stats import t

def main():
    datasets = ["covertype", "digits"]
    settings = ["stationary", "recurring", "novel"]
    seeds = [90, 91, 92, 93, 94]

    results = {}
    rows = []

    for dataset in datasets:
        for setting in settings:
            methods = ["original", "explore", "memory"]

            if setting == "recurring":
                methods.append("fifo")

            for method in methods:
                for seed in seeds:
                    filename = (
                        f"final_results/final_v1_{dataset}"
                        f"_{method}_{setting}"
                        f"_rounds100000_interval25000"
                        f"_seed{seed}.npz"
                    )

                    if not os.path.exists(filename):
                        raise FileNotFoundError(filename)

                    with np.load(filename) as data:
                        rewards = data["rewards"].copy()
                        elapsed = float(data["elapsed_seconds"])

                    if rewards.shape != (100000,):
                        raise ValueError(
                            f"unexpected reward length: {filename}"
                        )

                    if not np.isin(rewards, [0, 1]).all():
                        raise ValueError(
                            f"invalid rewards: {filename}"
                        )

                    values = {
                        "overall": float(rewards.mean()),
                        "seconds": elapsed,
                    }

                    if setting == "recurring":
                        # both patterns return in the last two segments
                        values["returning"] = float(
                            rewards[50000:].mean()
                        )

                        # first 1000 rounds after each return
                        early = np.concatenate([
                            rewards[50000:51000],
                            rewards[75000:76000],
                        ])

                        values["early_return"] = float(
                            early.mean()
                        )

                    results[
                        dataset, setting, method, seed
                    ] = values

                    rows.append({
                        "dataset": dataset,
                        "setting": setting,
                        "method": method,
                        "seed": seed,
                        **values,
                    })

    with open(
        "final_run_metrics.csv", "w", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "dataset",
                "setting",
                "method",
                "seed",
                "overall",
                "seconds",
                "returning",
                "early_return",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    comparison_rows = []

    for dataset in datasets:
        for setting in settings:
            print(f"\n{dataset}, {setting}")

            methods = ["original", "explore", "memory"]

            if setting == "recurring":
                methods.append("fifo")

            metrics = ["overall", "seconds"]

            if setting == "recurring":
                metrics += ["returning", "early_return"]

            for method in methods:
                for metric in metrics:
                    values = np.array([
                        results[
                            dataset, setting, method, seed
                        ][metric]
                        for seed in seeds
                    ])

                    mean = values.mean()
                    std = values.std(ddof=1)

                    summary_rows.append({
                        "dataset": dataset,
                        "setting": setting,
                        "method": method,
                        "metric": metric,
                        "mean": mean,
                        "std": std,
                    })

                    if metric == "overall":
                        print(
                            f"{method}: "
                            f"{100 * mean:.2f}% "
                            f"+/- {100 * std:.2f} pp"
                        )

            comparisons = [
                ("explore", "original"),
                ("memory", "explore"),
            ]

            if setting == "recurring":
                comparisons.append(("memory", "fifo"))

            for method, baseline in comparisons:
                for metric in metrics:
                    if metric == "seconds":
                        continue

                    differences = np.array([
                        results[
                            dataset, setting, method, seed
                        ][metric]
                        - results[
                            dataset, setting, baseline, seed
                        ][metric]
                        for seed in seeds
                    ])

                    mean = differences.mean()
                    standard_error = (
                        differences.std(ddof=1)
                        / np.sqrt(len(seeds))
                    )

                    margin = (
                        t.ppf(0.975, len(seeds) - 1)
                        * standard_error
                    )

                    comparison_rows.append({
                        "dataset": dataset,
                        "setting": setting,
                        "method": method,
                        "baseline": baseline,
                        "metric": metric,
                        "mean_difference_pp": 100 * mean,
                        "ci_lower_pp": 100 * (mean - margin),
                        "ci_upper_pp": 100 * (mean + margin),
                        "positive_seeds": int(
                            (differences > 0).sum()
                        ),
                    })

                    print(
                        f"{method} minus {baseline}, "
                        f"{metric}: {100 * mean:+.2f} pp, "
                        f"95% CI "
                        f"[{100 * (mean - margin):+.2f}, "
                        f"{100 * (mean + margin):+.2f}], "
                        f"positive seeds: "
                        f"{(differences > 0).sum()}/5"
                    )

    for filename, values in [
        ("final_summary.csv", summary_rows),
        ("final_comparisons.csv", comparison_rows),
    ]:
        with open(filename, "w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(values[0]),
            )
            writer.writeheader()
            writer.writerows(values)

    print("\nchecked all 100 result files")
    print("saved final_run_metrics.csv")
    print("saved final_summary.csv")
    print("saved final_comparisons.csv")


if __name__ == "__main__":
    main()