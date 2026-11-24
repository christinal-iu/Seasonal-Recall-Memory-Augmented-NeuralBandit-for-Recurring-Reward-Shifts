import csv
import numpy as np

def main():
    with open("final_run_metrics.csv", newline="") as file:
        rows = list(csv.DictReader(file))

    output = []

    for dataset in ["covertype", "digits"]:
        for setting in ["stationary", "recurring", "novel"]:
            print(f"\n{dataset}, {setting}")

            methods = ["original", "explore", "memory"]

            if setting == "recurring":
                methods.append("fifo")

            times = {}

            for method in methods:
                selected = [
                    row for row in rows
                    if row["dataset"] == dataset
                    and row["setting"] == setting
                    and row["method"] == method
                ]

                selected.sort(key=lambda row: int(row["seed"]))

                if len(selected) != 5:
                    raise ValueError("expected five seeds")

                times[method] = np.array([
                    float(row["seconds"])
                    for row in selected
                ])

            baseline = times["explore"].mean()

            for method in methods:
                values = times[method]
                mean = values.mean()
                std = values.std(ddof=1)
                ratio = mean / baseline

                print(
                    f"{method}: {mean:.1f} +/- {std:.1f} seconds, "
                    f"{ratio:.2f}x exploration baseline"
                )

                output.append({
                    "dataset": dataset,
                    "setting": setting,
                    "method": method,
                    "mean_seconds": mean,
                    "std_seconds": std,
                    "ratio_to_explore": ratio,
                })

            overhead = (
                times["memory"].mean() / baseline - 1
            ) * 100

            print(f"memory runtime overhead: {overhead:.1f}%")

    with open("final_runtime.csv", "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(output[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(output)

    print("\nsaved final_runtime.csv")


if __name__ == "__main__":
    main()