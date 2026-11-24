import numpy as np
from sklearn.datasets import load_digits

def load_digits_dataset():
    features, labels = load_digits(return_X_y=True)

    # scale pixels from 0-16 to 0-1
    features = features.astype(np.float32) / 16.0

    labels = labels + 1

    return features, labels

if __name__ == "__main__":
    features, labels = load_digits_dataset()

    print("features:", features.shape)
    print("labels:", labels.shape)
    print("actions:", np.unique(labels))
    print("feature range:", features.min(), features.max())