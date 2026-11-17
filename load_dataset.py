import numpy as np
from sklearn.datasets import fetch_covtype

def load_dataset():
    features, labels = fetch_covtype(return_X_y=True)
    return features, labels

if __name__ == "__main__":
    features, labels = load_dataset()
    print("features:", features.shape)
    print("labels:", labels.shape)
    print("classes:", np.unique(labels))