import numpy as np
from sklearn.datasets import fetch_covtype

def load_dataset():
    features, labels = fetch_covtype(return_X_y=True)
    return features, labels

def encode_features(features):
    binary_features = features[:, 10:]
    numeric_features = []

    # split each numeric col into 5 bins
    for column in range(10):
        values = features[:, column]
        cutoffs = np.quantile(values, [0.2, 0.4, 0.6, 0.8])
        bins = np.searchsorted(cutoffs, values, side='right')
        
        # make one binary col for each bin
        for bin_number in range(5):
            numeric_features.append(bins == bin_number)
            
    numeric_features = np.column_stack(numeric_features)
    encoded_features = np.column_stack((numeric_features, binary_features))
    return encoded_features.astype(np.float32)

if __name__ == "__main__":
    features, labels = load_dataset()
    encoded_features = encode_features(features)
    print("features:", features.shape)
    print("labels:", labels.shape)
    print("classes:", np.unique(labels))
    print("encoded features:", encoded_features.shape)