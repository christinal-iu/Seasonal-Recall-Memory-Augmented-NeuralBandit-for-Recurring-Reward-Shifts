import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from load_dataset import encode_features, load_dataset

def main():
    torch.manual_seed(67)

    batch_size = 1024
    epochs = 30
    lr = 0.001

    features, labels = load_dataset()
    features = torch.from_numpy(encode_features(features))

    # cross entropy uses class numbers from 0 to 6
    targets = torch.tensor(labels - 1, dtype=torch.long)

    dataset = TensorDataset(features, targets)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    # make a classifier with one hidden layer
    model = nn.Sequential(
        nn.Linear(94, 100),
        nn.ReLU(),
        nn.Linear(100, 7),
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0

        for contexts, classes in train_loader:
            outputs = model(contexts)
            loss = loss_fn(outputs, classes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(classes)
            correct += (outputs.argmax(dim=1) == classes).sum().item()

        print(
            f"epoch {epoch + 1}/{epochs}, "
            f"loss: {total_loss / len(dataset):.4f}, "
            f"training accuracy: {correct / len(dataset):.4f}",
            flush=True,
        )

    # predict every row in its original dataset order
    eval_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    model.eval()
    predictions = []

    with torch.no_grad():
        for contexts, classes in eval_loader:
            outputs = model(contexts)
            predicted_classes = outputs.argmax(dim=1) + 1
            predictions.append(predicted_classes.numpy())

    predictions = np.concatenate(predictions)
    accuracy = np.mean(predictions == labels)

    print(f"reference accuracy on fitted dataset: {accuracy:.4f}")

    # save predictions for the regret calculation
    np.savez_compressed(
        "reference_predictions.npz",
        predictions=predictions,
        labels=labels,
    )

    # save the trained model and its settings
    torch.save(
        {
            "model_state": model.state_dict(),
            "seed": 67,
            "hidden_size": 100,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
        },
        "reference_model.pt",
    )

if __name__ == "__main__":
    main()