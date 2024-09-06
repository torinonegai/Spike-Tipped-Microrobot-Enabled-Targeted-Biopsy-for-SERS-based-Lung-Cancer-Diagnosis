import torch
def train_loop(dataloader, model, loss_fn, optimizer, device = "cuda"):
    size = len(dataloader.dataset)
    train_loss, train_accuracy = 0, 0
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()*len(X)
        train_accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()

    train_loss /= size
    train_accuracy /= size

    return train_loss, train_accuracy


def test_loop(dataloader, model, loss_fn, device = "cuda"):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, test_accuracy = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()*len(X)
            test_accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    test_accuracy /= size

    return test_loss, test_accuracy