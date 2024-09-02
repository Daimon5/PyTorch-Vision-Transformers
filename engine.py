

import torch
from tqdm.auto import tqdm

def train_step(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module, optimizer: torch.optim, device: str, accuracy_fn):
    model = model.to(device)
    model.train()
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        X = X.to(device)
        y = y.to(device)

        y_logits = model(X)
        loss = loss_fn(y_logits, y)
        train_loss += loss
        train_acc += accuracy_fn(y_logits.argmax(dim= 1), y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    return train_loss, train_acc


def test_step(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module, device: str, accuracy_fn):
    model.eval()
    with torch.inference_mode():
        test_loss, test_acc = 0, 0
        for X, y in data_loader:
            X = X.to(device)
            y = y.to(device)

            y_logits = model(X)
            loss = loss_fn(y_logits, y)
            test_loss += loss
            test_acc += accuracy_fn(y_logits.argmax(dim= 1), y)

        test_loss /= len(data_loader)
        test_acc /= len(data_loader)

    return test_loss, test_acc


def train(model: torch.nn.Module, train_data_loader: torch.utils.data.DataLoader,
          test_data_loader: torch.utils.data.DataLoader,loss_fn: torch.nn.Module,
          optimizer: torch.optim, device: str, num_of_epochs: int, accuracy_fn):
    epochs = num_of_epochs
    results = {'train_loss' : [],
               'test_loss' : []}

    for epoch in tqdm(range(num_of_epochs)):
        train_loss, train_acc = train_step(model, train_data_loader, loss_fn,
                                           optimizer, device, accuracy_fn)

        test_loss, test_acc = test_step(model, test_data_loader, loss_fn, device,
                                        accuracy_fn)

        print()
        print(f"\n****** For Epoch: {epoch+1} ******")
        print(f"Train_loss: {train_loss} | Train_acc: {train_acc}")
        print(f"Test_loss: {test_loss} | Test_acc: {test_acc}\n")
        results['train_loss'].append(train_loss.cpu().detach().numpy())
        results['test_loss'].append(test_loss.cpu().detach().numpy())

    
    return results


