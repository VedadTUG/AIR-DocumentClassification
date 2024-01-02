import matplotlib
matplotlib.use('TkAgg')
from tqdm import tqdm
import torch

from sklearn.metrics import accuracy_score




def TrainModel(model, loss_fn, optimizer, train_loader, val_loader, epochs):
    gpu_avail = torch.cuda.is_available()
    print(f"Is the GPU available? {gpu_avail}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Device", device)
    model.to(device)

    for i in range(1, epochs + 1):
        model.train()
        losses = []
        for batch, (X, Y) in enumerate(tqdm(train_loader)):
            token_tensor = torch.stack(X)
            token_tensor = token_tensor.permute(1, 0)
            token_tensor = token_tensor.to(device)
            target_tensor = Y
            target_tensor = target_tensor.to(device)

            Y_preds = model(token_tensor)
            loss = loss_fn(Y_preds, target_tensor)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Train Loss : {:.3f}".format(torch.tensor(losses).mean()))

        model.eval()
        with torch.no_grad():
            Y_shuffled, Y_preds = [], []
            valid_loss = 0
            count = 0
            for X, Y in val_loader:
                token_tensor = torch.stack(X)
                token_tensor = token_tensor.permute(1, 0)
                token_tensor = token_tensor.to(device)
                target_tensor = Y
                target_tensor = target_tensor.to(device)
                preds = model(token_tensor)
                loss = loss_fn(preds, target_tensor)
                valid_loss += loss.item()
                count += 1

                Y_shuffled.append(Y)
                Y_preds.append(preds.argmax(dim=-1))

            Y_shuffled = torch.cat(Y_shuffled).cpu()
            Y_preds = torch.cat(Y_preds).cpu()

            print("Valid Loss : {:.3f}".format(valid_loss / count))
            print(
                "Valid Acc  : {:.3f}".format(accuracy_score(Y_shuffled.detach().numpy(), Y_preds.detach().numpy())))


