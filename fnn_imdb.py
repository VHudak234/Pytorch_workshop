import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_from_disk
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
import json
from torch.optim.lr_scheduler import ReduceLROnPlateau
import text_models

def train(device, args, model, train_loader, optimizer, criterion, epoch, privacy_engine=None, scheduler=None):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, batch in enumerate(train_loader):
        inputs = batch["bow"].float().to(device)
        labels = batch["label"].float().to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        prediction = (outputs >= 0.5).float()
        correct += (prediction == labels).sum().item()

        train_loss += loss.item() * inputs.size(0)
        total += labels.size(0)

        if batch_idx % args.log_interval == 0:
            if args.private:
                epsilon = privacy_engine.get_epsilon(args.delta)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} (ε = {:.2f}, δ = {})'.format(
                    epoch, batch_idx * labels.size(0), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item(), epsilon, args.delta))
            else:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * labels.size(0), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
    if args.scheduler:
        scheduler.step(train_loss)
    return train_loss / len(train_loader), correct / total

def test(device, model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["bow"].float().to(device)
            labels = batch["label"].float().to(device).unsqueeze(1)

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            test_loss += loss.item() * labels.size(0)

            prediction = (outputs >= 0.5).float()
            correct += (prediction == labels).sum().item()
            total += labels.size(0)
    avg_loss = test_loss / total
    accuracy = correct / total
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, len(test_loader.dataset), 100.0 * accuracy))
    return avg_loss, accuracy

def construct_parser():
    parser = argparse.ArgumentParser(description='Model conficuration for training FNN on text data')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epsilon', type=int, default=5)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--delta', type=float, default=1e-5)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('--private', action='store_true')
    parser.add_argument('--scheduler', action='store_true')
    parser.add_argument('--load-model', action='store_true')
    parser.add_argument('--model-name', type=str, default=None)
    parser.add_argument('--save-model', action='store_true')

    return parser

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    privacy_engine = None
    scheduler = None

    seed = torch.randint(0, 2**32, (1,)).item()
    torch.manual_seed(seed)

    train_dataset = load_from_disk(f"{args.input}/imdb_train")
    test_dataset = load_from_disk(f"{args.input}/imdb_test")

    batch_size = args.batch_size

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    with open(f"{args.input}/vocab.json", "r") as f:
        vocab = json.load(f)

    vocab_size = len(vocab)

    model = text_models.ImdbFNN(vocab_size)

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    if args.load_model:
        loaded_model = torch.load(f'{args.output}/saved_models/{args.model_name}')
        if "SGD" not in args.model_name:
            modified_dict = {k.replace("_module.", ""): v for k, v in loaded_model["model_state_dict"].items()}
            model.load_state_dict(modified_dict)
            modified_optimizer_dict = {
                "state": {},
                "param_groups": loaded_model["optimizer_state_dict"]["param_groups"],
            }
            for k, v in loaded_model["optimizer_state_dict"]["state"].items():
                if "momentum_buffer" in v:
                    modified_optimizer_dict["state"][k] = {"exp_avg": v["exp_avg"], "exp_avg_sq": v["exp_avg_sq"]}
            optimizer.load_state_dict(modified_optimizer_dict)
        else:
            model.load_state_dict(loaded_model["model_state_dict"])
            optimizer.load_state_dict(loaded_model["optimizer_state_dict"])

    if args.private:
        privacy_engine = PrivacyEngine(accountant='rdp')
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=args.epochs,
            target_epsilon=args.epsilon,
            target_delta=args.delta,
            max_grad_norm=args.max_grad_norm,
        )

    if args.scheduler:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.85)

    if args.private:
        model_name = f'FNN_Text_{args.epsilon}_{args.lr}_Epochs{args.epochs}_BatchS{args.batch_size}_Sch{args.scheduler}_{seed}'
    else:
        model_name = f'FNN_SGDText_{args.lr}_Epochs{args.epochs}_BatchS{args.batch_size}_{seed}'
    if args.load_model:
        model_name = f'{model_name}_Loaded_model'

    os.makedirs(args.output, exist_ok=True)
    log_path = f'{args.output}/{model_name}.log'
    log_fh = open(log_path, 'w')

    num_epochs = args.epochs
    for epoch in range(1, num_epochs+1):
        if args.private:
            train_loss, train_acc = train(device, args, model, train_loader, optimizer, criterion, epoch, privacy_engine, scheduler)
        else:
            train_loss, train_acc = train(device, args, model, train_loader, optimizer, criterion, epoch, scheduler=scheduler)
        test_loss, test_acc = test(device, model, test_loader, criterion)
        print(f'{epoch}, {train_loss}, {train_acc},{test_loss},{test_acc}', file=log_fh)

    if args.save_model:
        os.makedirs(f"{args.output}/saved_models", exist_ok=True)
        model_save = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(model_save, f'{args.output}/saved_models/{model_name}.pth')

    log_fh.close()

if __name__ == '__main__':
    parser = construct_parser()
    args = parser.parse_args()
    main(args)