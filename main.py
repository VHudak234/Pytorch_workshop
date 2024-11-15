import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator


def train(args, model, device, train_loader, optimizer, epoch, privacy_engine):
    model.train()
    train_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.item() * len(data)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            epsilon = privacy_engine.get_epsilon(args.delta)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} (ε = {:.2f}, δ = {})'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), epsilon, args.delta))

    train_loss /= len(train_loader.dataset)
    accuracy = correct / len(train_loader.dataset)
    return train_loss, accuracy


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item() * len(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    accuracy = correct / len(test_loader.dataset)
    return test_loss, accuracy

def construct_parser():
    parser = argparse.ArgumentParser(description='DP-SGD Opacus CIFAR10 Baseline')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=None, metavar='S',
                        help='random seed (default: random number)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging '
                             'training status')
    parser.add_argument('--epsilon', type=float, required=True,
                        help='Target epsilon for differential privacy')
    parser.add_argument('--delta', type=float, default=1e-5,
                        help='Target delta for differential privacy (default: 1e-5)')
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                        help='Maximum gradient norm for differential privacy')
    parser.add_argument('-i', '--input', required=True, help='Path to the '
                                                             'input data for the model to read')
    parser.add_argument('-o', '--output', required=True, help='Path to the '
                                                              'directory to write output to')
    return parser


def main(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.seed is None:
        args.seed = torch.randint(0, 2**32, (1,)).item()
        print(f'You did not set --seed, {args.seed} was chosen')
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(args.input, train=True, download=False, transform=transform),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(args.input, train=False, download=False, transform=transform),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model = models.resnet18(num_classes=10)
    model = ModuleValidator.fix(model).to(device)

    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    privacy_engine = PrivacyEngine()

    # Check if a checkpoint exists
    checkpoint_path = os.path.join(args.output, 'checkpoint.pt')
    start_epoch = 1
    best_loss = float('inf')
    if os.path.exists(checkpoint_path):
        print(f'Loading checkpoint from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
    else:
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=args.epochs,
            target_epsilon=args.epsilon,
            target_delta=args.delta,
            max_grad_norm=args.max_grad_norm,
        )

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = train(args, model, device, train_loader, optimizer, epoch, privacy_engine)
        test_loss, test_acc = test(args, model, device, test_loader)
        scheduler.step()

        if test_loss < best_loss:
            best_loss = test_loss
            print(f"Saving new best model at epoch {epoch}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
            }, checkpoint_path)

    print("Training complete!")


if __name__ == '__main__':
    parser = construct_parser()
    args = parser.parse_args()
    main(args)