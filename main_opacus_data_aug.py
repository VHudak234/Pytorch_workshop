import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import cross_entropy
from torchvision import datasets, transforms, models
from torchvision.transforms import v2
from torch.optim.lr_scheduler import StepLR
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

augmentation_pipeline = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
])

def train(args, model, device, train_loader, optimizer, epoch, privacy_engine):
    model.train()
    augmentations = args.augmult
    train_loss = 0
    correct = 0

    # Instantiate the loss criterion
    criterion = nn.CrossEntropyLoss(reduction='none')
    if args.private:
        print(args.private)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        batch_size = data.size(0)

        # Apply augmentations to each input image and concatenate
        augmented_data = torch.cat([augmentation_pipeline(data.clone().cpu()).to(device) for _ in range(augmentations)], dim=0)
        augmented_target = torch.cat([target] * augmentations, dim=0)

        # Forward pass and compute loss for all augmented examples
        output = model(augmented_data)
        loss = criterion(output, augmented_target)  # Use instantiated criterion here

        # Reshape loss and outputs back to original batch size
        loss = loss.view(augmentations, batch_size)
        per_example_loss = loss.mean(dim=0)

        # Backward pass and update parameters
        per_example_loss.mean().backward()
        optimizer.step()

        # Update training statistics
        train_loss += per_example_loss.sum().item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(augmented_target.view_as(pred)).sum().item()

        if batch_idx % args.log_interval == 0:
            epsilon = privacy_engine.get_epsilon(args.delta)
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {train_loss / len(data):.6f} '
                  f'(ε = {epsilon:.2f}, δ = {args.delta})')

    train_loss /= len(train_loader.dataset)
    accuracy = correct / len(train_loader.dataset)
    return train_loss, accuracy
    # model.train()
    # train_loss = 0
    # correct = 0
    # criterion = nn.CrossEntropyLoss()
    # for batch_idx, (data, target) in enumerate(train_loader):
    #     data, target = data.to(device), target.to(device)
    #     optimizer.zero_grad()
    #     output = model(data)
    #     loss = criterion(output, target)
    #     train_loss += loss.item() * len(data)
    #     pred = output.argmax(dim=1, keepdim=True)
    #     correct += pred.eq(target.view_as(pred)).sum().item()
    #     loss.backward()
    #     optimizer.step()
    #
    #     if batch_idx % args.log_interval == 0:
    #         epsilon = privacy_engine.get_epsilon(args.delta)
    #         print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} (ε = {:.2f}, δ = {})'.format(
    #             epoch, batch_idx * len(data), len(train_loader.dataset),
    #                    100. * batch_idx / len(train_loader), loss.item(), epsilon, args.delta))
    #
    # train_loss /= len(train_loader.dataset)
    # accuracy = correct / len(train_loader.dataset)
    # return train_loss, accuracy


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
            pred = output.argmax(dim=1, keepdim=True)
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
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--augmult', type=int, default=8, metavar='AM',
                        help='Set augmentation multiplicity')
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
    parser.add_argument('-i', '--input', required=True,
                        help='Path to the input data for the model to read')
    parser.add_argument('-o', '--output', required=True,
                        help='Path to the directory to write output to')
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
    # transform = v2.Compose([
    #     v2.RandomCrop(size=(22,22)),
    #     v2.RandomHorizontalFlip(p=0.3),
    #     v2.Normalize(cifar10_mean, cifar10_std)
    # ])
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(args.input, train=True, download=False, transform=transform),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(args.input, train=False, download=False, transform=transform),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model = models.resnet18(num_classes=10)
    model = ModuleValidator.fix(model).to(device)

    #TODO try momentum with values in range [0.3,0.9]
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    privacy_engine = PrivacyEngine(accountant='rdp')

    # Check if a checkpoint exists
    # checkpoint_path = os.path.join(args.output, 'checkpoint.pt')
    config_args = [str(vv) for kk, vv in vars(args).items()
                   if kk in ['epsilon', 'lr', 'gamma', 'seed']]
    model_name = '_'.join(config_args)

    start_epoch = 1
    best_loss = float('inf')
    # if os.path.exists(checkpoint_path):
    #     print(f'Loading checkpoint from {checkpoint_path}')
    #     checkpoint = torch.load(checkpoint_path)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    #     start_epoch = checkpoint['epoch'] + 1
    #     best_loss = checkpoint['best_loss']
    #     privacy_engine.load_state_dict(checkpoint['privacy_engine_state_dict'])
    #     privacy_engine.attach(model, optimizer, train_loader)
    # else:
    #     model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    #         module=model,
    #         optimizer=optimizer,
    #         data_loader=train_loader,
    #         epochs=args.epochs,
    #         target_epsilon=args.epsilon,
    #         target_delta=args.delta,
    #         max_grad_norm=args.max_grad_norm,
    #     )

    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=args.epochs,
        target_epsilon=args.epsilon,
        target_delta=args.delta,
        max_grad_norm=args.max_grad_norm,
    )
    os.makedirs(args.output, exist_ok=True)
    log_path = f'{args.output}/{model_name}.log'
    print(f"Opening log file at: {log_path}")
    log_fh = open(log_path, 'w')
    print('epoch,trn_loss,trn_acc,vld_loss,vld_acc', file=log_fh)
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = train(args, model, device, train_loader, optimizer, epoch, privacy_engine)
        test_loss, test_acc = test(args, model, device, test_loader)
        print(f'{epoch}, {train_loss}, {train_acc},{test_loss},{test_acc}', file=log_fh)
        scheduler.step()

        if test_loss < best_loss:
            best_loss = test_loss
            print(f"Saving new best model at epoch {epoch}")
            torch.save(model.state_dict(),
                       f"{args.output}/{model_name}.best.pt")
            print(f"Model saved at", f"{args.output}/{model_name}.best.pt")

    torch.save(model.state_dict(), f"/disk/scratch/s2209005/cifar10/output/{model_name}.final.pt")
    log_fh.close()
    print("Training complete!")


if __name__ == '__main__':
    parser = construct_parser()
    args = parser.parse_args()
    main(args)