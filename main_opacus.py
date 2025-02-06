import sys
import os
import argparse
import torch
from torch.utils.data import DataLoader, RandomSampler
import torch.nn as nn
import torch.optim as optim
import transformers
from torchvision import datasets, transforms, models
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import timm
from datasets import load_dataset, load_from_disk
from transformers import DataCollatorWithPadding, RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig
from opacus.utils.batch_memory_manager import BatchMemoryManager

def train(args, model, device, train_loader, optimizer, epoch, privacy_engine=None, scheduler=None):
    text = args.dataset == 'agnews'
    if text:
        model.train()
        criterion = nn.CrossEntropyLoss()
        train_loss = 0

        train_correct = 0
        train_samples = 0

        batch_loss = 0
        batch_correct = 0
        batch_samples = 0
        with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=args.batch_size,
            optimizer=optimizer
        ) as memory_safe_data_loader:
            for batch_idx, batch in enumerate(memory_safe_data_loader, start=1):
                optimizer.zero_grad()

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids = input_ids, attention_mask = attention_mask)
                logits = outputs.logits

                loss=criterion(logits, labels)
                loss.backward()

                optimizer.step()
                scheduler.step()

                predictions = torch.argmax(logits, dim=1)
                correct = (predictions == labels).sum().item()

                train_loss += loss.item()
                batch_loss += loss.item()
                batch_correct += correct
                batch_samples += labels.size(0)
                train_correct += correct
                train_samples += labels.size(0)

                if batch_idx % args.log_interval == 0:
                    avg_loss = batch_loss / args.log_interval
                    batch_accuracy = batch_correct / batch_samples * 100.
                    if args.private:
                        epsilon = privacy_engine.get_epsilon(args.delta)
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.2f}% (ε = {:.2f}, δ = {})'.format(
                                epoch,batch_idx * batch["labels"].size(0),len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                                avg_loss, batch_accuracy, epsilon, args.delta))
                    else:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.2f}%'.format(
                                epoch, batch_idx * batch["labels"].size(0), len(train_loader.dataset),
                                100. * batch_idx / len(train_loader), avg_loss, batch_accuracy))
                    batch_loss = 0
                    batch_correct = 0
                    batch_samples = 0
        train_loss /= len(train_loader)
        accuracy = train_correct / train_samples
        return train_loss, accuracy
    else:
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
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                if args.private:
                    if args.Dice:
                        epsilon = privacy_engine.get_privacy_spent()
                        print(epsilon, loss)
                    else:
                        epsilon = privacy_engine.get_epsilon(args.delta)
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} (ε = {:.2f}, δ = {})'.format(
                            epoch, batch_idx * len(data), len(train_loader.dataset),
                                100. * batch_idx / len(train_loader), loss.item(), epsilon, args.delta))
                else:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss.item()))
        train_loss /= len(train_loader.dataset)
        accuracy = correct / len(train_loader.dataset)
        return train_loss, accuracy


def test(args, model, device, test_loader, optimizer=None):
    model.eval()
    if args.dataset == 'agnews':
        test_loss = 0.0
        correct = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            with BatchMemoryManager(
                    data_loader=test_loader,
                    max_physical_batch_size=args.test_batch_size,
                    optimizer=optimizer
            ) as memory_safe_data_loader:
                for batch_idx, batch in enumerate(memory_safe_data_loader):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits

                    loss = criterion(logits, labels)
                    test_loss += loss.item() * labels.size(0)

                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == labels).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = correct / len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), 100.0 * accuracy))
        return test_loss, accuracy
    else:
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
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=None, metavar='S',
                        help='random seed (default: random number)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging '
                             'training status')
    parser.add_argument('--epsilon', type=float, default=5,
                        help='Target epsilon for differential privacy')
    parser.add_argument('--delta', type=float, default=1e-5,
                        help='Target delta for differential privacy (default: 1e-5)')
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                        help='Maximum gradient norm for differential privacy')
    parser.add_argument('--max-ef-norm', type=float, default=1.0,
                        help='Maximum norm for error feedback mechanism')
    parser.add_argument('-i', '--input', required=True,
                        help='Path to the input data for the model to read')
    parser.add_argument('-o', '--output', required=True,
                        help='Path to the directory to write output to')
    parser.add_argument('--private', type=int, default=1,
                        help='Enable differential privacy (Enabled by default)')
    parser.add_argument('--momentum', type=float, default=0.0,
                        help='Set momentum of optimizer')
    parser.add_argument('--Dice', type=int, default=0,
                        help='Enable error feedback mechanism for eliminating clipping bias (Disabled by default)')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='Choose dataset (cifar10 for image, ag news for text)')
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

    private = args.private
    dice = args.Dice
    text = args.dataset == 'agnews'

    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if args.dataset not in ['cifar10', 'agnews']:
        raise ValueError('Dataset selected needs to be cifar10 or agnews')

    if args.dataset == 'cifar10':
        train_loader = DataLoader(
            datasets.CIFAR10(args.input, train=True, download=False, transform=transform),
            batch_size=args.batch_size*4, shuffle=True, **kwargs)
        test_loader = DataLoader(
            datasets.CIFAR10(args.input, train=False, download=False, transform=transform),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    else:
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        data_collator = DataCollatorWithPadding(tokenizer)
        tokenized_agnews = load_from_disk(args.input)
        tokenized_agnews.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        train_loader = DataLoader(
            tokenized_agnews["train"], sampler=RandomSampler(tokenized_agnews["train"]), batch_size=args.batch_size, collate_fn=data_collator
        )
        test_loader = DataLoader(
            tokenized_agnews["test"], sampler=RandomSampler(tokenized_agnews["test"]), batch_size=args.test_batch_size, collate_fn=data_collator
        )

    if text:
        # TODO load pretrained transformer
        config = RobertaConfig(
            vocab_size=50265,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            num_labels=4
        )
        model = RobertaForSequenceClassification(config)
        model = ModuleValidator.fix(model)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        total_steps = len(train_loader) * args.epochs
        # TODO 10% warmup steps
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    else:
        model = models.resnet18(num_classes=10)
        model = ModuleValidator.fix(model)
        model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    config_args = [str(vv) for kk, vv in vars(args).items()
                   if kk in ['epsilon', 'lr', 'seed', 'momentum']]
    if private:
        model_name = '_'.join(config_args)
        if dice:
            model_name = 'Dice' + '_'.join(config_args)
        elif text:
            model_name = 'Text' + '_'.join(config_args)
    else:
        model_name = 'SGD' + '_'.join(config_args)
    start_epoch = 1
    best_loss = float('inf')

    os.makedirs(args.output, exist_ok=True)

    log_path = f'{args.output}/{model_name}.log'
    print(f"Opening log file at: {log_path}")

    log_fh = open(log_path, 'w')
    print('epoch,trn_loss,trn_acc,vld_loss,vld_acc', file=log_fh)
    if text:
        model.train()
        privacy_engine = PrivacyEngine(accountant='rdp')
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=args.epochs,
            target_epsilon=args.epsilon,
            target_delta=args.delta,
            max_grad_norm=args.max_grad_norm,
            poisson_sampling=False
        )

        for epoch in range(start_epoch, args.epochs + 1):
            train_loss, train_acc = train(args, model, device, train_loader, optimizer, epoch, privacy_engine, scheduler)
            test_loss, test_acc = test(args, model, device, test_loader, optimizer)
            print(f'{epoch}, {train_loss}, {train_acc},{test_loss},{test_acc}', file=log_fh)

            if test_loss < best_loss:
                best_loss = test_loss
                print(f"Saving new best model at epoch {epoch}")
                torch.save({
                    "state_dict": model.state_dict(),
                    "args": args
                }, f"{args.output}/{model_name}.best.pt")
                print(f"Model saved at", f"{args.output}/{model_name}.best.pt")
    # if dice:
        # trans_cifar = transforms.Compose([transforms.Resize(224),transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
        # trans_cifar_train = transforms.Compose([transforms.Resize(224), transforms.RandomHorizontalFlip(), CIFAR10Policy(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
        # dataset_train = datasets.CIFAR10(args.input, train=True, download=True, transform=trans_cifar_train)
        # dataset_test = datasets.CIFAR10(args.input, train=False, download=True, transform=trans_cifar)
        # train_loader = torch.utils.data.DataLoader(dataset_train,batch_size=args.batch_size,shuffle=True,drop_last=False, pin_memory = True)
        # test_loader = torch.utils.data.DataLoader(dataset_test,batch_size=args.batch_size*2,shuffle=False,drop_last=False, pin_memory = False)
        # DiceSGD(model, train_loader, test_loader, args.batch_size, args.batch_size, args.batch_size/2, args.epochs, args.max_grad_norm, args.max_ef_norm, device, args.lr, 'sgd', log_fh)
        # privacy_engine = PrivacyEngine_Dice(
        #     model,
        #     batch_size=args.batch_size,
        #     sample_size=args.batch_size,
        #     epochs=args.epochs,
        #     target_epsilon=args.epsilon,
        #     max_grad_norm=args.max_grad_norm,
        #     error_max_grad_norm=args.max_ef_norm,
        #     loss_reduction='mean',
        #     clipping_fn='Abadi'
        # )
        # privacy_engine.attach_dice(optimizer)
        # privacy_engine = PrivacyEngine(accountant='rdp')
        # optimizer = DiceSGDOptimizer(optimizer, noise_multiplier=0.1, max_grad_norm=args.max_grad_norm, max_ef_norm=args.max_ef_norm, expected_batch_size=args.batch_size)
        # model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        #     module=model,
        #     optimizer=optimizer,
        #     data_loader=train_loader,
        #     epochs=args.epochs,
        #     target_epsilon=args.epsilon,
        #     target_delta=args.delta,
        #     max_grad_norm=args.max_grad_norm,
        # )
        # for epoch in range(start_epoch, args.epochs + 1):
        #     train_loss, train_acc = train(args, model, device, train_loader, optimizer, epoch, privacy_engine)
        #     test_loss, test_acc = test(args, model, device, test_loader)
        #     print(f'{epoch}, {train_loss}, {train_acc},{test_loss},{test_acc}', file=log_fh)
        #
        #     if test_loss < best_loss:
        #         best_loss = test_loss
        #         print(f"Saving new best model at epoch {epoch}")
        #         torch.save({
        #             "state_dict": model.state_dict(),
        #             "args": args
        #         }, f"{args.output}/{model_name}.best.pt")
        #         print(f"Model saved at", f"{args.output}/{model_name}.best.pt")
    elif private:
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

        for epoch in range(start_epoch, args.epochs + 1):
            train_loss, train_acc = train(args, model, device, train_loader, optimizer, epoch, privacy_engine)
            test_loss, test_acc = test(args, model, device, test_loader)
            print(f'{epoch}, {train_loss}, {train_acc},{test_loss},{test_acc}', file=log_fh)

            if test_loss < best_loss:
                best_loss = test_loss
                print(f"Saving new best model at epoch {epoch}")
                torch.save({
                    "state_dict": model.state_dict(),
                    "args": args
                }, f"{args.output}/{model_name}.best.pt")
                print(f"Model saved at", f"{args.output}/{model_name}.best.pt")
    else:
        for epoch in range(start_epoch, args.epochs + 1):
            train_loss, train_acc = train(args, model, device, train_loader, optimizer, epoch)
            test_loss, test_acc = test(args, model, device, test_loader)
            print(f'{epoch}, {train_loss}, {train_acc},{test_loss},{test_acc}', file=log_fh)

            if test_loss < best_loss:
                best_loss = test_loss
                print(f"Saving new best model at epoch {epoch}")
                torch.save({
                    "state_dict": model.state_dict(),
                    "args": args
                }, f"{args.output}/{model_name}.best.pt")
                print(f"Model saved at", f"{args.output}/{model_name}.best.pt")


    torch.save(model.state_dict(), f"/disk/scratch/s2209005/cifar10/output/{model_name}.final.pt")
    log_fh.close()
    print("Training complete!")


if __name__ == '__main__':
    parser = construct_parser()
    args = parser.parse_args()
    main(args)