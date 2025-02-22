import torch
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch import nn


def train(args, model, device, train_loader, optimizer, epoch, privacy_engine=None, scheduler=None):
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


def test(args, model, device, test_loader, optimizer=None):
    model.eval()
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