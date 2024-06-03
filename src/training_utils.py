import wandb
import torch
import logging

from torch.utils.data import DataLoader
from torch import DeviceObjType

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def optimal_model(model,
                  train_loader: DataLoader,
                  val_loader: DataLoader,
                  num_epochs: int,
                  patience: int,
                  device: DeviceObjType,
                  checkpoint_path,
                  optimizer,
                  criterion,
                  scheduler,
                  wandb_enabled: bool):

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        logger.info(f"RUNNING EPOCH {epoch + 1} ...")
        train_loss, train_acc = train_model(model = model,
                                            train_loader = train_loader,
                                            optimizer = optimizer,
                                            criterion = criterion,
                                            scheduler = scheduler,
                                            device = device)
        val_loss, val_acc = validate_model(model = model,
                                           val_loader = val_loader,
                                           criterion = criterion,
                                           device = device)
        logger.info(f"Training loss: {train_loss:.3}, Training accuracy: {train_acc:.3}")
        logger.info(f"Validation loss: {val_loss:.3}, Validation accuracy: {val_acc:.3}")

        if wandb_enabled:
            wandb.log({
                "train/loss":train_loss,
                "train/accuracy":train_acc,
                "val/loss":val_loss,
                "val/accuracy":val_acc
            })

        torch.save(model.state_dict(), checkpoint_path / f'epoch-{epoch + 1}.pth')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path / f'best.pth')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                logger.info(f"Validation loss did not imporove for {patience} epochs. Killing the training...")
                break
    model.load_state_dict(torch.load(checkpoint_path / f'best.pth'))


def train_model(model, train_loader: DataLoader, optimizer, criterion, scheduler, device):
    logger.info("Training phase ...")
    model.train()
    model.to(device)
    train_loss = 0
    train_accuracy = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        train_accuracy += calc_accuracy(labels, torch.argmax(input = outputs, dim = 1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
    train_loss /= len(train_loader)
    train_accuracy /= len(train_loader)
    return train_loss, train_accuracy

def validate_model(model, val_loader: DataLoader, criterion, device):
    logger.info("Validation phase ...")
    model.eval()
    model.to(device)
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_accuracy += calc_accuracy(labels, torch.argmax(input = outputs, dim = 1))

    val_loss /= len(val_loader)
    val_accuracy /= len(val_loader)
    return val_loss, val_accuracy

def test_model(model, test_loader: DataLoader, device):
    model.eval()
    model.to(device)
    test_accuracy = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_accuracy += calc_accuracy(labels, outputs.argmax(dim = 1))
    test_accuracy /= len(test_loader)

    return test_accuracy


def calc_accuracy(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    logger.info(f"Correct predictions: {correct} / {len(y_pred)}")
    return acc