import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR

import numpy as np
from tqdm import tqdm
import argparse

from modules import Classifier
from dataset import get_dataloaders

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(model, train_loader, criterion, optimiser, scheduler, scaler, device):
    """Train for one epoch with progress bar."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimiser.zero_grad()

        # Mixed precision training
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Backpropagation with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()

        # Step scheduler (for OneCycleLR, step per batch)
        scheduler.step()

        # Track metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%',
            'lr': f'{scheduler.get_last_lr()[0]:.6f}'
        })

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy

def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validating'):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def main():
    parser = argparse.ArgumentParser(description='ConvNeXt for ADNI')

    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing AD_NC folder')
    parser.add_argument('--val_split', type=float, default=0.20,
                        help='Validation split ratio')

    # Model arguments
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Base learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Data loader workers')

    # Loss arguments
    parser.add_argument('--class_weight_nc', type=float, default=1.0,
                        help='Weight for NC class (to handle imbalance)')
    parser.add_argument('--class_weight_ad', type=float, default=2.0,
                        help='Weight for AD class')

    # Misc arguments
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory to save models')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Seed
    set_seed(args.seed)

    # Device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create model
    model = Classifier(
        num_classes=2,
        pretrained=True,
        dropout=args.dropout
    )

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split
    )

    # Class-weighted loss (important for imbalanced datasets)
    class_weights = torch.tensor([args.class_weight_nc, args.class_weight_ad]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimiser
    optimiser = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # OneCycleLR scheduler
    scheduler = OneCycleLR(
        optimiser,
        max_lr=args.lr * 5,  # Peak at 5x base learning rate
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.3,  # Warmup for 30% of training
        anneal_strategy='cos',
        div_factor=10,  # Initial lr = max_lr / 10
        final_div_factor=1e4  # Final lr = max_lr / 1e4
    )

    # Mixed precision scaler
    scaler = GradScaler()

    # Training history
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0.0

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 60)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimiser, scheduler, scaler, device
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  Current LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_path)
            print(f"  -> New best model saved! (Val Acc: {val_acc:.2f}%)")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved to: {os.path.join(args.save_dir, 'best_model.pth')}")
    print("=" * 60)


if __name__ == '__main__':
    main()