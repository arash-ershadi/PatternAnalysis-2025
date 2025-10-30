import os

import torch
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
import argparse

from modules import Classifier
from dataset import get_dataloaders

def load_model(model_path, device, dropout=0.5):
    """Load trained model from checkpoint."""
    print(f"Loading model from: {model_path}")

    # Create model
    model = Classifier(
        num_classes=2,
        pretrained=False,
        dropout=dropout
    )

    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model

def predict(model, data_loader, device):
    """Generate slice-level predictions."""
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='Predicting slices'):
            images = images.to(device)

            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def calculate_metrics(y_true, y_pred):
    """Calculate classification metrics."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix
    )

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }


def print_confusion_matrix(cm, class_names=None):
    """Print formatted confusion matrix."""
    if class_names is None:
        class_names = ['NC', 'AD']
    print("\nConfusion Matrix:")
    print(f"{'':>15} {'Predicted':>15}")
    print(f"{'':>15} {class_names[0]:>7} {class_names[1]:>7}")

    for i, class_name in enumerate(class_names):
        row = f"{'True ' + class_name:>15} "
        for j in range(len(class_names)):
            row += f"{cm[i][j]:>7}"
        print(row)

def main():
    parser = argparse.ArgumentParser(description='Predict with trained model')

    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing AD_NC folder')

    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability')

    # Testing arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Data loader workers')

    args = parser.parse_args()

    # Device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load model
    model = load_model(args.model_path, device, args.dropout)

    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)

    # Load test data
    print(f"Running slice-level predictions on test set...")
    _, _, test_loader = get_dataloaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split
    )

    preds, labels, probs = predict(model, test_loader, device)

    print(f"Total slices: {len(preds)}")

    # Calculate metrics
    metrics = calculate_metrics(labels, preds)

    print(f"\nSlice-Level Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")

    print_confusion_matrix(metrics['confusion_matrix'])

    print("=" * 60)

