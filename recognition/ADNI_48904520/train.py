import argparse

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

if __name__ == '__main__':
    main()