import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # Dataset parameters
    parser.add_argument("--data_path", type=str, default="/home/vasquez/python/Dataset_ellipsoide.h5", help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--train_split", type=float, default=0.8, help="Percentage of data used for training")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay for optimizer")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for some optimizers (SGD)")
    parser.add_argument("--early_stop", type=int, default=16, help="Epochs without improvement")
    parser.add_argument("--model_path", type=str, default="/home/vasquez/python/beamforming/models", help="Path to save the trained model")

    # Mode selection (train/test)
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train", help="Train or Test Mode")

    # Optimizer & Scheduler
    parser.add_argument("--optimizer", type=str, choices=["adam", "sgd", "rmsprop"], default="adam", help="Optimizer selection")
    parser.add_argument("--scheduler", type=str, choices=["step", "cosine", "none"], default="none", help="Learning rate scheduler")
    
    # Misc
    parser.add_argument("--shuffle", type=bool, default=True, help="Shuffle dataset during training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # **Fix: Ignore unknown arguments that Jupyter adds**
    args, unknown = parser.parse_known_args()

    # return parser.parse_args()
    return args