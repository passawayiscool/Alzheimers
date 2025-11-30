"""Script to train Simple LSTM model"""

import argparse
import logging
import sys
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cyberdefense.models import SimpleLSTMGenerator, VAEGANModel
from cyberdefense.preprocessing import EventTokenizer, create_dataloaders
from cyberdefense.utils import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--dataset-file", type=str, default="sequences.json")
    parser.add_argument("--tokenizer-dir", type=str, default="tokenizer")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory to save checkpoints")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train")
    args = parser.parse_args()

    config = load_config(args.config)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    
    # Load Tokenizer
    # Default to LANL tokenizer if not specified
    if args.tokenizer_dir == "tokenizer":
        tokenizer_path = Path(args.data_dir) / "tokenizer_lanl"
    else:
        tokenizer_path = Path(args.data_dir) / args.tokenizer_dir
        
    if not tokenizer_path.exists():
        logger.warning(f"Tokenizer not found at {tokenizer_path}, falling back to default")
        tokenizer_path = Path(args.data_dir) / "tokenizer"

    tokenizer = EventTokenizer.load(tokenizer_path)
    config["model"]["max_vocab_size"] = len(tokenizer.vocab)
    
    # Load Data
    # Default to Local Synthetic parquet if not specified
    if args.dataset_file == "sequences.json":
        data_path = Path(args.data_dir) / "local_train.parquet"
    else:
        data_path = Path(args.data_dir) / args.dataset_file
        
    logger.info(f"Loading data from {data_path}...")
    
    if data_path.suffix == ".parquet":
        import pandas as pd
        df = pd.read_parquet(data_path)
        sequences = []
        # Assuming data is already sorted by time in preprocess step
        # We need to group by user and host
        unique_pairs = df.groupby(["user", "host"]).size().sort_values(ascending=False).index
        
        # Limit to top 10 most active users to save memory
        max_users = 10
        logger.info(f"Processing top {max_users} active users out of {len(unique_pairs)}...")
        
        for i, (user, host) in enumerate(unique_pairs):
            if i >= max_users:
                break
            group = df[(df["user"] == user) & (df["host"] == host)]
            full_seq = group.to_dict(orient="records")
            
            # Chunk the sequence
            chunk_size = config["generation"]["max_sequence_length"]
            for j in range(0, len(full_seq), chunk_size):
                chunk = full_seq[j : j + chunk_size]
                if len(chunk) >= config["generation"]["min_sequence_length"]:
                    sequences.append(chunk)
                
        logger.info(f"Reconstructed {len(sequences)} sequences (chunked).")
    else:
        with open(data_path, "r") as f:
            sequences = json.load(f)
        
    train_loader, _, _ = create_dataloaders(
        sequences=sequences,
        tokenizer=tokenizer,
        batch_size=config["training"]["batch_size"],
        train_split=0.9,
        val_split=0.1
    )
    
    # Model Setup
    model = VAEGANModel(config)
    
    # Trainer
    from cyberdefense.training.trainer import VAEGANTrainer
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir else config["training"].get("checkpoint_dir", "checkpoints")
    trainer = VAEGANTrainer(model, config, device=device, checkpoint_dir=checkpoint_dir)
    
    # Training Loop
    logger.info("Starting training...")
    
    # Override config epochs with command line arg if provided
    num_epochs = args.epochs if args.epochs > 1 else config["training"]["num_epochs"]
    
    for epoch in range(num_epochs):
        metrics = trainer.train_epoch(train_loader, epoch)
        logger.info(f"Epoch {epoch+1}: {metrics}")
        
        # Save checkpoint
        if (epoch + 1) % config["training"]["save_every"] == 0:
            trainer.save_checkpoint(epoch+1, metrics)

if __name__ == "__main__":
    main()
