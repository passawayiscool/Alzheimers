
import pandas as pd
from pathlib import Path
import sys
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cyberdefense.preprocessing.event_tokenizer import EventTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_tokenizer():
    # Input: The detailed host network dataset
    data_path = "data/processed/host_network_train.parquet"
    save_path = "data/processed/tokenizer_host"
    
    if not Path(data_path).exists():
        logger.error(f"Dataset not found: {data_path}")
        return

    logger.info(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    # Convert DataFrame to list of event dicts
    events = df.to_dict(orient="records")
    tokenizer = EventTokenizer()
    
    # Build vocabulary
    # We use a larger vocab size to accommodate IPs and Paths
    logger.info("Building vocabulary...")
    tokenizer.build_vocabulary(events, max_vocab_size=20000) 
    
    logger.info(f"Saving tokenizer to {save_path}...")
    tokenizer.save(save_path)
    logger.info("Done.")

if __name__ == "__main__":
    train_tokenizer()
