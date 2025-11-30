
import pandas as pd
from pathlib import Path
import sys
import logging

# Add src to path
# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cyberdefense.preprocessing.event_tokenizer import EventTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_tokenizer():
    # Input: The synthetic file we just generated
    data_path = "data/processed/local_train.parquet"
    save_path = "data/processed/tokenizer_lanl"
    
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    # Convert DataFrame to list of event dicts
    events = df.to_dict(orient="records")
    tokenizer = EventTokenizer()
    tokenizer.build_vocabulary(events, max_vocab_size=5000) # Larger vocab for real data
    
    logger.info(f"Saving tokenizer to {save_path}...")
    tokenizer.save(save_path)
    logger.info("Done.")

if __name__ == "__main__":
    train_tokenizer()
