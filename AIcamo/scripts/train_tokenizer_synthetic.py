"""Train a tokenizer on the synthetic local_train.parquet dataset"""

import sys
from pathlib import Path
import pandas as pd
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cyberdefense.preprocessing.event_tokenizer import EventTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    # Load synthetic dataset
    data_path = "data/processed/local_train.parquet"
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)

    logger.info(f"Loaded {len(df)} events")
    logger.info(f"Columns: {df.columns.tolist()}")

    # Convert to event dictionaries
    events = df.to_dict(orient="records")

    # Create tokenizer
    tokenizer = EventTokenizer()

    # Build vocab from events
    logger.info("Building vocabulary...")
    tokenizer.build_vocabulary(events, max_vocab_size=10000)

    # Save
    output_dir = "data/processed/tokenizer_synthetic"
    logger.info(f"Saving tokenizer to {output_dir}...")
    tokenizer.save(output_dir)

    logger.info("Done!")
    logger.info(f"Vocabulary size: {len(tokenizer.vocab)}")

    # Print sample tokens
    logger.info("\nSample tokens:")
    for i in range(min(30, len(tokenizer.vocab))):
        token = tokenizer.vocab.idx2token.get(i, "MISSING")
        logger.info(f"  {i}: {token}")

if __name__ == "__main__":
    main()
