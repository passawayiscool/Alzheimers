"""Script to preprocess collected event data"""

import argparse
import logging
import sys
import json
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cyberdefense.preprocessing import EventTokenizer, SequenceBuilder
from cyberdefense.utils import load_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Preprocess event data for training")
    parser.add_argument("--input", type=str, required=True, help="Input data directory")
    parser.add_argument(
        "--output", type=str, required=True, help="Output directory for processed data"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_config.yaml",
        help="Model configuration file",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    data_config = config["data"]
    model_config = config["model"]

    # Load raw data
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading data from {input_path}...")

    all_events = []

    # Load all parquet files
    for parquet_file in input_path.glob("*.parquet"):
        logger.info(f"Loading {parquet_file.name}...")
        df = pd.read_parquet(parquet_file)
        events = df.to_dict("records")
        all_events.extend(events)

    logger.info(f"Loaded {len(all_events)} events")

    # Build tokenizer
    logger.info("Building tokenizer...")
    tokenizer = EventTokenizer()
    tokenizer.build_vocabulary(all_events, max_vocab_size=model_config["max_vocab_size"])

    # Save tokenizer
    tokenizer_path = output_path / "tokenizer"
    tokenizer.save(tokenizer_path)
    logger.info(f"Tokenizer saved to {tokenizer_path}")

    # Build sequences
    logger.info("Building sequences...")
    sequence_builder = SequenceBuilder(
        max_sequence_length=data_config["max_sequence_length"],
        min_sequence_length=10,
        stride=64,
    )

    sequences = sequence_builder.build_sequences(all_events)
    logger.info(f"Built {len(sequences)} sequences")

    # Save sequences
    sequences_path = output_path / "sequences.json"
    with open(sequences_path, "w") as f:
        json.dump(sequences, f)
    logger.info(f"Sequences saved to {sequences_path}")

    # Save metadata
    metadata = {
        "num_events": len(all_events),
        "num_sequences": len(sequences),
        "vocab_size": len(tokenizer.vocab),
        "max_sequence_length": data_config["max_sequence_length"],
    }

    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Preprocessing complete!")
    logger.info(f"  Total events: {metadata['num_events']}")
    logger.info(f"  Total sequences: {metadata['num_sequences']}")
    logger.info(f"  Vocabulary size: {metadata['vocab_size']}")


if __name__ == "__main__":
    main()
