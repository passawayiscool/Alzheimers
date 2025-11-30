
import json
import pandas as pd
import logging
from pathlib import Path
from tqdm import tqdm
import collections

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def preprocess_lanl_host_logs(input_path, output_path, max_lines=None, min_seq_len=10):
    """
    Reads LANL host logs (JSON lines), filters for process events,
    groups by User/Computer, and saves as Parquet for training.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing {input_path}...")
    
    # Mappings
    event_map = {
        4688: "PROCESS_START",
        4689: "PROCESS_END",
        # 4624: "LOGON",
        # 4634: "LOGOFF",
        # 4625: "LOGON_FAILURE",
        # 4672: "PRIVILEGE_ASSIGNED"
    }
    
    # Store sequences: keys = (UserName, LogHost), value = list of events
    sequences = collections.defaultdict(list)
    
    # First pass: Identify top users to avoid memory explosion?
    # Actually, let's just stream and keep all valid sequences in memory if possible, 
    # or flush periodically. For 14GB, we can't keep all in memory.
    # Strategy: Process chunk by chunk, or just focus on a subset for now.
    # Let's process the first N lines to get a good sample dataset.
    
    limit = max_lines if max_lines else 5_000_000 # Process first 5M lines (approx 1-2GB)
    
    count = 0
    with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in tqdm(f, total=limit, desc="Reading Lines"):
            if count >= limit:
                break
                
            try:
                record = json.loads(line)
                
                # Filter for relevant events
                eid = record.get("EventID")
                if eid not in event_map:
                    continue
                    
                # Extract fields
                user = record.get("UserName", "Unknown")
                host = record.get("LogHost", "Unknown")
                timestamp = record.get("Time", 0)
                
                # Build Event Dict
                event_type = event_map[eid]
                event = {
                    "timestamp": timestamp,
                    "event_type": event_type,
                    "user": user,
                    "host": host
                }
                
                # Add specific fields
                if eid == 4688:
                    event["process_name"] = record.get("ProcessName", "").lower()
                    event["parent_process"] = record.get("ParentProcessName", "").lower()
                elif eid == 4624:
                    event["logon_type"] = record.get("LogonType", "")
                    event["auth_package"] = record.get("AuthenticationPackage", "")
                
                # Key for sequencing
                key = (user, host)
                sequences[key].append(event)
                
            except Exception as e:
                continue
            
            count += 1

    logger.info(f"Processed {count} lines. Found {len(sequences)} unique User/Host pairs.")
    
    # Convert to DataFrame
    all_data = []
    
    for (user, host), events in sequences.items():
        # Sort by time
        events.sort(key=lambda x: x["timestamp"])
        
        # Filter short sequences
        if len(events) < min_seq_len:
            continue
            
        # Add to dataset
        for e in events:
            all_data.append(e)
            
    if not all_data:
        logger.warning("No valid sequences found!")
        return
        
    df = pd.DataFrame(all_data)
    logger.info(f"Saving {len(df)} events to {output_path}...")
    
    # Save
    df.to_parquet(output_path, index=False)
    logger.info("Done.")

if __name__ == "__main__":
    # Input: The wls file we found earlier
    wls_path = r"C:\Users\Xander\DF_PROJECT\LANLdataset\wls_day-02\wls_day-02"
    output_path = r"C:\Users\Xander\DF_PROJECT\data\processed\lanl_train.parquet"
    
    # Process first 200k lines for a fast demo training
    preprocess_lanl_host_logs(wls_path, output_path, max_lines=200_000)
