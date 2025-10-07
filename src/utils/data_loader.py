import json
from pathlib import Path

def load_config(config_path: str):
    """
    Loads a helicopter configuration from a specified JSON file.
    """
    full_path = Path(config_path).resolve()
    print(f"[INFO] Successfully loaded configuration from: {full_path}")
    try:
        with open(full_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Config file not found at {full_path}")
        raise
    except json.JSONDecodeError:
        print(f"[ERROR] Could not decode JSON from {full_path}. Check for syntax errors.")
        raise

