# src/agents/agent_core/utils/config.py
import json
from pathlib import Path

def load_project_config(config_path: Path) -> dict:
    """Load project configuration from rwagent.json"""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        return json.load(f)

def save_project_config(config_path: Path, data: dict):
    """Save updated project configuration"""
    with open(config_path, 'w') as f:
        json.dump(data, f, indent=2)

