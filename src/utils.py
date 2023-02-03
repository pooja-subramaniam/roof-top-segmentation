from typing import Dict, Any
import json

def save_dict(d: Dict[str, Any], filename: str) -> None:
    with open(filename, 'w') as f:
        json.dump(d, f)
