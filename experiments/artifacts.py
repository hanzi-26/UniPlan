"""Read and bind the configuration of a resumable experiment."""

import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def bind_json(path, value):
    """Keep resumable measurements tied to the same inputs and settings."""
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Run settings changed at {path}; choose a new --output-dir")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)
